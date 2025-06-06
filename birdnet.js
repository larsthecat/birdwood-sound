importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest')
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu/dist/tf-backend-webgpu.js')
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/tf-tflite.min.js')
tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/')

async function main() {
    const backend = new URL(location.href).searchParams.get('backend')
    globalThis.useFastFFT = new URL(location.href).searchParams.get('fast_fft') === 'on'
    console.log('WORKER | backend:', backend, 'useFastFFT:', useFastFFT)
    if (backend) {
        await tf.setBackend(backend)
    } else {
        await tf.ready()
    }

    const areaModelPromise = tflite.loadTFLiteModel('models/birdnet/area-model.tflite')
    const BirdNetJS = await tf.loadLayersModel('models/birdnet/model.json', {
        weightPathPrefix: 'models/birdnet/',
        onProgress: (progress) => {
            postMessage({ message: 'loading', progress })
        }
    })
    postMessage({ message: 'warmup' })
    tf.engine().startScope()
    await BirdNetJS.predict(tf.zeros([1, 144000]), { batchSize: 1 }).data()
    tf.engine().endScope()
    const areaModel = await areaModelPromise
    postMessage({ message: 'loaded' })
    onmessage = async function({ data }) {
        if (data.message === 'predict') {
            const audioChunkTensor = tf.tensor(data.audioBuf, [data.batchSize, 144000])
            const prediction = await BirdNetJS.predict(audioChunkTensor).data()
            audioChunkTensor.dispose()
            postMessage({ message: 'predict', prediction })
        }
        if (data.message === 'area-scores') {
            const areaTensor = tf.tensor([[data.latitude, data.longitude, data.week]])
            const areaScores = await areaModel.predict(areaTensor).data()
            areaTensor.dispose()
            postMessage({ message: 'area-scores', areaScores })
        }
    }
}

main()

class MelSpecLayerSimple extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.sampleRate = config.sampleRate
        this.specShape = config.specShape
        this.frameStep = config.frameStep
        this.frameLength = config.frameLength
        this.melFilterbank = tf.tensor2d(config.melFilterbank)
    }
    build(inputShape) {
        this.magScale = this.addWeight(
            'magnitude_scaling',
            [],
            'float32',
            tf.initializers.constant({ value: 1.23 })
        );
        super.build(inputShape)
    }
    computeOutputShape(inputShape) {
        return [inputShape[0], this.specShape[0], this.specShape[1], 1];
    }
    call(inputs) {
        return tf.tidy(() => {
        inputs = inputs[0]
        return tf.stack(inputs.split(inputs.shape[0]).map((input) => {
                let spec = input.squeeze()
                spec = tf.sub(spec, tf.min(spec, -1, true))
                spec = tf.div(spec, tf.max(spec, -1, true).add(0.000001))
                spec = tf.sub(spec, 0.5)
                spec = tf.mul(spec, 2.0)
                if (globalThis.useFastFFT === false) {
                    spec = tf.signal.stft(spec, this.frameLength, this.frameStep, this.frameLength, tf.signal.hannWindow)
                    spec = tf.cast(spec, 'float32')
                } else {
                    spec = stft(spec, this.frameLength, this.frameStep, this.frameLength, tf.signal.hannWindow)
                }
                spec = tf.matMul(spec, this.melFilterbank)
                spec = spec.pow(2.0)
                spec = spec.pow(tf.div(1.0, tf.add(1.0, tf.exp(this.magScale.read()))))
                spec = tf.reverse(spec, -1)
                spec = tf.transpose(spec)
                spec = spec.expandDims(-1)
                // spec = spec.expandDims(0)
                return spec;
            }))
        })
    }
    static get className() { return 'MelSpecLayerSimple' }
}
tf.serialization.registerClass(MelSpecLayerSimple)

function stft(signal, frameLength, frameStep, fftLength, windowFn) {
    const framedSignal = tf.engine().runKernel('FRAME', {input: signal, frameLength, frameStep })
    const input = tf.mul(framedSignal, windowFn(frameLength))
    let innerDim = input.shape[input.shape.length - 1]
    const batch = input.size / innerDim
    const realValues = tf.engine().runKernel('FFT2', {input: tf.reshape(input, [batch, innerDim])})
    const half = Math.floor(innerDim / 2) + 1
    const realComplexConjugate = tf.split(
        realValues, [half, innerDim - half],
        realValues.shape.length - 1)
    const outputShape = input.shape.slice()
    outputShape[input.shape.length - 1] = half
    return tf.reshape(realComplexConjugate[0], outputShape)
}

tf.registerKernel({
    kernelName: 'FFT2',
    backendName: 'webgl',
    kernelFunc: ({ backend, inputs: { input } }) => {
        const innerDim = input.shape[input.shape.length - 1] / 2
        const batch = tf.util.sizeFromShape(input.shape) / innerDim / 2
        let currentTensor = backend.runWebGLProgram({
            variableNames: ['mapvalue'],
            outputShape: [batch, innerDim * 2],
            userCode: `
void main() {
  ivec2 coords = getOutputCoords();
  int p = coords[1] % ${innerDim};
  int k = 0;
  for (int i = 0; i < ${Math.log2(innerDim)}; ++i) {
    if ((p & (1 << i)) != 0) { k |= (1 << (${Math.log2(innerDim) - 1} - i)); }
  }
  if (coords[1] < ${innerDim}) {
    setOutput(getMapvalue(coords[0], 2 * k));
  } else {
    setOutput(getMapvalue(coords[0], 2 * (k % ${innerDim}) + 1));
  }
}`
        }, [input], 'float32')
        for (let len = 1; len < innerDim; len *= 2) {
            let prevTensor = currentTensor
            currentTensor = backend.runWebGLProgram({
                variableNames: [`x`],
                outputShape: [batch, innerDim * 2],
                userCode: `void main() {
ivec2 coords = getOutputCoords();
int batch = coords[0];
int i = coords[1];
int k = i % ${innerDim};
int isHigh = (k % ${len * 2}) / ${len};
int highSign = (1 - isHigh * 2);
int baseIndex = k - isHigh * ${len};
float t = ${Math.PI / len} * float(k % ${len});
float a = cos(t);
float b = sin(-t);
float oddK_re = getX(batch, baseIndex + ${len});
float oddK_im = getX(batch, baseIndex + ${len + innerDim});
if (i / ${innerDim} == 0) { // real
    float evenK_re = getX(batch, baseIndex);
    float outp = evenK_re + (oddK_re * a - oddK_im * b) * float(highSign);
    setOutput(outp);
} else { // imaginary
    float evenK_im = getX(batch, baseIndex + ${innerDim});
    float outp = evenK_im + (oddK_re * b + oddK_im * a) * float(highSign);
    setOutput(outp);
}
}` }, [currentTensor], 'float32')
            backend.disposeIntermediateTensorInfo(prevTensor)
        }

        let prevTensor = currentTensor
        currentTensor = backend.runWebGLProgram({
            variableNames: ['x'],
            outputShape: [batch, innerDim * 2],
            userCode: `
void main() {
    ivec2 coords = getOutputCoords();
    int i = coords[1];
    int batch = coords[0];

    int k = i <= ${innerDim} ? i : ${innerDim * 2} - i;
    int zI = k % ${innerDim};
    int conjI = (${innerDim} - k) % ${innerDim};
    float Zk0 = getX(batch, zI);
    float Zk_conj0 = getX(batch, conjI);
    float t = ${-2 * Math.PI} * float(k) / float(${innerDim * 2});
    float result = (Zk0 + Zk_conj0 + cos(t) * (getX(batch, zI+${innerDim}) + getX(batch, conjI+${innerDim})) + sin(t) * (Zk0 - Zk_conj0)) * 0.5;
    setOutput(result);
}`
        }, [currentTensor], 'float32')
        backend.disposeIntermediateTensorInfo(prevTensor)
        return currentTensor
    }
})
tf.registerKernel({
    kernelName: 'FRAME',
    backendName: 'webgl',
    kernelFunc: ({ backend, inputs: { input, frameLength, frameStep } }) => {
        const outpLen = (input.size - frameLength + frameStep) / frameStep | 0
        
        return backend.runWebGLProgram({
            variableNames: ['x'],
            outputShape: [outpLen, frameLength],
            userCode: `
    void main() {
    ivec2 coords = getOutputCoords();
    int j = coords[1];
    int b = coords[0];
    int i = b * ${frameLength} + j;
    setOutput(getX((i / ${frameLength}) * ${frameStep} + i % ${frameLength}));
    }`
        }, [input], 'float32')
    }
})
function arrayProduct (arr) {
    let product = 1;
    for (let i = 0; i < arr.length; i++) { product *= arr[i] }
    return product;
}
function flatDispatchLayout(shape) { return {x: shape.map((d, i) => i)} }
function computeDispatch(layout, outputShape, workgroupSize = [1, 1, 1], elementsPerThread = [1, 1, 1]) {
return [Math.ceil(arrayProduct(layout.x.map(d => outputShape[d])) /(workgroupSize[0] * elementsPerThread[0])),
    layout.y ? Math.ceil(arrayProduct(layout.y.map(d => outputShape[d])) / (workgroupSize[1] * elementsPerThread[1])) : 1,
    layout.z ? Math.ceil(arrayProduct(layout.z.map(d => outputShape[d])) / (workgroupSize[2] * elementsPerThread[2])) : 1]
}

if (!globalThis.tf) { globalThis.tf = {} }
tf.registerKernel?.({
    kernelName: 'FFT2',
    backendName: 'webgpu',
    kernelFunc: ({ backend, inputs: { input } }) => {
        const innerDim = input.shape[input.shape.length - 1] / 2
        const batch = tf.util.sizeFromShape(input.shape) / innerDim / 2
        const workgroupSize = [64, 1, 1]
        const dispatchLayout = flatDispatchLayout([batch, innerDim * 2])
        const dispatch = computeDispatch(dispatchLayout, [batch, innerDim * 2], workgroupSize, [2, 1, 1])
        let currentTensor = backend.runWebGPUProgram({
            variableNames: ['X'],
            outputShape: [batch, innerDim * 2],
            workgroupSize,
            shaderKey: `fft_permut_${innerDim}`,
            dispatchLayout,
            dispatch,
            getUserCode: () => `
fn main(index: i32) {
let batch = index / ${innerDim};
let p = index % ${innerDim};
let outIndexReal = batch * ${innerDim * 2} + p;
let outIndexImag = outIndexReal + ${innerDim};
var k = 0;
for (var i: u32 = 0; i < ${Math.log2(innerDim)}; i = i + 1) {
    if ((p & (1 << i)) != 0) { k |= (1 << (${Math.log2(innerDim) - 1} - i)); }
}
setOutputAtIndex(outIndexReal, getX(batch, 2 * k));
setOutputAtIndex(outIndexImag, getX(batch, 2 * (k % ${innerDim}) + 1));
}`
        }, [input], 'float32')
        for (let len = 1; len < innerDim; len *= 2) {
            let prevTensor = currentTensor
            currentTensor = backend.runWebGPUProgram({
                variableNames: [`value`],
                outputShape: [batch, innerDim * 2],
                workgroupSize,
                shaderKey: `fft_step_${innerDim}_${len}`,
                dispatchLayout,
                dispatch,
                getUserCode: () => `fn main(index: i32) {
                    let batch = index / ${innerDim};
                    let i = index % ${innerDim};
                    let outIndexReal = batch * ${innerDim * 2} + i;
                    let outIndexImag = outIndexReal + ${innerDim};
                    let k = i % ${innerDim};
                    let isHigh = (k % (${len} * 2)) / ${len};
                    let highSign = (1 - isHigh * 2);
                    let baseIndex = k - isHigh * ${len};
                    let t = ${Math.PI} / f32(${len}) * f32(k % ${len});
                    let a = cos(t);
                    let b = sin(-t);
                    let oddK_re = getValue(batch, baseIndex + ${len});
                    let oddK_im = getValue(batch, baseIndex + ${len} + ${innerDim});

                    let evenK_re = getValue(batch, baseIndex);
                    let outpR = (evenK_re + (oddK_re * a - oddK_im * b) * f32(highSign));
                    setOutputAtIndex(outIndexReal, outpR);
                    let evenK_im = getValue(batch, baseIndex + ${innerDim});
                    let outpI = (evenK_im + (oddK_re * b + oddK_im * a) * f32(highSign));
                    setOutputAtIndex(outIndexImag, outpI);
                    }`
                }, [currentTensor], 'float32')
            backend.disposeData(prevTensor.dataId)
        }
        let prevTensor = currentTensor
        currentTensor = backend.runWebGPUProgram({
            variableNames: ['x'],
            outputShape: [batch, innerDim * 2],
            workgroupSize,
            shaderKey: `fft_post_${innerDim}`,
            dispatchLayout,
            dispatch: computeDispatch(flatDispatchLayout([batch, innerDim * 2]), [batch, innerDim * 2], workgroupSize, [1, 1, 1]),
            getUserCode: () => `
fn main(index: i32) {
    let coords = getOutputCoords();
    let i = coords[1];
    let batch = coords[0];
    var k = i;
    if (i > ${innerDim}) {
      k = ${innerDim * 2} - i;
    }
    let zI = k % ${innerDim};
    let conjI = (${innerDim} - k) % ${innerDim};
    let Zk0 = getX(batch, zI);
    let Zk_conj0 = getX(batch, conjI);
    let t = ${-2 * Math.PI} * f32(k) / f32(${innerDim * 2});
    let result = (Zk0 + Zk_conj0 + cos(t) * (getX(batch, zI+${innerDim}) + getX(batch, conjI+${innerDim})) + sin(t) * (Zk0 - Zk_conj0)) * 0.5;
    setOutputAtIndex(index, result);
}`
        }, [currentTensor], 'float32')
        backend.disposeData(prevTensor.dataId)
        return currentTensor
    }
})
tf.registerKernel({
    kernelName: 'FRAME',
    backendName: 'webgpu',
    kernelFunc: ({ backend, inputs: { input, frameLength, frameStep } }) => {
        const workgroupSize = [64, 1, 1]
        const outpLen = (input.size - frameLength + frameStep) / frameStep | 0
        const dispatchLayout = flatDispatchLayout([outpLen, frameLength])
        return backend.runWebGPUProgram({
            variableNames: ['x'],
            outputShape: [outpLen, frameLength],
            workgroupSize,
            shaderKey: `frame_${frameLength}_${frameStep}`,
            dispatchLayout,
            dispatch: computeDispatch(dispatchLayout, [outpLen, frameLength], workgroupSize),
            getUserCode: () => `
    fn main(i: i32) {
        setOutputAtIndex(i, getX((i / ${frameLength}) * ${frameStep} + i % ${frameLength}));
    }`
        }, [input], 'float32')
    }
})