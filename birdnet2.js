importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest')

main()
async function main() {
    const navigatorLang = new URL(location.href).searchParams.get('lang')
    await tf.setBackend('webgl')
    const BirdNetJS = await predictModel()
    postMessage({ message: 'warmup', progress: 70 })
    await BirdNetJS.warmup()
    postMessage({ message: 'load_geomodel', progress: 90 })
    const areaModel = await tf.loadGraphModel('/birdnet-web/models/birdnet/area-model/model.json')
    postMessage({ message: 'load_labels', progress: 95 })
    const supportedLanguages = ['af', 'da', 'en_us', 'fr', 'ja', 'no', 'ro', 'sl', 'tr', 'ar', 'de', 'es',
        'hu', 'ko', 'pl', 'ru', 'sv', 'uk', 'cs', 'en_uk', 'fi', 'it', 'nl', 'pt', 'sk', 'th', 'zh']
    const lang = supportedLanguages.find(l => l.startsWith(navigatorLang.split('-')[0])) || 'en_us'
    const birdsList     = (await fetch('/birdnet-web/models/birdnet/labels/en_us.txt').then(r => r.text())).split('\n')
    const birdsListI18n = (await fetch(`/birdnet-web/models/birdnet/labels/${lang}.txt`).then(r => r.text())).split('\n')
    const birds = new Array(remap.length)
    for (let i = 0; i < remap.length; i++) {
        birds[i] = {
            geoscore: 1,
            name:     remap[i] === -1 ? '' : birdsList[remap[i]].split('_')[1],
            nameI18n: remap[i] === -1 ? '' : birdsListI18n[remap[i]].split('_')[1],
        }
    }
    postMessage({ message: 'loaded' })
    const MIN_AUDIO_CONFIDENCE = 0.1
    const MIN_AREA_CONFIDENCE = 0.1
    onmessage = async function({ data }) {
        if (data.message === 'predict') {
            const predictionList = await BirdNetJS.predict(tf.tensor(data.pcmAudio, BirdNetJS.shape))
            const prediction = []
            for (let i = 0; i < predictionList.length; i++) {
                const confidence = predictionList[i]
                if (confidence > MIN_AUDIO_CONFIDENCE && birds[i].geoscore > MIN_AREA_CONFIDENCE) {
                    prediction.push({ ...birds[i], confidence })
                }
            }
            postMessage({ message: 'predict', prediction })
        }
        if (data.message === 'area-scores') {
            tf.engine().startScope()
            const startOfYear = new Date(new Date().getFullYear(), 0, 1);
            startOfYear.setDate(startOfYear.getDate() + (1 - (startOfYear.getDay() % 7)))
            const week = Math.round((new Date() - startOfYear) / 604800000) + 1
            const areaTensor = tf.tensor([[data.latitude, data.longitude, week]])
            const areaScores = await areaModel.predict(areaTensor).data()
            tf.engine().endScope()
            for (let i = 0; i < birds.length; i++) {
                birds[i].geoscore = remap[i] === -1 ? 0 : areaScores[remap[i]]
            }
            postMessage({ message: 'area-scores' })
        }
    }
}

async function predictModel() {
    const BirdNetJS = await tf.loadGraphModel('models/birdnet_v2/model.json', {
        onProgress: (progress) => postMessage({ message: 'load_model', progress: progress * 70 | 0 })
    })
    const MatMulBuf = new Float32Array(await fetch('models/birdnet_v2/MatMul.bin').then(r => r.arrayBuffer()))
    const matMulWeights = tf.tensor(MatMulBuf, [128, MatMulBuf.length / 128])

    async function predict(signal) {
        const stft = tf.engine().runKernel('STFT', { signal, frameLength: 512, frameStep: 128 })
        signal.dispose()
        const spec = tf.matMul(stft, matMulWeights, false, true)
        stft.dispose()
        const spectrogram = tf.engine().runKernel('STFT_normalize', { spec })
        spec.dispose()
        const spectrogram4d = tf.expandDims(spectrogram, 0)
        spectrogram.dispose()
        const spectrogramR = tf.transpose(spectrogram4d, [0, 2, 1, 3])
        spectrogram4d.dispose()
        const resTensor = BirdNetJS.predict(spectrogramR)
        spectrogramR.dispose()
        const result = await resTensor.array()
        resTensor.dispose()
        return result[0]
    }
    const shape = [1, 512 + 128 * 511]
    return {
        shape,
        async warmup() {
            await predict(tf.zeros(shape))
        },
        predict
    }

}

tf.registerKernel({
    kernelName: 'STFT',
    backendName: 'webgl',
    kernelFunc: ({ backend, inputs: { signal, frameLength, frameStep } }) => {
        const innerDim = frameLength / 2
        const batch = (signal.size - frameLength + frameStep) / frameStep | 0
        let currentTensor = backend.runWebGLProgram({
            variableNames: ['x'],
            outputShape: [batch, frameLength],
            userCode: `
            void main() {
                ivec2 coords = getOutputCoords();
                int p = coords[1] % ${innerDim};
                int k = 0;
                for (int i = 0; i < ${Math.log2(innerDim)}; ++i) {
                    if ((p & (1 << i)) != 0) { k |= (1 << (${Math.log2(innerDim) - 1} - i)); }
                }
                int i = 2 * k;
                if (coords[1] >= ${innerDim}) {
                    i = 2 * (k % ${innerDim}) + 1;
                }
                int q = coords[0] * ${frameLength} + i;
                float val = getX((q / ${frameLength}) * ${frameStep} + q % ${frameLength});
                float cosArg = ${2.0 * Math.PI / frameLength} * float(q);
                float mul = 0.5 - 0.5 * cos(cosArg);
                setOutput(val * mul);
            }`
        }, [signal], 'float32')
        for (let len = 1; len < innerDim; len *= 2) {
            let prevTensor = currentTensor
            currentTensor = backend.runWebGLProgram({
                variableNames: ['x'],
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
                    if (i < ${innerDim}) { // real
                        float evenK_re = getX(batch, baseIndex);
                        setOutput(evenK_re + (oddK_re * a - oddK_im * b) * float(highSign));
                    } else { // imaginary
                        float evenK_im = getX(batch, baseIndex + ${innerDim});
                        setOutput(evenK_im + (oddK_re * b + oddK_im * a) * float(highSign));
                    }
                }`
            }, [currentTensor], 'float32')
            backend.disposeIntermediateTensorInfo(prevTensor)
        }
        const real = backend.runWebGLProgram({
            variableNames: ['x'],
            outputShape: [batch, innerDim + 1],
            userCode: `void main() {
                ivec2 coords = getOutputCoords();
                int batch = coords[0];
                int i = coords[1];
                int zI = i % ${innerDim};
                int conjI = (${innerDim} - i) % ${innerDim};
                float Zk0 = getX(batch, zI);
                float Zk1 = getX(batch, zI+${innerDim});
                float Zk_conj0 = getX(batch, conjI);
                float Zk_conj1 = -getX(batch, conjI+${innerDim});
                float t = ${-2 * Math.PI} * float(i) / float(${innerDim * 2});
                float diff0 = Zk0 - Zk_conj0;
                float diff1 = Zk1 - Zk_conj1;
                float real = (Zk0 + Zk_conj0 + cos(t) * diff1 + sin(t) * diff0) * 0.5;
                float imag = (Zk1 + Zk_conj1 - cos(t) * diff0 + sin(t) * diff1) * 0.5;
                setOutput(sqrt(real * real + imag * imag));
            }`
        }, [currentTensor], 'float32')
        backend.disposeIntermediateTensorInfo(currentTensor)
        return real
    }
})
tf.registerKernel({
    kernelName: 'STFT_normalize',
    backendName: 'webgl',
    kernelFunc: ({ backend, inputs: { spec } }) => {
        return backend.runWebGLProgram({
            variableNames: ['x'],
            outputShape: [...spec.shape, 3],
            userCode: `void main() {
                ivec3 coords = getOutputCoords();
                if (coords[2] == 0) {
                    setOutput(floor((max(min(log(max(getX(coords[0], ${spec.shape[1] - 1} - coords[1]), 1.000000013351432e-10)) * 0.4342944622039795 * 20.0 - 3.6737916469573975, 0.0), -100.0) + 100.0) * 0.009999999776482582 * 255.0));
                } else {
                    setOutput(0.0);
                }
            }`
        }, [spec], 'float32')
    }
})

const remap = [-1,6063,1702,1709,1707,1713,1701,1703,1708,1711,5314,292,1190,1887,1883,1884,309,307,306,308,726,729,728,727,1803,1799,1800,1801,183,5885,140,5593,5591,3392,3390,3388,267,258,261,629,5588,
3471,1418,757,3272,3526,4162,4119,4117,4116,4108,4114,4113,4307,4305,4310,4308,3942,4071,1894,1473,1471,1471,886,883,885,1833,1843,4010,4011,3491,700,1058,1854,6298,6296,3016,3015,3017,4321,4422,4283,
643,2428,2430,4136,2355,2356,5856,1631,166,164,1489,1490,1491,4268,4278,4276,4271,4270,4277,4280,4275,5755,5750,5747,5746,5753,1147,1497,1499,1502,1501,1495,1419,2491,3189,3188,6470,6466,6467,6469,2608,
1690,1689,1691,5917,1958,1957,3661,2458,1073,1062,1412,1414,4598,1439,1445,1440,2209,2208,1282,829,5794,5795,2786,2788,1721,1719,1715,1720,3316,1265,1267,1266,3305,3969,5511,2831,5860,3972,4409,375,378,
377,382,380,373,376,981,972,973,970,967,966,965,3964,3965,5697,676,5759,1141,1143,425,423,428,422,107,1830,4392,4394,4396,4402,4399,1466,1468,1470,1467,5491,85,3547,2212,3028,466,465,893,894,5476,5480,
5481,5479,1807,661,924,194,196,195,2848,2081,1212,1211,5221,5217,5216,5219,5214,1670,3208,4226,4227,3692,187,451,446,448,449,4859,4860,2420,2421,2376,2373,4854,4855,190,5210,6458,3574,386,1629,3118,3119,
3113,3114,3115,452,5021,368,2597,771,770,2789,2791,5251,5249,2643,2644,2638,4755,4752,4753,1182,1168,1179,1171,1166,6364,6355,6353,6358,6349,1183,1176,1163,2991,2990,649,3958,3954,3955,3225,3222,3217,
3218,5423,5424,2409,2408,2413,82,83,6158,6160,6153,6162,6152,6159,6151,6157,6156,485,486,852,853,854,864,862,861,867,717,1078,5326,1275,1274,1273,1276,3196,3198,3101,3094,3091,3096,3106,3097,3089,3089,
3105,3104,3098,3093,3099,5337,4054,5723,5724,2456,2828,1208,5714,5717,5716,5980,5978,5979,4472,5867,5865,4769,4765,4763,87,4057,2280,2451,2449,293,3858,4724,3544,6019,4945,6464,704,703,2974,2977,3961,
3968,2541,2045,2053,2052,2047,803,804,750,467,468,470,473,4211,2075,3172,4341,2073,5614,3924,5635,5634,2780,39,21,31,35,24,26,26,1367,1371,3584,2670,2666,2929,2496,5330,796,801,5336,4216,2493,790,788,
791,787,779,782,795,784,783,6315,6315,6315,4157,4141,4145,4143,4154,4158,5000,2612,3431,3413,3417,3423,3421,3428,3419,3424,3414,3432,3415,3278,5130,5128,749,742,743,746,2528,2519,2522,2520,2521,2513,2524,
2530,2531,5886,3558,567,569,568,1307,1306,1305,1304,5769,5765,5760,5767,5768,5773,5764,5770,5766,536,533,537,534,97,96,99,98,3915,3914,3918,3921,2270,4421,4418,4420,6196,6198,6197,6206,6188,6194,6204,
6203,6203,6195,6200,6192,6201,6321,4000,4002,3535,3529,2851,3638,3640,650,651,2077,2215,152,1838,4285,2662,6079,3406,3403,1126,1220,1221,3933,3981,3649,3648,2399,2989,4979,4993,4995,4994,4987,4968,957,
5490,613,609,5065,5061,5228,5224,5229,5227,5225,5230,5226,2995,4616,5611,5612,5609,5610,3444,3454,3449,3451,3448,3447,3459,3462,3442,3445,4606,4605,6452,1881,1876,1878,1879,1985,1990,1988,1991,1975,1994,
1973,1981,1974,1288,1288,916,913,909,1941,4633,4632,4639,4636,1996,1998,1997,1056,1057,1048,1464,1463,1460,1455,1458,1457,997,2750,3557,3555,3556,3553,985,2899,3582,1846,3583,2299,2297,2289,2298,2295,
2294,5013,5016,5011,5010,3288,3764,737,731,734,733,4648,4649,4650,218,203,202,198,200,206,201,211,1774,2268,2267,455,4125,435,440,439,438,436,6017,5003,5002,5005,5001,5004,4689,4694,1804,2884,662,3333,
5918,5338,5999,6009,5994,5996,6010,6002,5997,6015,5990,5991,5300,2035,5986,5987,2969,5158,3831,3830,1909,3820,2755,2763,3575,2340,1962,1968,1967,1966,1969,1961,1970,2874,2878,2876,1102,1101,1099,1092,
5166,5165,3809,3813,3810,5410,4317,3805,3806,3798,4810,5408,3815,3818,142,141,2650,2652,3819,2620,2866,6408,3484,1527,1525,2578,2562,2572,2564,2570,1695,2855,3824,3823,2585,5069,5386,3233,72,5270,3543,
2214,3797,2078,5466,5444,5451,5436,5453,5449,5435,5472,5468,5445,5467,2332,2333,2331,2336,1157,5418,5419,5417,5421,5580,1857,1858,2539,1856,3862,1870,1867,1868,1869,6432,6437,6434,6444,6438,6436,1891,
929,926,3149,3142,6421,6421,6422,5919,2388,3244,4460,5853,5851,1889,1423,6039,622,620,620,623,617,618,410,5810,3157,4374,4373,383,562,558,1661,1657,1656,1651,1120,5842,5824,5831,5819,5840,5843,5816,5815,
5829,5818,5828,5847,5826,5841,6299,3884,372,1200,1201,1202,1198,1199,3151,3151,3382,3383,3321,3324,1084,1083,4670,5334,5202,3238,4902,4901,6068,5390,5394,5395,3064,4197,4188,4056,5951,4672,4712,4709,4713,
5864,3600,3174,3176,4527,1614,1613,3767,3281,3282,4043,2734,2725,2741,2729,2732,4791,4794,6088,6084,1431,6094,6096,6095,6095,5179,2799,3788,4102,921,922,5788,3402,3399,256,4384,984,6302,3756,3754,3759,
2060,2068,2071,2064,2057,2055,2061,2063,5510,5506,6480,6482,2284,2953,3761,3122,3616,1534,1538,1535,1539,1533,1540,1430,2132,2139,2128,2138,2135,2134,2140,2136,2130,2131,2133,5381,5382,5383,5172,1482,
5245,601,602,604,605,5563,5321,3732,3731,3725,3722,3723,3726,3724,3733,3326,4684,4685,3412,3789,3792,3790,3750,3752,3751,3125,6309,6305,6314,6306,6313,6312,6307,6308,6311,4910,326,2501,4323,1559,3023,
5082,1796,1797,2862,2859,6405,6403,6209,4204,4205,4207,6388,6373,6385,6396,6389,6375,6401,6387,6381,6378,6400,6398,6383,6390,6395,6379,6382,6372,4090,4086,4082,4099,4097,521,5948,5284,5273,5275,5282,1911,
2886,5962,3070,3073,3082,3077,3066,3078,3074,3083,4337,1780,888,889,4999,1763,1757,1764,1765,1751,1755,2623,1749,1748,405,404,409,408,2442,1791,6328,6329,1866,1864,4604,4603,4601,4602,3953,3952,3952,5176,
5175,1592,1585,1575,1595,1581,1579,1578,1577,4352,1725,4331,3274,4778,4777,4772,4771,4774,4780,4779,4775,1745,1747,639,637,638,636,635,4239,4237,4236,615,5261,3604,3611,3303,144,145,2405,2401,2172,839,
3476,4213,4135,4870,4877,4880,4889,4875,4885,4879,1395,1391,2934,2796,2793,62,68,66,69,54,3250,3254,3245,3407,1951,4756,4757,4759,5325,5870,5874,4905,4903,4906,5700,5159,2802,1853,4346,4344,2742,2888,
2891,2890,710,5145,5134,5142,5136,4578,4531,4554,4551,4547,4587,4579,4541,4541,4552,4591,4562,4586,4559,4583,4533,4532,4593,6342,5966,1128,1129,0,4499,2811,2814,2808,2813,88,89,4920,5801,5803,1737,1733,
1734,1736,1735,1730,1739,1293,1152,5554,6447,6520,6518,6511,1771,4842,4840,4837,2180,2179,5402,4298,160,157,6169,6166,2772,2773,3135,3231,491,493,492,488,489,2441,2436,1572,5256,5255,5253,5565,5575,5574,
5564,5567,1115,1112,1113,3559,5231,4816,4821,4813,4814,4819,4815,4820,4812,5341,3565,3564,3563,1038,6180,6186,6184,6185,6186,6183,6181,1411,1410,1408,1409,6054,6048,942,931,940,4436,4440,4438,4433,4441,
4439,4437,4432,6053,6050,6051,951,949,950,6340,2748,2746,1828,1828,1827,1826,1335,2544,2542,5783,5782,2546,49,45,3486,2031,3393,6113,6118,6115,6110,6117,6114,6112,4072,3593,3590,3592,3587,3591,5545,5544,
5543,3702,3699,3697,3703,3701,2980,1025,1030,1035,1026,1037,1032,2850,6285,6265,6243,6253,6235,6246,6223,6220,6274,6239,6244,6254,6275,6247,6233,6222,6242,6257,6238,6277,6229,6266,6283,3680,3685,1105,
1544,1550,1546,2219,2176,713,3109,3306,3307,3309,3794,3795,3793,3753,5922,5924,2311,2304,2316,2312,2305,4478,4483,4482,3657,5377,5378,5374,5375,4038,4031,698,697,5100,4407,1902,1905,1904,3171,3169,1343,
1354,2964,4366,3263,3264,2194,186,4914,4245,4248,4252,4357,3668,3671,3675,3669,3674,3666,361,339,357,366,340,363,360,364,352,343,348,2370,2371,1234,2234,2239,1436,1435,1001,4642,5182,3206,2648,2649,2647,
1215,3226,7,3293,3297,3294,994,5504,5500,5628,5625,5626,5621,5629,5622,5618,842,843,844,5308,4723,2091,2094,2097,2098,2103,2112,1253,4360,4364,4359,4361,4358,231,230,229,516,514,5639,5638,5636,5640,5637,
244,1263,837,498,500,511,506,502,5642,4258,2992,2993,6489,6490,6488,6491,6487,526,525,4850,237,238,239,4257,1059,1060,3514,3513,3512,3520,3515,3519,137,138,4655,4657,4656,573,2902,6410,1950,5777,5776,
5776,3137,221,1018,4922,4925,4928,4926,822,818,823,816,817,819,2927,2926,2910,2924,2906,2914,2913,2911,2920,124,3633,3632,1947,1948,2227,2228,5208,5204,5205,5207,5203,2885,1728,2540,123,127,1295,5475,
2707,4229,4230,6366,6367,3626,4911,3221,4075,3131,3128,3129,3130,3132,3133,4058,2484,2479,2488,2483,2481,2489,5518,5534,5524,5536,5516,5513,5532,5525,5515,5522,5530,5529,5535,5514,5528,5531,5519,5521,
5520,5523,5526,5537,5527,5517,5538,655,658,654,652,653,660,3779,3782,3785,3781,3786,3774,3778,986,987,989,988,3746,3744,4677,4682,4680,4679,4675,2636,2634,1016,1015,991,993,4429,4430,2591,1776,1778,1775,
1750,4260,4259,4262,4265,4261,5631,4232,3579,2204,5876,3061,5239,5234,296,2014,4662,5248,6030,6033,6032,5895,5965,1931,4488,4489,1946,5305,4851,5548,5551,2121,2124,6406,5669,5651,5672,5674,5655,1608,1453,
6055,3488,532,5360,5353,5344,5350,5348,5358,5357,5346,5351,5349]