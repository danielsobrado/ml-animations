export const steps = [
    {
        id: 1,
        title: 'Tokenization & Embeddings',
        path: '/step1',
        description: 'How text becomes numbers',
        component: 'Step1Tokenization'
    },
    {
        id: 2,
        title: 'Positional Encoding',
        path: '/step2',
        description: 'Adding position information',
        component: 'Step2Positional'
    },
    {
        id: 3,
        title: 'Multi-Head Attention',
        path: '/step3',
        description: 'The core of transformers',
        component: 'Step3Attention'
    },
    {
        id: 4,
        title: 'Feed-Forward Network',
        path: '/step4',
        description: 'Processing each position',
        component: 'Step4FFN'
    },
    {
        id: 5,
        title: 'Layer Norm & Residuals',
        path: '/step5',
        description: 'Stable training',
        component: 'Step5Norm'
    },
    {
        id: 6,
        title: 'Full Architecture',
        path: '/step6',
        description: 'Putting it all together',
        component: 'Step6Architecture'
    },
    {
        id: 7,
        title: 'Weight Tying',
        path: '/step7',
        description: 'Optimization: Parameter sharing',
        component: 'Step7WeightTying'
    },
    {
        id: 8,
        title: 'Training Optimizations',
        path: '/step8',
        description: 'Making training efficient',
        component: 'Step8Training'
    },
    {
        id: 9,
        title: 'Inference Optimizations',
        path: '/step9',
        description: 'Fast generation',
        component: 'Step9Inference'
    },
];
