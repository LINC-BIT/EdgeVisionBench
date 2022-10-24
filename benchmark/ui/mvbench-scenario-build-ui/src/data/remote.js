export const getAvailableDatasets = () => {
  const mockData = [
    {
      title: 'Image Classification',
      value: 'Image Classification',
      children: [
        {
          title: 'Generic Object',
          value: 'Generic Object',
          children: [
            { title: 'CIFAR10', value: 'CIFAR10' },
            { title: 'STL10', value: 'STL10' },
          ]
        },
        {
          title: 'Digit and Letter',
          value: 'Digit and Letter',
          children: [
            { title: 'MNIST', value: 'MNIST' },
            { title: 'EMNIST', value: 'EMNIST' },
          ]
        },
      ],
    },
    {
      title: 'Object Detection',
      value: 'Object Detection',
      children: [
        {
          title: 'Generic Object',
          value: 'Generic Object',
          children: [
            { title: 'COCO2017', value: 'COCO2017' },
            { title: 'VOC2012', value: 'VOC2012' },
          ]
        },
        {
          title: 'Face Mask',
          value: 'Face Mask',
          children: [
            { title: 'WI_Mask', value: 'WI_Mask' },
            { title: 'MakeML_Mask', value: 'MakeML_Mask' },
          ]
        },
      ],
    },
    {
      title: 'Semantic Segmentation',
      value: 'Semantic Segmentation',
      children: [
        {
          title: 'Autonomous Driving',
          value: 'Autonomous Driving',
          children: [
            { title: 'Cityscapes', value: 'Cityscapes' },
            { title: 'GTA5', value: 'GTA5' },
          ]
        },
        {
          title: 'Person',
          value: 'Person',
          children: [
            { title: 'BaiduPerson', value: 'BaiduPerson' },
            { title: 'SuperviselyPerson', value: 'SuperviselyPerson' },
          ]
        },
      ],
    },
    {
      title: 'Action Recognition',
      value: 'Action Recognition',
      children: [
        {
          title: 'Web Video',
          value: 'Web Video',
          children: [
            { title: 'UCF101', value: 'UCF101' },
            { title: 'HMDB51', value: 'HMDB51' },
            { title: 'IXMAS', value: 'IXMAS' },
          ]
        }
      ],
    },
  ];

  return new Promise(resolve => {
    setTimeout(() => {
      resolve(mockData);
    }, 1000);
  });
}

export const getAvailableModels = () => {
  const mockData = [
    {
      title: 'Image Classification',
      value: 'Image Classification',
      children: [
        { title: 'ResNet-18', value: 'ResNet-18' },
        { title: 'ResNet-50', value: 'ResNet-50' },
        { title: 'SENet-18', value: 'SENet-18' },
        { title: 'SENet-50', value: 'SENet-50' },
        { title: 'WideResNet', value: 'WideResNet' },
        { title: 'MobileNetV2', value: 'MobileNetV2' },
        { title: 'ResNeXt', value: 'ResNeXt' },
        { title: 'ResNet-18 (CBAM)', value: 'ResNet-18 (CBAM)' },
      ],
    },
    {
      title: 'Object Detection',
      value: 'Object Detection',
      children: [
        { title: 'YOLOV3', value: 'YOLOV3' },
        { title: 'YOLOX', value: 'YOLOX' },
        { title: 'RetinaNet', value: 'RetinaNet' },
        { title: 'SSD', value: 'SSD' },
      ],
    },
    {
      title: 'Semantic Segmentation',
      value: 'Semantic Segmentation',
      children: [
        { title: 'FCN', value: 'FCN' },
        { title: 'DeepLabV3', value: 'DeepLabV3' }
      ],
    },
    {
      title: 'Action Recognition',
      value: 'Action Recognition',
      children: [
        { title: 'TSN', value: 'TSN' },
        { title: 'TRN', value: 'TRN' }
      ],
    },
  ];

  return new Promise(resolve => {
    setTimeout(() => {
      resolve(mockData);
    }, 1000);
  });
}

export const getAvailableModelCompressionAlgs = () => {
  const mockData = [
    {
      title: 'Structure Pruning',
      value: 'Structure Pruning',
      children: [
        { title: 'L1 Filter Pruning', value: 'L1 Filter Pruning' },
        { title: 'L2 Filter Pruning', value: 'L2 Filter Pruning' },
        { title: 'FPGM', value: 'FPGM' },
      ],
    },
  ];

  return new Promise(resolve => {
    setTimeout(() => {
      resolve(mockData);
    }, 1000);
  });
}

export const getAvailableDAAlgs = () => {
  const mockData = [
    {
      title: 'Close Set DA',
      value: 'Close Set DA',
      children: [
        { title: 'EATA', value: 'EATA' },
        { title: 'CoTTA', value: 'CoTTA' },
        { title: 'Tent', value: 'Tent' },
        { title: 'SHOT', value: 'SHOT' },
        { title: 'CUA', value: 'CUA' },
      ],
    },
    {
      title: 'Partial DA',
      value: 'Partial DA',
      children: [
        { title: 'PADA', value: 'PADA' },
        { title: 'ETN', value: 'ETN' },
        { title: 'ARPDA', value: 'ARPDA' }
      ],
    },
    {
      title: 'Open Set DA',
      value: 'Open Set DA',
      children: [
        { title: 'OSDABP', value: 'OSDABP' },
        { title: 'ROS', value: 'ROS' },
        { title: 'Inheritune', value: 'Inheritune' }
      ],
    },
    {
      title: 'Universal DA',
      value: 'Universal DA',
      children: [
        { title: 'UDA', value: 'UDA' }
      ],
    },
  ];

  return new Promise(resolve => {
    setTimeout(() => {
      resolve(mockData);
    }, 1000);
  });
}

export const getAvailableHpSearchAlg = () => {
  const mockData = [
    { title: 'Grid Search', value: 'Grid Search' },
    { title: 'Random Search', value: 'Random Search' },
  ];

  return new Promise(resolve => {
    setTimeout(() => {
      resolve(mockData);
    }, 1000);
  });
}

export const getAvailableMetrics = () => {
  const mockData = [
    { title: 'Model accuracy before starting re-training on all domains', value: 'Model accuracy before starting re-training on all domains' },
    { title: 'Model accuracy before re-training on each domain', value: 'Model accuracy before re-training on each domain' },
    { title: 'Model accuracy after re-training on each domain', value: 'Model accuracy after re-training on each domain' },
    { title: 'Training loss', value: 'Training loss' },
    { title: 'Time usage on each domain', value: 'Time usage on each domain' },
    { title: 'Energy consumption on each domain', value: 'Energy consumption on each domain' },
    { title: 'CPU utilization', value: 'CPU utilization' },
    { title: 'GPU utilization', value: 'GPU utilization' },
    { title: 'I/O utilization', value: 'I/O utilization' },
    { title: 'RAM', value: 'RAM' },
    { title: 'VRAM', value: 'VRAM' },
  ];

  return new Promise(resolve => {
    setTimeout(() => {
      resolve(mockData);
    }, 1000);
  });
}


export const getBuildScenarioRes = ({ sourceDatasets, targetDomains, daSettingOnLabelSpace }) => {
//   const mockData = {
//     classesRenameMap: {
//       CIFAR10: {
//         automobile: 'car'
//       }
//     },

//     targetSourceRelationshipMap: {
//       CIFAR10: {
//         STL10: 'Image Corruptions\nShifts',
//         ImageNet: 'Image\nCorruptions\nShifts',
//         MNIST: 'Image Corruptions Shifts'
//       },
//       EMNIST: {
//         ImageNet: 'Image Corruptions Shifts'
//       }
//     },

//     classesInEachDatasetMap: {
//       CIFAR10: {
//         knownClasses: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse'],
//         unknownClasses: ['ship', 'truck'],
//         discardedClasses: []
//       } 
//     },

//     indexClassMap: {
//       0: 'airplane',
//       1: 'car'
//     },

//     generatedCode: `build_scenario_manually(
//     source_datasets_name=['SVHN', 'CIFAR10'], 
//     target_datasets_order=['MNIST', 'STL10', 'USPS', 'MNIST', 'STL10', 'USPS'], 
//     da_mode='da',
//     num_samples_in_each_target_domain=100
// )`
//   }
  
  sourceDatasets = [...sourceDatasets]
  sourceDatasets.sort()
  targetDomains = [...new Set(targetDomains)]
  targetDomains.sort()
  daSettingOnLabelSpace = {
    'Close Set DA': 'da',
    'Partial DA': 'partial_da',
    'Open Set DA': 'open_set_da',
    'Universal DA': 'universal_da'
  }[daSettingOnLabelSpace]

  return new Promise(resolve => {
    import(`./hard_data/demo_hard_data/${sourceDatasets.join('+')}|${targetDomains.join('+')}|${daSettingOnLabelSpace}.json`).then(res => {
      resolve(res);
    }).catch(e => {
      console.log('hard json load error')
      resolve({
        'targetSourceRelationshipMap': {},
        'classesInEachDatasetMap': {
          'error': true
        },
        'indexClassMap': {}
      });
    })
  })

  // return new Promise(resolve => {
  //   setTimeout(() => {
  //     resolve(mockData);
  //   }, 1000);
  // });
}


const getDevDatasetPreviewImagesInfo = ({ datasetName, classesName, numImagesPerClass }) => {
  const res = [];

  // const devImgSizeMap = {
  //   32: ['CIFAR-10', 'MNIST', 'EMNIST', 'STL-10'] 
  // }
  // const imgSize = Object.keys(devImgSizeMap).filter(k => devImgSizeMap[k].includes(datasetName))[0];
  // console.log(classesName)
  for (const className of classesName) {
    for (let imgId = 1; imgId <= numImagesPerClass; imgId++) {
      res.push({
        url: `https://edgevisionbench-1258477338.cos.ap-beijing.myqcloud.com/demo_dataset_imgs/${datasetName}/${className}_${imgId}.png`, // from image bed
        class: className
      })
    }
  }

  return res;
}

export const getDatasetPreviewImagesInfo = ({ datasetName, classesName, numImagesPerClass }) => {
  // {datasetName}_{className}_{img_size}_{id}.png
  const mockData = getDevDatasetPreviewImagesInfo({ datasetName, classesName, numImagesPerClass });
  return new Promise(resolve => {
    setTimeout(() => {
      resolve(mockData);
    }, 100);
  });
}
