import { Form, TreeSelect, Card, Button, Carousel, Typography, Steps, InputNumber, Radio, Alert, message, Space } from 'antd';
import { ConsoleSqlOutlined, LoadingOutlined, WindowsFilled } from '@ant-design/icons';
import React, { useEffect, useState, useRef } from 'react';
import {
  getAvailableDatasets,
  getAvailableModels,
  getAvailableModelCompressionAlgs,
  getBuildScenarioRes,
  getAvailableDAAlgs,
  getAvailableHpSearchAlg,
  getAvailableMetrics
} from '../data/remote';
import TargetDomainsPreview from './TargetDomainsPreview';
import BuildingStepsVis from './BuildingStepsVis';
import {
  CarTwoTone,
  SettingTwoTone,
  FileImageTwoTone,
  FileTextTwoTone,
  FundTwoTone
} from '@ant-design/icons';

const { TreeNode } = TreeSelect;
const { Title, Paragraph, Text } = Typography;

const edgeVisionApplications = [
  'Image Classification',
  'Object Detection',
  'Semantic Segmentation',
  'Action Recognition'
];

const forDemoScreenshot = false;
const blankSpace = (n) => forDemoScreenshot ? new Array(n).fill(0).map(_ => <div align='center' style={{color: 'white'}}>1</div>) : <></>

export default function DataSettings() {
  const [availableDatasets, setAvailableDatasets] = useState(null);
  const [availableModels, setAvailableModels] = useState(null);
  const [availableModelCompressionAlgs, setAvailableModelCompressionAlgs] = useState(null);
  const [availableDAAlgs, setAvailableDAAlgs] = useState(null);
  const [availableHpSearchAlgs, setAvailableHpSearchAlgs] = useState(null);
  const [availableMetrics, setAvailableMetrics] = useState(null);

  const [edgeVisionApplication, setEdgeVisionApplication] = useState(edgeVisionApplications[0]);
  const [sourceDatasets, setSourceDatasets] = useState([]);
  const [autoRandomizeTargetDomains, setAutoRandomizeTargetDomains] = useState(true);
  const [manualNumTargetDomains, setManualNumTargetDomains] = useState(30);
  const [manualTargetDomains, setManualTargetDomains] = useState([]);
  const [targetDomains, setTargetDomains] = useState([]);
  // const [numTargetSamples, setNumTargetSamples] = useState(100);
  const [daSettingOnLabelSpace, setDASettingOnLabelSpace] = useState('Close Set DA');

  const [buildingScenario, setBuildingScenario] = useState(false);
  const [buildingScenarioResData, setBuildingScenarioResData] = useState({
    classesRenameMap: {},
    targetSourceRelationshipMap: {},
    classesInEachDatasetMap: {},
    indexClassMap: {},
    generatedCode: ''
  });

  const [model, setModel] = useState(null);
  const [modelCompressionAlg, setModelCompressionAlg] = useState(null);
  const [DAAlg, setDAAlg] = useState(null);
  const [hpSearchAlg, setHpSearchAlg] = useState(null);
  const [metrics, setMetrics] = useState([]);

  const [evaluating, setEvaluating] = useState(false);

  const carousel = useRef(null);

  const convertAvailableDatasetsToSourceDatasetsTreeValue = (availableDatasets) => {
    if (availableDatasets === null) {
      return null;
    }

    // firstly enter this function
    if (edgeVisionApplications.includes(availableDatasets[0].title)) {
      availableDatasets = availableDatasets.filter(
        availableDataset => availableDataset.title === edgeVisionApplication)
      if (availableDatasets.length === 0) {
        return null;
      }
      return availableDatasets[0].children
    }

    return availableDatasets.map(availableDataset => ({
      ...availableDataset,
      selectable: availableDataset.children === undefined,
      children: availableDataset.children ?
        convertAvailableDatasetsToSourceDatasetsTreeValue(availableDataset.children) :
        undefined,
    }))
  }

  const convertCommonTreeDataToTreeValue = (availableDatasets, topFilterKeyword) => {
    if (availableDatasets === null) {
      return null;
    }

    if (topFilterKeyword) {
      availableDatasets = availableDatasets.filter(
        availableDataset => availableDataset.title === topFilterKeyword)

      if (availableDatasets.length === 0) {
        return null;
      }
      return availableDatasets[0].children
    }

    return availableDatasets.map(availableDataset => ({
      ...availableDataset,
      selectable: availableDataset.children === undefined,
      children: availableDataset.children ?
        convertAvailableDatasetsToSourceDatasetsTreeValue(availableDataset.children) :
        undefined,
    }))
  }

  const autoRandomizeTargetDomainsFunc = (manualTargetDomains, manualNumTargetDomains) => {
    if (manualTargetDomains.length === 0) {
      return [];
    }

    if (manualTargetDomains.length === 1) {
      return new Array(manualNumTargetDomains).fill(manualTargetDomains[0]);
    }

    const res = [];
    for (let i = 0; i < manualNumTargetDomains; i++) {
      const lastDomain = res.length === 0 ? '' : res[res.length - 1];
      let randomDomain = '';
      while (true) {
        randomDomain = manualTargetDomains[parseInt(Math.random() * (manualTargetDomains.length - 0) + 0)]
        if (randomDomain !== lastDomain) {
          break;
        }
      }
      res.push(randomDomain);
    }
    console.log(res)
    return res;
  }

  const convertAvailableDatasetsToTargetDomainTreeValue = (availableDatasets) => {
    if (availableDatasets === null) {
      return null;
    }

    // firstly enter this function
    if (edgeVisionApplications.includes(availableDatasets[0].title)) {
      availableDatasets = availableDatasets.filter(
        availableDataset => availableDataset.title === edgeVisionApplication)[0].children
    }

    const getDatasetAppearTimes = (arr, item) => {
      return arr.reduce((prev, curr) => curr === item ? prev + 1 : prev, 0);
    }

    return availableDatasets.map(availableDataset => ({
      ...availableDataset,
      selectable: false,
      title: availableDataset.children ?
        availableDataset.title :
        <Button onClick={() => setTargetDomains([...targetDomains, availableDataset.value])} size='small'>
          {`${availableDataset.title} (appears ${getDatasetAppearTimes(targetDomains, availableDataset.value)} times)`}
        </Button>,
      children: availableDataset.children ?
        convertAvailableDatasetsToTargetDomainTreeValue(availableDataset.children) :
        undefined,
    }))
  }

  const datasetSelectDisabled = availableDatasets === null;

  const checkBeforeBuilding = () => {
    if (sourceDatasets.length === 0) {
      return [true, 'Error: Please select at least 1 source dataset in 2.3 Source Datasets!']
    } else if (targetDomains.length === 0) {
      return [true, 'Error: Please select at least 1 target domain in 2.4 Target Domains!']
    }

    const getCommonElemsInTwoArr = (arr1, arr2) => {
      const arr1Map = {};
      arr1.forEach(v => arr1Map[v] = 1);
      return [...new Set(arr2.filter(v => arr1Map[v]))];
    }

    const commonDatasets = getCommonElemsInTwoArr(sourceDatasets, targetDomains);
    if (commonDatasets.length > 0) {
      return [true, `Error: ${commonDatasets.length} datasets (${commonDatasets.join(', ')}) are both in source and target domains!`]
    }
    return [false, 'Data settings are legal!']
  }
  const [buildDisabled, noteBeforeBuilding] = checkBeforeBuilding();

  const buildScenario = () => {
    setBuildingScenario(true);

    getBuildScenarioRes({ sourceDatasets, targetDomains, daSettingOnLabelSpace }).then(res => {
      console.log('build', sourceDatasets, targetDomains, daSettingOnLabelSpace)
      setBuildingScenarioResData(res);
      setBuildingScenario(false);
      // message.success('Build successfully!')
    })
  }

  useEffect(() => {
    getAvailableDatasets().then(res => {
      setAvailableDatasets(res);
    })

    getAvailableModels().then(res => {
      setAvailableModels(res);
    })

    getAvailableModelCompressionAlgs().then(res => {
      setAvailableModelCompressionAlgs(res);
    })

    getAvailableDAAlgs().then(res => {
      setAvailableDAAlgs(res);
    })

    getAvailableHpSearchAlg().then(res => {
      setAvailableHpSearchAlgs(res);
    })

    getAvailableMetrics().then(res => {
      setAvailableMetrics(res);
    })
  }, []);

  return (
    <>
      <Carousel ref={carousel} dots={false} afterChange={() => { window.scrollTo(0, 0) }}>
        <div>
          <Title level={2}><FileTextTwoTone /> 1. Introduction </Title>
          <Paragraph>
            EdgeVisionBench is built as a workflow from data and domain preparation to final evaluation. 
            In this user interface, every small steps in the workflow are detailed. 
          </Paragraph>
          <Paragraph>
            You can follow the instructions to conduct your own evaluation. Click "Next" to continue!
          </Paragraph>

          <Paragraph></Paragraph>
          <div align='center'>
            <Button type='primary' onClick={() => carousel.current.next()}>Next</Button>
          </div>
        </div>

        <div>
          <Title level={2}><FileImageTwoTone /> 2. Data and domain preparation </Title>
          <Title level={3}>2.1 DA setting (shifts in the label space)</Title>
          {/* <Paragraph>
            (xxx).
          </Paragraph> */}
          <Radio.Group onChange={e => setDASettingOnLabelSpace(e.target.value)} value={daSettingOnLabelSpace}>
            <Radio value='Close Set DA'>Close Set DA</Radio>
            <Radio value='Partial DA'>Partial DA</Radio>
            <Radio value='Open Set DA'>Open Set DA</Radio>
            <Radio value='Universal DA'>Universal DA</Radio>
          </Radio.Group>

          <Title level={3}>2.2 Edge vision application</Title>
          {/* <Paragraph>
            (xxx).
          </Paragraph> */}
          <Radio.Group onChange={e => setEdgeVisionApplication(e.target.value)} value={edgeVisionApplication}>
            {
              edgeVisionApplications.map(edgeVisionApplication => (
                <Radio
                  key={edgeVisionApplication}
                  value={edgeVisionApplication}
                >
                  {edgeVisionApplication}
                </Radio>
              ))
            }
          </Radio.Group>

          <Title level={3}>2.3 Source datasets</Title>
          <Paragraph>
            The selected datasets will be merged into a dataset as the source domain.
          </Paragraph>
          <TreeSelect
            placement='bottomLeft'
            showSearch
            style={{ width: '100%' }}
            value={sourceDatasets}
            treeData={convertAvailableDatasetsToSourceDatasetsTreeValue(availableDatasets)}
            dropdownStyle={{ maxHeight: 400, overflow: 'auto' }}
            placeholder={datasetSelectDisabled ? 'Loading available datasets...' : 'Select datasets in source domain'}
            allowClear
            multiple
            treeDefaultExpandAll
            onChange={setSourceDatasets}
            disabled={datasetSelectDisabled}
          />

          <Title level={3}>2.4 Target domains</Title>
          {/* <Paragraph>
            Each selected dataset will be a target domain.
          </Paragraph> */}
          <Radio.Group
            onChange={e => {
              setTargetDomains([]);
              setManualTargetDomains([]);
              setAutoRandomizeTargetDomains(e.target.value)
            }
            }
            value={autoRandomizeTargetDomains}
          >
            <Radio value={true}>Automatically randomize the order of target domains</Radio>
            <Radio value={false}>Manually order target domains</Radio>
          </Radio.Group>
          <Paragraph></Paragraph>
          {
            autoRandomizeTargetDomains ?
              <div>
                {/* <Paragraph>
                  EdgeVisionBench will automatically randomize the appearence of selected target datasets to construct evolving target domains.
                </Paragraph> */}
                <Title level={4}>2.4.1 The number of target domains</Title>
                <InputNumber
                  min={1}
                  value={manualNumTargetDomains}
                  onChange={e => {
                    setManualNumTargetDomains(e);
                    setTargetDomains(autoRandomizeTargetDomainsFunc(manualTargetDomains, e));
                  }}
                />
                <Title level={4}>2.4.2 Target datasets</Title>
                <TreeSelect
                  placement='topLeft'
                  showSearch
                  style={{ width: '100%' }}
                  value={manualTargetDomains}
                  treeData={convertAvailableDatasetsToSourceDatasetsTreeValue(availableDatasets)}
                  dropdownStyle={{ maxHeight: 400, overflow: 'auto' }}
                  placeholder={datasetSelectDisabled ? 'Loading available datasets...' : 'Select datasets in target domain'}
                  allowClear
                  multiple
                  treeDefaultExpandAll
                  onChange={e => {
                    setManualTargetDomains(e);
                    setTargetDomains(autoRandomizeTargetDomainsFunc(e, manualNumTargetDomains))
                  }}
                  disabled={datasetSelectDisabled}
                />
              </div> :
              <div>
                <Paragraph>
                  You should manually determine every target domain to construct evolving target domains.
                </Paragraph>
                <TreeSelect
                  placement='bottomLeft'
                  style={{ width: '100%' }}
                  value={targetDomains}
                  treeData={convertAvailableDatasetsToTargetDomainTreeValue(availableDatasets)}
                  dropdownStyle={{ maxHeight: 400, overflow: 'auto' }}
                  placeholder={datasetSelectDisabled ? 'Loading available datasets...' : 'Select target domains'}
                  allowClear
                  multiple
                  treeDefaultExpandAll
                  onChange={e => { setTargetDomains(targetDomains.filter(d => e.includes(d))) }}
                  disabled={datasetSelectDisabled}
                />
              </div>
          }

          <Title level={4}>2.4.3 Preview of evolving target domains → </Title>
          <Paragraph></Paragraph>
          <TargetDomainsPreview
            targetDomains={targetDomains}
            onRemoveTargetDomains={(removedTargetDomainIndex) =>
              setTargetDomains(targetDomains.filter((_, i) => i !== removedTargetDomainIndex))
            }
          />

          <Paragraph></Paragraph>
          <Alert message={noteBeforeBuilding} type={buildDisabled ? "error" : "success"} showIcon />

          <Paragraph></Paragraph>
          <div align='center'>
            <Space size='large'>
              <Button type='primary' onClick={() => carousel.current.prev()}>Back</Button>
              <Button
                type='primary' 
                onClick={() => {carousel.current.next();}}
                disabled={buildDisabled}
              >Next</Button>
            </Space>
          </div>
          
        </div>

        <div>

          <Title level={2}><CarTwoTone /> 3. Evolving domains construction </Title>

          {/* <Paragraph>
            {noteBeforeBuilding}
          </Paragraph> */}
          
          <div align='center'>
            <Button
              type='primary'
              disabled={buildDisabled}
              loading={buildingScenario}
              onClick={buildScenario}
            >
              {buildingScenario ? 'Building scenario...' : 'Clike me to build scenario!'}
            </Button>
          </div>
          <Paragraph></Paragraph>

          <BuildingStepsVis sourceDatasets={sourceDatasets} targetDomains={targetDomains} {...buildingScenarioResData} />
          
          <Paragraph></Paragraph>
          <div align='center'>
            <Space size='large'>
              <Button type='primary' onClick={() => carousel.current.prev()}>Back</Button>
              <Button
                type='primary' 
                onClick={() => carousel.current.next()}
                disabled={Object.keys(buildingScenarioResData.indexClassMap).length === 0}
              >Next</Button>
            </Space>
          </div>
        </div>

        <div>
          <Title level={2}><SettingTwoTone /> 4. Evaluation settings </Title>

          <Title level={3}>4.1 Model</Title>
          <TreeSelect
            placement='bottomLeft'
            showSearch
            style={{ width: '100%' }}
            value={model}
            treeData={convertCommonTreeDataToTreeValue(availableModels, edgeVisionApplication)}
            dropdownStyle={{ maxHeight: 400, overflow: 'auto' }}
            placeholder={availableModels === null ? 'Loading available models...' : 'Select a model'}
            allowClear
            // multiple
            treeDefaultExpandAll
            onChange={setModel}
            disabled={availableModels === null}
          />
          {blankSpace(12)}

          <Title level={3}>4.2 Model compression algorithm</Title>
          <TreeSelect
            placement='bottomLeft'
            showSearch
            style={{ width: '100%' }}
            value={modelCompressionAlg}
            treeData={convertCommonTreeDataToTreeValue(availableModelCompressionAlgs, undefined)}
            dropdownStyle={{ maxHeight: 400, overflow: 'auto' }}
            placeholder={availableModelCompressionAlgs === null ? 'Loading available model compression algorithms...' : 'Select a model compression algorithm'}
            allowClear
            // multiple
            treeDefaultExpandAll
            onChange={setModelCompressionAlg}
            disabled={availableModelCompressionAlgs === null}
          />
          {blankSpace(8)}

          <Title level={3}>4.3 DA algorithm</Title>
          <TreeSelect
            placement='bottomLeft'
            showSearch
            style={{ width: '100%' }}
            value={DAAlg}
            treeData={convertCommonTreeDataToTreeValue(availableDAAlgs, daSettingOnLabelSpace)}
            dropdownStyle={{ maxHeight: 400, overflow: 'auto' }}
            placeholder={availableDAAlgs === null ? 'Loading available DA algorithms...' : 'Select a DA algorithm'}
            allowClear
            // multiple
            treeDefaultExpandAll
            onChange={setDAAlg}
            disabled={availableDAAlgs === null}
          />

          {blankSpace(5)}


          <Title level={3}>4.4 Period of Domain occurrence</Title>
          <InputNumber
            min={1}
            value={60}
          // onChange={setNumTargetSamples}
          /> (minutes)

          <Title level={3}>4.5 Hyperparameter</Title>
          <TreeSelect
            placement='bottomLeft'
            showSearch
            style={{ width: '100%' }}
            value={hpSearchAlg}
            treeData={convertCommonTreeDataToTreeValue(availableHpSearchAlgs, undefined)}
            dropdownStyle={{ maxHeight: 400, overflow: 'auto' }}
            placeholder={availableHpSearchAlgs === null ? 'Loading available hyperparameter search algorithms...' : 'Select a hyperparameter search algorithm'}
            allowClear
            // multiple
            treeDefaultExpandAll
            onChange={setHpSearchAlg}
            disabled={availableHpSearchAlgs === null}
          />
          {blankSpace(5)}

          <Title level={3}>4.6 Metrics</Title>
          <TreeSelect
            placement='bottomLeft'
            showSearch
            style={{ width: '100%' }}
            value={metrics}
            treeData={convertCommonTreeDataToTreeValue(availableMetrics, undefined)}
            dropdownStyle={{ maxHeight: 400, overflow: 'auto' }}
            placeholder={availableMetrics === null ? 'Loading available metrics...' : 'Select metrics you need'}
            allowClear
            multiple
            treeDefaultExpandAll
            onChange={setMetrics}
            disabled={availableMetrics === null}
          />

          {blankSpace(16)}

          <Paragraph></Paragraph>
          <div align='center'>
            <Space size='large'>
              <Button type='primary' onClick={() => carousel.current.prev()}>Back</Button>
              <Button
                type='primary' 
                onClick={() => carousel.current.next()}
                disabled={!(model && modelCompressionAlg && DAAlg && hpSearchAlg && metrics.length > 0)}
              >Next</Button>
            </Space>
          </div>

        </div>

        <div>
          <Title level={2}><FundTwoTone /> 5. Evaluation </Title>

          <div align='center'>
            <Space size='large'>
              <Button type='primary' disabled={evaluating} onClick={() => {
                message.success('start evaluation successfully!'); 
                setEvaluating(true);
              }}>{evaluating ? 'Evaluating...' : 'Start Evaluation'}</Button>
            </Space>
          </div>

          <Paragraph></Paragraph>
          <div align='center'>
            <Space size='large'>
              <Button type='primary' onClick={() => carousel.current.prev()}>Back</Button>
            </Space>
          </div>
        </div>

      </Carousel>



    </>
  );
}
