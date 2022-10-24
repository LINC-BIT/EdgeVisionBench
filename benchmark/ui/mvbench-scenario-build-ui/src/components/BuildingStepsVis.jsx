import { TreeSelect, Typography, Table, Tooltip, message, Collapse, Alert, Timeline, Divider } from 'antd';
import React, { useEffect, useState, useMemo } from 'react';
import Graphviz from 'graphviz-react';
import DatasetPreview from './DatasetPreview';
import {ArrowDownOutlined} from '@ant-design/icons';

const { TreeNode } = TreeSelect;
const { Panel } = Collapse;
const { Title, Paragraph, Text } = Typography;


const renamedClassesTableColumns = [
  { title: 'Dataset', dataIndex: 'dataset', key: 'dataset' },
  { title: 'Original class', dataIndex: 'originalClass', key: 'originalClass' },
  { title: 'Renamed class', dataIndex: 'renamedClass', key: 'renamedClass' },
]

const classesTableColumns = [
  { title: 'Dataset', dataIndex: 'dataset', key: 'dataset' },
  { title: 'Known classes', dataIndex: 'knownClasses', key: 'knownClasses', ellipsis: true },
  { title: 'Unknown classes', dataIndex: 'unknownClasses', key: 'unknownClasses', ellipsis: true },
  { title: 'Discarded classes', dataIndex: 'discardedClasses', key: 'discardedClasses', ellipsis: true },
]

const globalClassIndexTableColumns = [
  { title: 'Class', dataIndex: 'index', key: 'index' },
  { title: 'Index', dataIndex: 'class', key: 'class' }
]


export default function BuildingStepsVis({ 
  sourceDatasets,
  targetDomains,
  classesRenameMap, 
  targetSourceRelationshipMap, 
  classesInEachDatasetMap,
  indexClassMap,
  generatedCode
}) {
  // console.log({ classesRenameMap, targetSourceRelationshipMap, classesInEachDatasetMap, indexClassMap, generatedCode })

  const renamedClassesDataSource = [];
  for (const datasetName in classesRenameMap) {
    const datasetInfo = classesRenameMap[datasetName];

    for (const originalClass in datasetInfo) {
      const renamedClass = datasetInfo[originalClass];

      renamedClassesDataSource.push({
        key: `${datasetName}_${originalClass}`,
        dataset: datasetName,
        originalClass,
        renamedClass
      })
    }
  }

  let stRelationshipGraphvizDot = ''
  if (Object.keys(targetSourceRelationshipMap).length === 0) {
    stRelationshipGraphvizDot = 'digraph G {}'
  } else {
    const sourceDatasets = [];
    const targetDomains = [];
    const sourceToTarget = [];
    for (const targetDomain in targetSourceRelationshipMap) {
      targetDomains.push(targetDomain);
      const relationship = targetSourceRelationshipMap[targetDomain];

      for (const sourceDataset in relationship) {
        const shiftType = relationship[sourceDataset];

        sourceDatasets.push(sourceDataset);
        sourceToTarget.push(`
        ${sourceDataset}->${targetDomain} [
          label="${shiftType}"
        ]
        `);
      }
    }

    for (const dataset in classesInEachDatasetMap) {
      if (!targetDomains.includes(dataset)) {
        sourceDatasets.push(dataset);
      }
    }

    stRelationshipGraphvizDot = `
    digraph G {
      // rankdir=LR;
      fontsize="22"

      subgraph cluster1 {
        label = "Source Datasets";
        rank=same;
        ${sourceDatasets.map(d => d + '[constraint=false, fontsize="18"]').join(' ')};
        style = "dashed";
      }
    
      subgraph cluster2 {
        label = "Target Domains";
        labelloc="b";
        rank=same;
        ${targetDomains.map(d => d + '[constraint=false, fontsize="18"]').join(' ')};
        style = "dashed";
      }

      ${sourceToTarget.map(d => d + '[fontsize="18"]').join('\n')}
    }
    `;
  }

  // const classesTableDataSource = [];
  // for (const datasetName in classesInEachDatasetMap) {
  //   const classes = classesInEachDatasetMap[datasetName];
  //   classesTableDataSource.push({
  //     key: datasetName,
  //     dataset: datasetName,
  //     knownClasses: `${classes.knownClasses.length} classes: ${classes.knownClasses.join(', ')}`,
  //     unknownClasses: `${classes.unknownClasses.length} classes: ${classes.unknownClasses.join(', ')}`,
  //     discardedClasses: `${classes.discardedClasses.length} classes: ${classes.discardedClasses.join(', ')}`,
  //   })
  // }

  const globalClassIndexDataSource = [];
  for (const index in indexClassMap) {
    const c = indexClassMap[index];
    globalClassIndexDataSource.push({
      key: index,
      index: `"${index}"`,
      class: c
    })
  }

  const copyGeneratedCode = () => {
    navigator.clipboard.writeText(generatedCode).then(() => {
      message.success('Copied!')
    })
  }

  // const [graphvizConfig] = useMemo(() => {
  //   const graphvizConfig = {
  //     height: document.body.clientHeight / 4.5,
  //     width: Math.min(1000, document.body.clientWidth) * 0.95
  //   }
  //   return [graphvizConfig];
  // }, []);
  const graphvizConfig = {
    height: document.body.clientHeight / 4.5,
    width: Math.min(1000, document.body.clientWidth) * 0.95
  }

  const graphviz = useMemo(() => (
    <Graphviz
      dot={stRelationshipGraphvizDot}
      options={graphvizConfig}
    />
  ), [stRelationshipGraphvizDot])

  if (Object.keys(classesInEachDatasetMap).length === 0) {
    return <></>
  }

  if (classesInEachDatasetMap.error) {
    return <Alert 
      message={'Build failed. This is because the object categories in source datasets and target datasets have NO intersection. Click "Back" to update the scenario preparation.'}
      type='error'
      showIcon
    />
  }

  // const graphvizConfig = {
  //   height: document.body.clientHeight / 4.5,
  //   width: Math.min(1000, document.body.clientWidth) * 0.95
  // }


 

  return <>
    {/* <Title level={3}>2. Visualization of building steps</Title> */}

    {/* <Collapse>
      <Panel header="Build successfully! Intermediate outputs..."> */}
        {/* <Title level={3}>3.1 Determine name of classes</Title>
        <Paragraph>
          Sometimes an object has different names in different datasets (e.g. "automobile" in CIFAR-10 and "car" in STL-10).
        </Paragraph>
        <Table 
          size='small'
          columns={renamedClassesTableColumns}
          dataSource={renamedClassesDataSource}
          pagination={false}
        /> */}

        <Title level={3}>3.1 Relationship between source datasets and target domain</Title>
        {/* <Paragraph>
          (xxx).
        </Paragraph> */}
        {
          graphviz
        }
        

        {/* <Title level={3}>3.3 Classes in each dataset</Title>
        <Paragraph>
          (xxx).
        </Paragraph>
        <Table 
          size='small'
          columns={classesTableColumns}
          dataSource={classesTableDataSource}
          pagination={false}
        /> */}

        <Title level={3}>3.2 Index of each class</Title>
        {/* <Paragraph>
          (xxx).
        </Paragraph> */}
        <Table 
          size='small'
          columns={globalClassIndexTableColumns}
          dataSource={globalClassIndexDataSource}
        />

        <Title level={3}>3.3 Evolving target domains visualization</Title>
        <div>
          <Title level={4} className='no-dash' style={{
            textAlign: 'center',
            color: '#1890ff'
            // fontSize: '24px',
            // fontWeight: 'bold'
          }}>Source Domain</Title>
          {
            sourceDatasets.map((datasetName, i) => (
              <DatasetPreview
                key={i}
                datasetName={datasetName}
                classesName={classesInEachDatasetMap[datasetName].knownClasses}
                unknownClassesName={classesInEachDatasetMap[datasetName].unknownClasses}
                numImagesPerClass={1}
              />
            ))
          }
        </div>

        <div align='center'>
          <ArrowDownOutlined style={{
            fontSize: '20px',
            color: '#1890ff'
          }}/>
          <Divider>
            <span style={{
            // display: 'inline-block',
            // height: '60px',
            fontSize: '20px',
            // lineHeight: '28px',
            color: '#1890ff',
            fontWeight: 'normal'
          }}>(from source domain to <b style={{fontSize: '28px'}}>evolving</b> target domains)</span>
          </Divider>
          {/* <div style={{
            // display: 'inline-block',
            // height: '60px',
            fontSize: '20px',
            color: '#1890ff'
          }}>{`${1}-th domain shifts and label shifts `}(from source to target)</div> */}
          <ArrowDownOutlined style={{
            fontSize: '20px',
            color: '#1890ff'
          }}/>
        </div>

        {/* <Title level={4} className='no-dash' style={{
            textAlign: 'center',
            color: '#1890ff'
            // fontSize: '24px',
            // fontWeight: 'bold'
          }}>3.3.2 Evolving target domains</Title> */}

        {
          targetDomains.map((datasetName, i) => (
            <>
              <DatasetPreview
                key={i}
                domainIndex={i + 1}
                datasetName={datasetName}
                classesName={classesInEachDatasetMap[datasetName].knownClasses}
                unknownClassesName={classesInEachDatasetMap[datasetName].unknownClasses}
                numImagesPerClass={1}
              />
              {
                i === targetDomains.length - 1 ? <></> :
                <div align='center'>
                  <ArrowDownOutlined style={{
                    fontSize: '20px',
                    color: '#1890ff'
                  }}/>
                  {/* <div style={{
                    // display: 'inline-block',
                    // height: '60px',
                    fontSize: '20px',
                    color: '#1890ff'
                  }}>{`${i + 2}-th domain shifts and label shifts`}</div>
                  <ArrowDownOutlined style={{
                    fontSize: '20px',
                    color: '#1890ff'
                  }}/> */}
                </div>
              }
              
            </>
          ))
        }
      {/* </Panel>
    </Collapse> */}

    

    {/* <Title level={2}>2.3 Generated code</Title>
    <Paragraph>
      Copy this code into your project:
    </Paragraph>
    <Tooltip
      title="Click me to copy this code!"
    >
      <pre onClick={copyGeneratedCode}>
        <code>
          {generatedCode}
        </code>
      </pre>
    </Tooltip> */}
  </>
}