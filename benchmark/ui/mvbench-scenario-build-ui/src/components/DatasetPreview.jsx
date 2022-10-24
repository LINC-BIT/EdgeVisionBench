import { Form, TreeSelect, List, Image, Typography } from 'antd';
import { ConsoleSqlOutlined, LoadingOutlined } from '@ant-design/icons';
import React, { useEffect, useState } from 'react';
import { getDatasetPreviewImagesInfo } from '../data/remote';
import useDeepCompareEffect from 'use-deep-compare-effect'

const { TreeNode } = TreeSelect;
const { Title, Paragraph, Text } = Typography;

export default function DatasetPreview({ domainIndex, datasetName, classesName, unknownClassesName, numImagesPerClass }) {
  // console.log(datasetName, classesName, unknownClassesName, numImagesPerClass)

  const [datasetPreviewImagesInfo, setDatasetPreviewImagesInfo] = useState([]);

  const maxNumClasses = 20;
  const tooMuchClasses = classesName.length > maxNumClasses;
  if (classesName.length > maxNumClasses) {
    classesName = classesName.slice(0, maxNumClasses);
  }
  classesName = [...classesName, ...unknownClassesName.slice(0, 1)]
  // console.log(classesName)
  useDeepCompareEffect(() => {
    // console.log(classesName, typeof classesName, classesName.length)
    // const resClassesName = [];
    // for (let ci = 0; ci < classesName.length; ci++) {
    //   const c = classesName[ci];
    //   resClassesName.push(/^[a-z]$/.test(c) ? c.toUpperCase() : c)
    // }
    // classesName ? [...classesName].map(c => /^[a-z]$/.test(c) ? c.toUpperCase() : c) : classesName
    const resClassesName = classesName.map(c => /^[a-z]$/.test(c) ? c.toUpperCase() : c);
    getDatasetPreviewImagesInfo({ datasetName, classesName: resClassesName, numImagesPerClass }).then(res => {
      setDatasetPreviewImagesInfo(res);
    })
  }, [datasetName, classesName, unknownClassesName, numImagesPerClass]);

  const stdHeight = 50;

  return <>
    {/* <Text strong>{datasetName} {tooMuchClasses ? ` (too many classes! show only ${maxNumClasses} classes)` : ''}</Text> */}
    <List
      size='small'
      bordered
      header={<Title level={4} className='no-dash' style={{
        textAlign: 'center',
        color: '#1890ff'
        // fontSize: '24px',
        // fontWeight: 'bold'
      }}>{domainIndex ? `${domainIndex}-th target domain: ` : ''} {datasetName} {tooMuchClasses ? ` (too many classes! show only ${maxNumClasses} classes)` : ''}</Title>}
      grid={{
        gutter: 12,
        xs: 6,
        sm: 6,
        md: 6,
        lg: 6,
        xl: 12,
        xxl: 12,
      }}
      dataSource={datasetPreviewImagesInfo}
      renderItem={item => (
        <div>
          <div align='center'>
            <Image 
              src={item.url}
              width={stdHeight}
              height={stdHeight}
            />
          </div>
          <div align='center'>
            {unknownClassesName.includes(item.class) ? '(unknown)' : `"${item.class}"`}
          </div>
        </div>
      )}
    />
  </>
}
