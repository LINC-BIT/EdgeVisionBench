import { Form, TreeSelect, Card, Button, Input, Space, Row, Col, Typography, Steps, Tag } from 'antd';
import { ConsoleSqlOutlined, LoadingOutlined } from '@ant-design/icons';
import React, { useEffect, useState } from 'react';
import { getAvailableDatasets } from '../data/remote';

const { TreeNode } = TreeSelect;
const { Title, Paragraph, Text } = Typography;

const presetTagColors = ["blue", "red", "green", "orange", "cyan",  "pink", "purple", "geekblue", "magenta", "volcano"];

export default function TargetDomainsPreview({ targetDomains, onRemoveTargetDomains }) {
  if (targetDomains.length === 0) {
    return <Paragraph>(No target domain yet)</Paragraph>
  }

  const colorMap = {};
  const targetDomainsSet = [...new Set(targetDomains)];
  targetDomainsSet.forEach((targetDomain, i) => colorMap[targetDomain] = presetTagColors[i % presetTagColors.length]);

  return <div style={{
    lineHeight: '2rem'
  }}>
    {
      targetDomains.map((targetDomain, i) =>
        <span key={`${targetDomain}_${i}`}>
          <Tag color={colorMap[targetDomain]} closable onClose={(e) => {e.preventDefault(); onRemoveTargetDomains(i)}}>
            {targetDomain}
          </Tag>
          { i === targetDomains.length - 1 ? '' : <span>→ </span> }
        </span>
      )
    }
  </div>
}