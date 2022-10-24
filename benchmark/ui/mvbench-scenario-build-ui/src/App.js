import React from 'react';
import './App.css';

import { Anchor, Typography } from 'antd';
import {
  SettingTwoTone,
  DatabaseTwoTone,
  FileTextTwoTone,
  FileImageTwoTone
} from '@ant-design/icons';

import DataSettings from './components/DataSettings';
const { Title, Paragraph, Text } = Typography;


const App = () => (
  <div className="App">
    <Typography>
      <div align='center'>
        <Title>EdgeVisionBench</Title>
      </div>
      
      <DataSettings />

    </Typography>
  </div>
);

export default App;