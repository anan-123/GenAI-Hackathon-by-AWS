import React, { useState } from 'react';
import Header from './components/Header';
import Navigation from './components/Navigation';
import DataInput from './components/DataInput';
import InventoryAI from './components/InventoryAI';
import ChatAgent from './components/ChatAgent';
import AILab from './components/AILab';

const App = () => {
  const [activeTab, setActiveTab] = useState('input');
  const [forecastData, setForecastData] = useState(null); 

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'input':
        return <DataInput setForecastData={setForecastData} />;
      case 'inventory':
        return <InventoryAI forecastData={forecastData} />;
      case 'chat':
        return <ChatAgent />;
      case 'playground':
        return <AILab />;
      default:
        return <DataInput setForecastData={setForecastData} />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
      <Header />
      <div className="max-w-7xl mx-auto px-6 py-6">
        <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
        {renderActiveTab()}
      </div>
    </div>
  );
};

export default App;
