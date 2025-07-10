
import React from 'react';
import { Upload, Package, MessageSquare, Brain } from 'lucide-react';

const TabButton = ({ id, icon: Icon, label, isActive, onClick }) => (
  <button
    onClick={() => onClick(id)}
    className={`flex items-center gap-2 px-4 py-3 rounded-lg transition-all duration-300 ${
      isActive 
        ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg shadow-cyan-500/25' 
        : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 hover:text-white'
    }`}
  >
    <Icon size={18} />
    <span className="font-medium">{label}</span>
  </button>
);

const Navigation = ({ activeTab, setActiveTab }) => {
  return (
    <div className="flex gap-2 mb-8">
      <TabButton id="input" icon={Upload} label="Data Input" isActive={activeTab === 'input'} onClick={setActiveTab} />
      <TabButton id="inventory" icon={Package} label="Inventory AI" isActive={activeTab === 'inventory'} onClick={setActiveTab} />
      <TabButton id="chat" icon={MessageSquare} label="Chat Agent" isActive={activeTab === 'chat'} onClick={setActiveTab} />
      <TabButton id="playground" icon={Brain} label="AI Lab" isActive={activeTab === 'playground'} onClick={setActiveTab} />
    </div>
  );
};

export default Navigation;