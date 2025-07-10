import React from 'react';
import { Package } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';

const InventoryAI = ({ forecastData }) => {
  console.log('forecastData:', forecastData);

  const transformedForecast = forecastData?.map((item) => ({
    month: new Date(item.ds).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    actual: null,
    predicted: item.yhat,
    safety: item.yhat * 0.8,
  }));

  const inventoryData = transformedForecast && transformedForecast.length > 0
    ? transformedForecast
    : [
        { month: 'Jan', actual: 4000, predicted: 4200, safety: 3000 },
        { month: 'Feb', actual: 3000, predicted: 3100, safety: 2800 },
        { month: 'Mar', actual: 5000, predicted: 4900, safety: 4200 },
        { month: 'Apr', actual: 2780, predicted: 2900, safety: 2500 },
        { month: 'May', actual: 1890, predicted: 2000, safety: 1800 },
        { month: 'Jun', actual: 2390, predicted: 2300, safety: 2100 }
      ];

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
      <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
        <Package size={20} className="text-blue-400" />
        Inventory Forecast
      </h3>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={inventoryData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="month" stroke="#9CA3AF" />
            <YAxis stroke="#9CA3AF" />
            <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }} />
            <Line type="monotone" dataKey="actual" stroke="#06b6d4" strokeWidth={2} />
            <Line type="monotone" dataKey="predicted" stroke="#3b82f6" strokeWidth={2} strokeDasharray="5 5" />
            <Line type="monotone" dataKey="safety" stroke="#ef4444" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default InventoryAI;
