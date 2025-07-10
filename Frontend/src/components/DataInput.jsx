
import React, { useState } from 'react';
import { Upload, TrendingUp, AlertCircle, CheckCircle, Loader, Database, BarChart3, Activity } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const DataInput = ({setForecastData}) => {
  const [supplierData, setSupplierData] = useState('');
  const [s3Url, setS3Url] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState(null);
  const [processedData, setProcessedData] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [metricsData, setMetricsData] = useState(null);
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [selectedModel, setSelectedModel] = useState('');

  
  const [performanceData, setPerformanceData] = useState([
    { metric: 'Data Quality', value: 85 },
    { metric: 'Completeness', value: 92 },
    { metric: 'Processing Speed', value: 88 },
    { metric: 'Analysis Depth', value: 75 }
  ]);

  const API_BASE_URL = 'http://localhost:8000';

  const createPerformanceMetrics = (analysis, metrics) => {
    const performanceMetrics = [];
    
    try {
 
      if (metrics?.data_quality) {
        performanceMetrics.push({
          metric: 'Data Quality',
          value: Math.round(metrics.data_quality.completeness_percentage || 85)
        });
      }
      

      if (analysis?.structure) {
        const totalColumns = analysis.structure.total_columns || 0;
        const totalRows = analysis.structure.total_rows || 0;
        const efficiency = Math.min(100, Math.max(0, (totalRows * totalColumns) / 100));
        performanceMetrics.push({
          metric: 'Processing Efficiency',
          value: Math.round(efficiency)
        });
      }
      

      if (analysis?.patterns) {
        const patterns = analysis.patterns;
        let depth = 50; // base score
        
        if (patterns.has_time_series) depth += 15;
        if (patterns.has_geographical) depth += 10;
        if (patterns.has_transactional) depth += 15;
        if (patterns.has_hierarchical) depth += 10;
        
        performanceMetrics.push({
          metric: 'Analysis Depth',
          value: Math.min(100, depth)
        });
      }
      
   
      if (analysis?.structure?.column_types) {
        const columnTypes = Object.keys(analysis.structure.column_types).length;
        const diversity = Math.min(100, (columnTypes * 20));
        performanceMetrics.push({
          metric: 'Data Diversity',
          value: Math.round(diversity)
        });
      }
      
 
      if (metrics?.basic_stats) {
        const totalRecords = metrics.basic_stats.total_records || 0;
        const completeness = Math.min(100, Math.max(0, (totalRecords / 1000) * 100));
        performanceMetrics.push({
          metric: 'Dataset Size Score',
          value: Math.round(completeness)
        });
      }
      
    
      while (performanceMetrics.length < 4) {
        const remainingMetrics = [
          { metric: 'Processing Speed', value: 88 },
          { metric: 'Data Consistency', value: 82 },
          { metric: 'Analysis Accuracy', value: 91 },
          { metric: 'Optimization Potential', value: 76 }
        ];
        
        const nextMetric = remainingMetrics[performanceMetrics.length - 1];
        if (nextMetric) {
          performanceMetrics.push(nextMetric);
        } else {
          break;
        }
      }
      
      return performanceMetrics.slice(0, 6); 
      
    } catch (error) {
      console.error('Error creating performance metrics:', error);
      return [
        { metric: 'Data Quality', value: 85 },
        { metric: 'Processing Efficiency', value: 78 },
        { metric: 'Analysis Depth', value: 82 },
        { metric: 'System Performance', value: 90 }
      ];
    }
  };

  const handleS3DataFetch = async () => {
    if (!s3Url) {
      setStatus({ type: 'error', message: 'Please enter a valid S3 URL' });
      return;
    }

    setIsLoading(true);
    setStatus(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/data/s3`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          s3_url: s3Url,
          data_type: 'supply_chain'
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.status === 'success') {
        setProcessedData(result.data);
        setAnalysisData(result.analysis);
        setMetricsData(result.metrics);
        
        setStatus({ 
          type: 'success', 
          message: `Successfully loaded ${result.shape[0]} rows and ${result.shape[1]} columns` 
        });
        
        const newMetrics = createPerformanceMetrics(result.analysis, result.metrics);
        setPerformanceData(newMetrics);
      }
    } catch (error) {
      setStatus({ 
        type: 'error', 
        message: `Failed to fetch S3 data: ${error.message}` 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;

    if (selectedFile.type === 'text/csv') {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const csvData = e.target.result;
        await processCSVData(csvData);
      };
      reader.readAsText(selectedFile);
    }
  };

  const processCSVData = async (csvData) => {
    setIsLoading(true);
    setStatus(null);

    try {
      const lines = csvData.split('\n');
      const headers = lines[0].split(',');
      const data = lines.slice(1).map(line => {
        const values = line.split(',');
        const row = {};
        headers.forEach((header, index) => {
          row[header.trim()] = values[index]?.trim();
        });
        return row;
      }).filter(row => Object.values(row).some(val => val));
      const response = await fetch(`${API_BASE_URL}/api/data/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data: data,
          processing_type: 'auto'
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.status === 'success') {
        setProcessedData(result.processed_data);
        setAnalysisData(result.analysis);
        setMetricsData(result.metrics);
        
        setStatus({ 
          type: 'success', 
          message: `Successfully processed ${data.length} rows from CSV` 
        });
      
        const newMetrics = createPerformanceMetrics(result.analysis, result.metrics);
        setPerformanceData(newMetrics);
      }
    } catch (error) {
      setStatus({ 
        type: 'error', 
        message: `Failed to process CSV: ${error.message}` 
      });
    } finally {
      setIsLoading(false);
    }
  };
  const analyzeLoadedData = async () => {
  if (!processedData || processedData.length === 0) {
    setStatus({ type: 'error', message: 'No data available to analyze. Please upload or load data first.' });
    return;
  }

  setIsLoading(true);
  setStatus(null);

  try {
    const response = await fetch(`${API_BASE_URL}/api/data/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        data: processedData,
        processing_type: 'auto'
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();

    if (result.status === 'success') {
      setProcessedData(result.processed_data);
      setAnalysisData(result.analysis);
      setMetricsData(result.metrics);

      setStatus({ 
        type: 'success', 
        message: `Successfully analyzed ${processedData.length} records` 
      });

      const newMetrics = createPerformanceMetrics(result.analysis, result.metrics);
      setPerformanceData(newMetrics);
    }
  } catch (error) {
    setStatus({ 
      type: 'error', 
      message: `Failed to analyze data: ${error.message}` 
    });
  } finally {
    setIsLoading(false);
  }
};

  const processSupplierData = async () => {
    if (!supplierData.trim()) {
      setStatus({ type: 'error', message: 'Please enter supplier data' });
      return;
    }

    setIsLoading(true);
    setStatus(null);

    try {
      let parsedData;
      try {
        parsedData = JSON.parse(supplierData);
      } catch {
        parsedData = parseCSVString(supplierData);
      }

      if (!Array.isArray(parsedData)) {
        parsedData = [parsedData];
      }

      const response = await fetch(`${API_BASE_URL}/api/data/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data: parsedData,
          processing_type: 'auto'
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.status === 'success') {
        setProcessedData(result.processed_data);
        setAnalysisData(result.analysis);
        setMetricsData(result.metrics);
        
        setStatus({ 
          type: 'success', 
          message: `Successfully processed ${parsedData.length} records` 
        });
        
        const newMetrics = createPerformanceMetrics(result.analysis, result.metrics);
        setPerformanceData(newMetrics);
      }
    } catch (error) {
      setStatus({ 
        type: 'error', 
        message: `Failed to process data: ${error.message}` 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const parseCSVString = (csvString) => {
    const lines = csvString.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    return lines.slice(1).map(line => {
      const values = line.split(',').map(v => v.trim());
      const row = {};
      headers.forEach((header, index) => {
        row[header] = values[index];
      });
      return row;
    });
  };

  // const generateForecast = async () => {
  //   if (!processedData || processedData.length === 0) {
  //     setStatus({ type: 'error', message: 'No data available for forecasting' });
  //     return;
  //   }

  //   setIsLoading(true);
  //   setStatus(null);

  //   try {
  //     const response = await fetch(`${API_BASE_URL}/api/forecast/timeseries`, {
  //       method: 'POST',
  //       headers: {
  //         'Content-Type': 'application/json',
  //       },
  //       body: JSON.stringify({
  //         data: processedData,
  //         periods: 30,
  //         forecast_type: 'inventory'
  //       })
  //     });

  //     if (!response.ok) {
  //       throw new Error(`HTTP error! status: ${response.status}`);
  //     }

  //     const result = await response.json();
      
  //     if (result.status === 'success') {
  //       setStatus({ 
  //         type: 'success', 
  //         message: `Forecast generated successfully with ${result.forecast.length} periods` 
  //       });
        
  //       // Update performance metrics with forecast accuracy
  //       const forecastMetrics = [
  //         { metric: 'Forecast Accuracy', value: Math.round(100 - (result.metrics.mape || 10)) },
  //         { metric: 'Trend Strength', value: Math.round(Math.abs(result.metrics.trend || 0)) },
  //         { metric: 'Seasonal Impact', value: Math.round(Math.abs(result.metrics.seasonal || 0)) },
  //         { metric: 'Model Performance', value: 92 }
  //       ];
        
  //       setPerformanceData(prev => [...prev.slice(0, 2), ...forecastMetrics]);
  //     }
  //   } catch (error) {
  //     setStatus({ 
  //       type: 'error', 
  //       message: `Failed to generate forecast: ${error.message}` 
  //     });
  //   } finally {
  //     setIsLoading(false);
  //   }
  // };
 
  const generateForecast = async (modelKey = 'Prophet') => {
  if (!processedData || processedData.length === 0) {
    setStatus({ type: 'error', message: 'No data available for forecasting' });
    return;
  }

  setIsLoading(true);
  setStatus(null);

  try {
     const response = await fetch(`${API_BASE_URL}/api/forecast/timeseries`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        data: processedData,
        periods: 30,
        forecast_type: 'inventory'
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
   // const modelKey = 'Prophet'; // or 'ARIMA', depending on which model you want to show
    const result = await response.json();
  
    // console.log(result); 
    if (result.status === 'success') {
      const modelOutput = result.results[modelKey];
      setStatus({
        type: 'success',
        message: `Forecast generated successfully with ${modelOutput.forecast.length} periods`,
      });
      
       if (setForecastData) {
          setForecastData(modelOutput.forecast);
          console.log(modelOutput.forecast)
        }
      // Update performance metrics only with available metrics (mae, mape, rmse)
      const forecastMetrics = [
        { metric: 'MAE/AIC', value: modelOutput.metrics.mae !== null ? modelOutput.metrics.mae.toFixed(2) : 'N/A' },
        // { metric: 'MAPE', value: modelOutput.metrics.mape !== null ? modelOutput.metrics.mape.toFixed(2) + '%' : 'N/A' },
        { metric: 'RMSE/MSE', value: modelOutput.metrics.rmse !== null ? modelOutput.metrics.rmse.toFixed(2) : 'N/A' },
      ];

      setPerformanceData(forecastMetrics);
    }
  } catch (error) {
    setStatus({
      type: 'error',
      message: `Failed to generate forecast: ${error.message}`,
    });
  } finally {
    setIsLoading(false);
  }
  };

  const StatusIndicator = ({ status }) => {
    if (!status) return null;

    return (
      <div className={`flex items-center gap-2 p-3 rounded-lg mb-4 ${
        status.type === 'success' 
          ? 'bg-green-900/20 text-green-400 border border-green-500/30' 
          : 'bg-red-900/20 text-red-400 border border-red-500/30'
      }`}>
        {status.type === 'success' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
        <span className="text-sm">{status.message}</span>
      </div>
    );
  };

  const AnalysisInsights = ({ analysis, metrics }) => {
    if (!analysis || !metrics) return null;

    return (
      <div className="mt-4 p-4 bg-gray-900/30 rounded-lg">
        <h4 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
          <Database size={16} />
          Analysis Insights
        </h4>
        <div className="space-y-2 text-xs">
          {analysis.structure && (
            <div className="flex justify-between">
              <span className="text-gray-400">Dataset Structure:</span>
              <span className="text-gray-200">
                {analysis.structure.total_rows} rows × {analysis.structure.total_columns} columns
              </span>
            </div>
          )}
          {metrics.data_quality && (
            <div className="flex justify-between">
              <span className="text-gray-400">Data Quality:</span>
              <span className="text-gray-200">
                {Math.round(metrics.data_quality.completeness_percentage)}% complete
              </span>
            </div>
          )}
          {analysis.patterns && (
            <div className="flex justify-between">
              <span className="text-gray-400">Patterns Detected:</span>
              <span className="text-gray-200">
                {Object.values(analysis.patterns).filter(Boolean).length} patterns
              </span>
            </div>
          )}
          {analysis.categories && (
            <div className="flex justify-between">
              <span className="text-gray-400">Column Types:</span>
              <span className="text-gray-200">
                {Object.keys(analysis.categories).length} categories
              </span>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
        <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Upload size={20} className="text-cyan-400" />
          Universal Data Input
        </h3>
        
        <StatusIndicator status={status} />
        
        <div className="space-y-4">
          {/* S3 Data Input */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">S3 Data URL</label>
            <div className="flex gap-2">
              <input
                type="url"
                value={s3Url}
                onChange={(e) => setS3Url(e.target.value)}
                className="flex-1 bg-gray-900/50 border border-gray-600 rounded-lg p-3 text-white placeholder-gray-400 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
                placeholder="https://bucket.s3.amazonaws.com/data.csv or s3://bucket/data.csv"
              />
              <button
                onClick={handleS3DataFetch}
                disabled={isLoading}
                className="px-4 py-2 bg-cyan-500 text-white rounded-lg hover:bg-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isLoading ? <Loader className="animate-spin" size={16} /> : 'Fetch'}
              </button>
            </div>
          </div>

          {/* Universal Data Input */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Data Input</label>
            <textarea
              value={supplierData}
              onChange={(e) => setSupplierData(e.target.value)}
              className="w-full h-24 bg-gray-900/50 border border-gray-600 rounded-lg p-3 text-white placeholder-gray-400 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
              placeholder="Paste any CSV data, JSON, or structured data..."
            />
          </div>
          
          {/* File Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Upload Data File</label>
            <input
              type="file"
              accept=".csv,.json"
              onChange={handleFileUpload}
              className="w-full bg-gray-900/50 border border-gray-600 rounded-lg p-3 text-white file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-cyan-500 file:text-white file:cursor-pointer"
            />
          </div>
          
          {/* Action Buttons */}
          <div className="flex gap-2">
            <button
              onClick={analyzeLoadedData}
              disabled={isLoading}
              className="flex-1 bg-gradient-to-r from-cyan-500 to-blue-500 text-white py-3 rounded-lg font-medium hover:from-cyan-600 hover:to-blue-600 transition-all shadow-lg shadow-cyan-500/25 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? <Loader className="animate-spin" size={16} /> : <Activity size={16} />}
              Analyze Data
            </button>
            
            {/* <button
              onClick={generateForecast}
              disabled={isLoading || !processedData}
              className="flex-1 bg-gradient-to-r from-green-500 to-emerald-500 text-white py-3 rounded-lg font-medium hover:from-green-600 hover:to-emerald-600 transition-all shadow-lg shadow-green-500/25 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? <Loader className="animate-spin" size={16} /> : <TrendingUp size={16} />}
              Generate Forecast
            </button> */}
              <button
    onClick={() => setShowModelDropdown(true)}
    disabled={isLoading || !processedData}
    className="flex-1 bg-gradient-to-r from-green-500 to-emerald-500 text-white py-3 rounded-lg font-medium hover:from-green-600 hover:to-emerald-600 transition-all shadow-lg shadow-green-500/25 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
  >
    {isLoading ? <Loader className="animate-spin" size={16} /> : <TrendingUp size={16} />}
    Generate Forecast
  </button>
  {showModelDropdown && (
  <div className="mt-2">
    <label className="block text-sm font-medium text-gray-300 mb-2">
      Select Forecast Model
    </label>
    <select
      value={selectedModel}
      onChange={async (e) => {
        const model = e.target.value;
        setSelectedModel(model);
        await generateForecast(model);
        setShowModelDropdown(false);
      }}
      className="w-full bg-gray-900/50 border border-gray-600 rounded-lg p-3 text-white focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
    >
      <option value="">-- Choose a Model --</option>
      <option value="Prophet">Prophet</option>
      <option value="ARIMA">ARIMA</option>
      <option value="LSTM">BEDROCK Claude</option>
      <option value="XGBoost">Transformer</option>
    </select>
  </div>
)}


          </div>

          {/* Analysis Insights */}
          <AnalysisInsights analysis={analysisData} metrics={metricsData} />

          {/* Data Preview */}
          {processedData && (
            <div className="mt-4 p-3 bg-gray-900/30 rounded-lg">
              <h4 className="text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
                <BarChart3 size={16} />
                Data Preview
              </h4>
              <div className="text-xs text-gray-400 mb-2">
                {processedData.length} records loaded
              </div>
              <div className="max-h-32 overflow-y-auto">
                <pre className="text-xs text-gray-300">
                  {JSON.stringify(processedData.slice(0, 3), null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
        <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <TrendingUp size={20} className="text-green-400" />
          Performance Analytics
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="metric" 
                stroke="#9CA3AF" 
                fontSize={12}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }} 
              />
              <Bar dataKey="value" fill="url(#gradient)" radius={[4, 4, 0, 0]} />
              <defs>
                <linearGradient id="gradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.8}/>
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
        </div>
        
        {/* Performance Summary */}
        {(analysisData || metricsData) && (
          <div className="mt-4 p-3 bg-gray-900/30 rounded-lg">
            <h4 className="text-sm font-medium text-gray-300 mb-2">Performance Summary</h4>
            <div className="text-xs text-gray-400">
              Dynamic metrics calculated from your data analysis and processing results.
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataInput;