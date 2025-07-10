import React, { useState } from 'react';
import { MessageSquare, Send, Loader2 } from 'lucide-react';

const ChatAgent = ({ forecastData, dataAnalysis, apiUrl = 'http://localhost:8000' }) => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async () => {
    if (inputMessage.trim()) {
      const userMessage = { text: inputMessage, sender: 'user' };
      
      setMessages(prev => [...prev, userMessage]);
      setIsLoading(true);
      
      try {
      
        const context = {
          ...(forecastData && { forecast_data: forecastData }),
          ...(dataAnalysis && { data_analysis: dataAnalysis })
        };

        const response = await fetch(`${apiUrl}/api/chat/query`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: inputMessage,
            context: Object.keys(context).length > 0 ? context : null
          })
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
 
        const aiMessage = { 
          text: data.response, 
          sender: 'ai',
          insights: data.insights || [],
          contextUsed: data.context_used,
          forecastAvailable: data.forecast_data_available
        };
        
        setMessages(prev => [...prev, aiMessage]);
        
      } catch (error) {
        console.error('Error calling chat API:', error);
  
        const errorMessage = { 
          text: 'Sorry, I encountered an error processing your request. Please try again.', 
          sender: 'ai',
          error: true
        };
        
        setMessages(prev => [...prev, errorMessage]);
      } finally {
        setIsLoading(false);
        setInputMessage('');
      }
    }
  };

  const renderMessage = (msg, idx) => {
    const isUser = msg.sender === 'user';
    
    return (
      <div key={idx} className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
        <div className={`max-w-xs px-4 py-2 rounded-lg ${
          isUser
            ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white'
            : msg.error 
            ? 'bg-red-900/50 text-red-200 border border-red-700'
            : 'bg-gray-700 text-gray-100'
        }`}>
          <div className="whitespace-pre-wrap">{msg.text}</div>
          
          {/* Show insights if available */}
          {msg.insights && msg.insights.length > 0 && (
            <div className="mt-2 pt-2 border-t border-gray-600">
              <div className="text-sm font-medium text-gray-300 mb-1">Key Insights:</div>
              {msg.insights.map((insight, i) => (
                <div key={i} className="text-sm text-gray-400 mb-1">
                  {insight.type === 'recommendation' ? (
                    <span className="text-cyan-400">💡 {insight.message}</span>
                  ) : (
                    <span>📊 {insight.model}: {insight.trend} trend ({insight.change_percentage?.toFixed(1)}%)</span>
                  )}
                </div>
              ))}
            </div>
          )}
          
          {/* Show context indicators */}
          {msg.contextUsed && (
            <div className="mt-2 text-xs text-gray-400">
              {msg.forecastAvailable && '📈 Forecast data analyzed'}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50 h-96">
      <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
        <MessageSquare size={20} className="text-purple-400" />
        Supply Chain AI Assistant
        {(forecastData || dataAnalysis) && (
          <span className="text-xs bg-green-900/50 text-green-300 px-2 py-1 rounded">
            Data Connected
          </span>
        )}
      </h3>
      
      <div className="flex flex-col h-full">
        <div className="flex-1 overflow-y-auto mb-4 space-y-3">
          {messages.length === 0 ? (
            <div className="text-center text-gray-400 py-8">
              <MessageSquare size={48} className="mx-auto mb-4 opacity-50" />
              <p>Ask me about supply chain optimization, inventory management, or demand forecasting.</p>
              {(forecastData || dataAnalysis) && (
                <p className="text-sm text-green-400 mt-2">
                  I have access to your forecast data and analysis results.
                </p>
              )}
            </div>
          ) : (
            messages.map(renderMessage)
          )}
          
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-700 text-gray-100 px-4 py-2 rounded-lg flex items-center gap-2">
                <Loader2 size={16} className="animate-spin" />
                Analyzing your data...
              </div>
            </div>
          )}
        </div>
        
        <div className="flex gap-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Ask about supply chain optimization..."
            className="flex-1 bg-gray-900/50 border border-gray-600 rounded-lg p-3 text-white placeholder-gray-400 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
            onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleSendMessage()}
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={isLoading}
            className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-500 text-white rounded-lg hover:from-cyan-600 hover:to-blue-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatAgent;