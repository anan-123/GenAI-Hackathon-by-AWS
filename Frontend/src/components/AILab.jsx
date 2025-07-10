import React, { useState, useEffect } from 'react';
import { Brain, Cpu, Play, Save, Trash2, Plus, Settings, Clock, CheckCircle, XCircle, Loader } from 'lucide-react';

const AILab = () => {
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [isCreating, setIsCreating] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState(null);
  const [executionHistory, setExecutionHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  

  const [agentForm, setAgentForm] = useState({
    name: '',
    description: '',
    systemPrompt: '',
    code: ''
  });
  
  const [executionForm, setExecutionForm] = useState({
    input: '',
    context: ''
  });

  const API_BASE = 'http://localhost:8000/api';


  useEffect(() => {
    fetchAgents();
  }, []);

  useEffect(() => {
    if (selectedAgent) {
      fetchExecutionHistory(selectedAgent.id);
    }
  }, [selectedAgent]);

  const fetchAgents = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/agents`);
      const data = await response.json();
      if (data.success) {
        setAgents(data.agents);
      }
    } catch (error) {
      console.error('Error fetching agents:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchExecutionHistory = async (agentId) => {
    try {
      const response = await fetch(`${API_BASE}/agents/${agentId}/executions`);
      const data = await response.json();
      if (data.success) {
        setExecutionHistory(data.executions);
      }
    } catch (error) {
      console.error('Error fetching execution history:', error);
    }
  };

  const createAgent = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE}/agents`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(agentForm),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setAgents(prevAgents => [...prevAgents, data.agent]);
        setAgentForm({ name: '', description: '', systemPrompt: '', code: '' });
   
        setIsCreating(false);
        setSuccess('Agent created successfully!');
       
        setSelectedAgent(data.agent);
      
        setTimeout(() => setSuccess(null), 3000);
      } else {
        setError(data.error || 'Failed to create agent');
      }
    } catch (error) {
      console.error('Error creating agent:', error);
      setError('Network error: Unable to create agent');
    } finally {
      setLoading(false);
    }
  };

  const executeAgent = async () => {
    if (!selectedAgent || !executionForm.input) return;
    
    try {
      setIsExecuting(true);
      setExecutionResult(null);
      setError(null);
      
      const response = await fetch(`${API_BASE}/agents/${selectedAgent.id}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(executionForm),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setExecutionResult(data.execution);
     
        setSelectedAgent(prev => ({
          ...prev,
          executionCount: prev.executionCount + 1,
          lastExecuted: new Date().toISOString()
        }));
       setAgents(prevAgents => 
          prevAgents.map(agent => 
            agent.id === selectedAgent.id 
              ? { ...agent, executionCount: agent.executionCount + 1, lastExecuted: new Date().toISOString() }
              : agent
          )
        );
        fetchExecutionHistory(selectedAgent.id);
  
        setExecutionForm({ input: '', context: '' });
        setSuccess('Agent executed successfully!');
        setTimeout(() => setSuccess(null), 3000);
      } else {
        setError(data.error || 'Failed to execute agent');
      }
    } catch (error) {
      console.error('Error executing agent:', error);
      setError('Network error: Unable to execute agent');
    } finally {
      setIsExecuting(false);
    }
  };

  const deleteAgent = async (agentId) => {
    try {
      setError(null);
      const response = await fetch(`${API_BASE}/agents/${agentId}`, {
        method: 'DELETE',
      });
      
      const data = await response.json();
      if (data.success) {
        setAgents(prevAgents => prevAgents.filter(agent => agent.id !== agentId));
        if (selectedAgent && selectedAgent.id === agentId) {
          setSelectedAgent(null);
          setExecutionHistory([]);
          setExecutionResult(null);
        }
        setSuccess('Agent deleted successfully!');
        setTimeout(() => setSuccess(null), 3000);
      } else {
        setError(data.error || 'Failed to delete agent');
      }
    } catch (error) {
      console.error('Error deleting agent:', error);
      setError('Network error: Unable to delete agent');
    }
  };

  const AgentCard = ({ agent, isSelected, onClick }) => (
    <div
      onClick={onClick}
      className={`bg-gray-800/50 backdrop-blur-sm rounded-xl p-4 border cursor-pointer transition-all hover:scale-105 ${
        isSelected ? 'border-cyan-500 bg-cyan-900/20' : 'border-gray-700/50 hover:border-gray-600'
      }`}
    >
      <div className="flex items-start justify-between mb-2">
        <h4 className="font-semibold text-white">{agent.name}</h4>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">{agent.executionCount} runs</span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              deleteAgent(agent.id);
            }}
            className="text-red-400 hover:text-red-300 transition-colors"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>
      <p className="text-sm text-gray-300 mb-3">{agent.description}</p>
      <div className="flex items-center gap-2 text-xs text-gray-400">
        <Clock size={12} />
        <span>{new Date(agent.createdAt).toLocaleDateString()}</span>
        <div className={`w-2 h-2 rounded-full ${agent.status === 'active' ? 'bg-green-400' : 'bg-gray-400'}`} />
        <span>{agent.status}</span>
      </div>
    </div>
  );

  if (loading && agents.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader className="animate-spin text-cyan-400" size={48} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Success/Error Messages */}
      {success && (
        <div className="bg-green-500/20 border border-green-500/50 rounded-lg p-4 flex items-center gap-2">
          <CheckCircle size={20} className="text-green-400" />
          <span className="text-green-300">{success}</span>
        </div>
      )}
      
      {error && (
        <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 flex items-center gap-2">
          <XCircle size={20} className="text-red-400" />
          <span className="text-red-300">{error}</span>
          <button 
            onClick={() => setError(null)}
            className="ml-auto text-red-400 hover:text-red-300"
          >
            ×
          </button>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <Brain className="text-pink-400" size={24} />
          AI Agent Lab
          <span className="text-sm text-gray-400 font-normal">
            ({agents.length} agent{agents.length !== 1 ? 's' : ''})
          </span>
        </h2>
        <button
          onClick={() => setIsCreating(true)}
          className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:from-purple-600 hover:to-pink-600 transition-all flex items-center gap-2"
        >
          <Plus size={18} />
          Create Agent
        </button>
      </div>

      {/* Create Agent Modal */}
      {isCreating && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-xl p-6 w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">Create New AI Agent</h3>
              <button
                onClick={() => setIsCreating(false)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                ×
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Agent Name <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  value={agentForm.name}
                  onChange={(e) => setAgentForm({ ...agentForm, name: e.target.value })}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg p-3 text-white focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
                  placeholder="e.g., Supply Chain Optimizer"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Description <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  value={agentForm.description}
                  onChange={(e) => setAgentForm({ ...agentForm, description: e.target.value })}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg p-3 text-white focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
                  placeholder="Brief description of what this agent does"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  System Prompt
                  <span className="text-xs text-gray-500 ml-2">(Will be enhanced by Claude)</span>
                </label>
                <textarea
                  value={agentForm.systemPrompt}
                  onChange={(e) => setAgentForm({ ...agentForm, systemPrompt: e.target.value })}
                  className="w-full h-32 bg-gray-700 border border-gray-600 rounded-lg p-3 text-white focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 resize-none"
                  placeholder="Define the agent's role, capabilities, and behavior..."
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Code Context (Optional)
                </label>
                <textarea
                  value={agentForm.code}
                  onChange={(e) => setAgentForm({ ...agentForm, code: e.target.value })}
                  className="w-full h-32 bg-gray-900 border border-gray-600 rounded-lg p-3 text-green-400 font-mono text-sm focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 resize-none"
                  placeholder="// Optional code context for the agent"
                />
              </div>
            </div>
            
            <div className="flex gap-2 mt-6">
              <button
                onClick={createAgent}
                disabled={!agentForm.name || !agentForm.description || loading}
                className="flex-1 px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-lg hover:from-green-600 hover:to-emerald-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader size={16} className="animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Save size={16} />
                    Create Agent
                  </>
                )}
              </button>
              <button
                onClick={() => {
                  setIsCreating(false);
                  setAgentForm({ name: '', description: '', systemPrompt: '', code: '' });
                }}
                className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-all"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Agents List */}
        <div className="lg:col-span-1">
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white flex items-center gap-2">
                <Settings size={20} className="text-cyan-400" />
                Your Agents
              </h3>
              <button
                onClick={fetchAgents}
                className="text-gray-400 hover:text-white transition-colors"
                title="Refresh agents"
              >
                <Loader size={16} className={loading ? 'animate-spin' : ''} />
              </button>
            </div>
            
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {agents.map(agent => (
                <AgentCard
                  key={agent.id}
                  agent={agent}
                  isSelected={selectedAgent?.id === agent.id}
                  onClick={() => setSelectedAgent(agent)}
                />
              ))}
              
              {agents.length === 0 && !loading && (
                <div className="text-center py-8 text-gray-400">
                  <Brain size={48} className="mx-auto mb-2 opacity-50" />
                  <p className="mb-2">No agents created yet</p>
                  <button
                    onClick={() => setIsCreating(true)}
                    className="text-cyan-400 hover:text-cyan-300 transition-colors text-sm"
                  >
                    Create your first agent
                  </button>
                </div>
              )}
              
              {loading && agents.length === 0 && (
                <div className="text-center py-8 text-gray-400">
                  <Loader size={48} className="mx-auto mb-2 animate-spin" />
                  <p>Loading agents...</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Execution Panel */}
        <div className="lg:col-span-2">
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
            <h3 className="text-xl font-semibold mb-4 text-white flex items-center gap-2">
              <Cpu size={20} className="text-orange-400" />
              Agent Execution
            </h3>
            
            {selectedAgent ? (
              <div className="space-y-4">
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <h4 className="font-semibold text-white mb-2">{selectedAgent.name}</h4>
                  <p className="text-gray-300 text-sm">{selectedAgent.description}</p>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Input</label>
                    <textarea
                      value={executionForm.input}
                      onChange={(e) => setExecutionForm({ ...executionForm, input: e.target.value })}
                      className="w-full h-24 bg-gray-700 border border-gray-600 rounded-lg p-3 text-white focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
                      placeholder="Enter your task or question..."
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Context (Optional)</label>
                    <textarea
                      value={executionForm.context}
                      onChange={(e) => setExecutionForm({ ...executionForm, context: e.target.value })}
                      className="w-full h-24 bg-gray-700 border border-gray-600 rounded-lg p-3 text-white focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
                      placeholder="Additional context or data..."
                    />
                  </div>
                </div>
                
                <button
                  onClick={executeAgent}
                  disabled={!executionForm.input || isExecuting}
                  className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg hover:from-blue-600 hover:to-purple-600 transition-all disabled:opacity-50 flex items-center gap-2"
                >
                  {isExecuting ? (
                    <>
                      <Loader size={18} className="animate-spin" />
                      Executing...
                    </>
                  ) : (
                    <>
                      <Play size={18} />
                      Execute Agent
                    </>
                  )}
                </button>
                
                {/* Execution Result */}
                {executionResult && (
                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-3">
                      {executionResult.success ? (
                        <CheckCircle size={20} className="text-green-400" />
                      ) : (
                        <XCircle size={20} className="text-red-400" />
                      )}
                      <span className="font-medium text-white">
                        {executionResult.success ? 'Success' : 'Error'}
                      </span>
                      <span className="text-gray-400 text-sm">
                        ({executionResult.executionTime}ms)
                      </span>
                    </div>
                    <div className="bg-gray-900 rounded-lg p-3">
                      <pre className="text-sm text-gray-300 whitespace-pre-wrap">
                        {executionResult.output}
                      </pre>
                    </div>
                  </div>
                )}
                
                {/* Execution History */}
                {executionHistory.length > 0 && (
                  <div className="mt-6">
                    <h4 className="font-semibold text-white mb-3">Recent Executions</h4>
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {executionHistory.slice(0, 5).map(execution => (
                        <div key={execution.id} className="bg-gray-700/30 rounded-lg p-3">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              {execution.success ? (
                                <CheckCircle size={16} className="text-green-400" />
                              ) : (
                                <XCircle size={16} className="text-red-400" />
                              )}
                              <span className="text-sm text-gray-400">
                                {new Date(execution.timestamp).toLocaleString()}
                              </span>
                            </div>
                            <span className="text-xs text-gray-500">
                              {execution.executionTime}ms
                            </span>
                          </div>
                          <p className="text-sm text-gray-300 truncate">
                            {execution.input}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-16 text-gray-400">
                <Cpu size={48} className="mx-auto mb-4 opacity-50" />
                <p>Select an agent to execute</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AILab;