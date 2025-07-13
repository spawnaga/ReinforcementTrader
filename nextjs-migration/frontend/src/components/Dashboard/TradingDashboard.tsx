/**
 * Main Trading Dashboard Component
 * Migrated from enhanced_trading_dashboard.html
 */
import React, { useEffect, useState } from 'react';
import { useTradingStore } from '@/store/tradingStore';
import { useWebSocket } from '@/hooks/useWebSocket';
import { AlgorithmType } from '@/types';
import TrainingChart from './TrainingChart';
import PerformanceMetrics from './PerformanceMetrics';
import TradeList from './TradeList';
import NeuralNetworkViz from './NeuralNetworkViz';

interface TradingDashboardProps {
  className?: string;
}

const TradingDashboard: React.FC<TradingDashboardProps> = ({ className }) => {
  const {
    activeSession,
    isTraining,
    performanceMetrics,
    setIsTraining,
    addTrainingMetric,
    addTrade,
    updatePerformanceMetrics,
  } = useTradingStore();

  const { isConnected, sendMessage } = useWebSocket({
    onTrainingUpdate: addTrainingMetric,
    onTradeUpdate: addTrade,
    onPerformanceUpdate: updatePerformanceMetrics,
  });

  const [selectedAlgorithm, setSelectedAlgorithm] = useState<AlgorithmType>(
    AlgorithmType.ANE_PPO
  );
  const [totalEpisodes, setTotalEpisodes] = useState(1000);
  const [transformerLayers, setTransformerLayers] = useState(2);
  const [attentionDim, setAttentionDim] = useState(256);

  const handleStartTraining = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          algorithm_type: selectedAlgorithm,
          total_episodes: totalEpisodes,
          transformer_layers: transformerLayers,
          attention_dim: attentionDim,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setIsTraining(true);
        // Join session room via WebSocket
        sendMessage({
          type: 'join_session',
          session_id: data.session_id,
        });
      }
    } catch (error) {
      console.error('Failed to start training:', error);
    }
  };

  const handleStopTraining = async () => {
    if (!activeSession) return;

    try {
      await fetch(`http://localhost:8000/api/training/${activeSession.id}/stop`, {
        method: 'POST',
      });
      setIsTraining(false);
    } catch (error) {
      console.error('Failed to stop training:', error);
    }
  };

  return (
    <div className={`min-h-screen bg-gray-900 text-white ${className}`}>
      <div className="container mx-auto p-4">
        {/* Header */}
        <header className="mb-8">
          <h1 className="text-4xl font-bold mb-2">
            AI Trading System
          </h1>
          <div className="flex items-center gap-4">
            <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-sm">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
            {isTraining && (
              <span className="ml-4 text-yellow-400">
                Training in Progress...
              </span>
            )}
          </div>
        </header>

        {/* Control Panel */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Training Controls</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Algorithm
                </label>
                <select
                  value={selectedAlgorithm}
                  onChange={(e) => setSelectedAlgorithm(e.target.value as AlgorithmType)}
                  className="w-full px-3 py-2 bg-gray-700 rounded-md"
                  disabled={isTraining}
                >
                  <option value={AlgorithmType.ANE_PPO}>ANE-PPO</option>
                  <option value={AlgorithmType.DQN}>DQN</option>
                  <option value={AlgorithmType.GENETIC}>Genetic</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Episodes: {totalEpisodes}
                </label>
                <input
                  type="range"
                  min="10"
                  max="10000"
                  step="10"
                  value={totalEpisodes}
                  onChange={(e) => setTotalEpisodes(Number(e.target.value))}
                  className="w-full"
                  disabled={isTraining}
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Transformer Layers: {transformerLayers}
                </label>
                <input
                  type="range"
                  min="1"
                  max="8"
                  value={transformerLayers}
                  onChange={(e) => setTransformerLayers(Number(e.target.value))}
                  className="w-full"
                  disabled={isTraining}
                />
              </div>

              <button
                onClick={isTraining ? handleStopTraining : handleStartTraining}
                className={`w-full py-2 px-4 rounded-md font-medium transition-colors ${
                  isTraining
                    ? 'bg-red-600 hover:bg-red-700'
                    : 'bg-green-600 hover:bg-green-700'
                }`}
              >
                {isTraining ? 'Stop Training' : 'Start Training'}
              </button>
            </div>
          </div>

          {/* Performance Metrics */}
          <PerformanceMetrics metrics={performanceMetrics} />

          {/* Neural Network Visualization */}
          <NeuralNetworkViz
            layers={transformerLayers}
            dimension={attentionDim}
          />
        </div>

        {/* Charts and Data */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <TrainingChart />
          <TradeList sessionId={activeSession?.id} />
        </div>
      </div>
    </div>
  );
};

export default TradingDashboard;