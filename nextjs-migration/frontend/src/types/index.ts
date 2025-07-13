/**
 * TypeScript type definitions for AI Trading System
 */

// Enums
export enum AlgorithmType {
  ANE_PPO = 'ane-ppo',
  DQN = 'dqn',
  GENETIC = 'genetic'
}

export enum SessionStatus {
  IDLE = 'idle',
  RUNNING = 'running',
  PAUSED = 'paused',
  COMPLETED = 'completed',
  ERROR = 'error'
}

export enum PositionType {
  LONG = 'long',
  SHORT = 'short',
  FLAT = 'flat'
}

// Data Models
export interface TrainingSession {
  id: string;
  algorithmType: AlgorithmType;
  status: SessionStatus;
  totalEpisodes: number;
  currentEpisode: number;
  createdAt: Date;
  updatedAt: Date;
  symbol: string;
  config?: Record<string, any>;
}

export interface Trade {
  id: number;
  sessionId: string;
  timestamp: Date;
  entryTime: Date;
  exitTime?: Date;
  entryPrice: number;
  exitPrice?: number;
  positionType: PositionType;
  quantity: number;
  profit?: number;
  commission: number;
}

export interface MarketData {
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PerformanceMetrics {
  cpuUsage: number;
  memoryUsage: number;
  gpuUsage: number;
  networkIo: number;
  timestamp: Date;
}

export interface TrainingMetrics {
  episode: number;
  reward: number;
  loss: number;
  winRate?: number;
  profit?: number;
  timestamp: Date;
}

export interface SessionSummary {
  sessionId: string;
  totalTrades: number;
  profitableTrades: number;
  totalProfit: number;
  winRate: number;
  avgProfitPerTrade: number;
  maxDrawdown: number;
  sharpeRatio?: number;
}

// WebSocket Message Types
export interface WSMessage<T = any> {
  type: string;
  data: T;
  timestamp: Date;
}

export interface WSTrainingUpdate extends WSMessage<TrainingMetrics> {
  type: 'training_update';
}

export interface WSTradeUpdate extends WSMessage<Trade> {
  type: 'trade_update';
}

export interface WSPerformanceUpdate extends WSMessage<PerformanceMetrics> {
  type: 'performance_update';
}

// API Request/Response Types
export interface TrainingStartRequest {
  algorithmType?: AlgorithmType;
  totalEpisodes?: number;
  symbol?: string;
  transformerLayers?: number;
  attentionDim?: number;
  learningRate?: number;
}

export interface APIResponse<T = any> {
  success: boolean;
  message?: string;
  data?: T;
}

// Chart Data Types
export interface ChartDataPoint {
  x: number | Date;
  y: number;
}

export interface ChartDataset {
  label: string;
  data: ChartDataPoint[];
  borderColor?: string;
  backgroundColor?: string;
  tension?: number;
  yAxisID?: string;
}

// 3D Visualization Types
export interface Portfolio3DData {
  positions: Array<{
    symbol: string;
    value: number;
    change: number;
  }>;
  totalValue: number;
}

// Neural Network Visualization
export interface NeuralNetworkLayer {
  neurons: number;
  activation: string;
  type: 'input' | 'hidden' | 'output';
}

export interface NeuralNetworkConfig {
  layers: NeuralNetworkLayer[];
  connections: Array<{
    from: number;
    to: number;
    weight: number;
  }>;
}