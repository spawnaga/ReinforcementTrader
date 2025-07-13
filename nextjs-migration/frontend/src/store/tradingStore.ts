/**
 * Zustand store for trading application state management
 */
import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';
import { 
  TrainingSession, 
  Trade, 
  PerformanceMetrics, 
  TrainingMetrics,
  SessionStatus,
  AlgorithmType
} from '@/types';

interface TradingState {
  // Session Management
  sessions: TrainingSession[];
  activeSession: TrainingSession | null;
  
  // Trading Data
  trades: Trade[];
  sessionTrades: Record<string, Trade[]>;
  
  // Performance Metrics
  performanceMetrics: PerformanceMetrics | null;
  trainingMetrics: TrainingMetrics[];
  
  // UI State
  isTraining: boolean;
  isConnected: boolean;
  
  // Actions
  setActiveSession: (session: TrainingSession | null) => void;
  addSession: (session: TrainingSession) => void;
  updateSession: (id: string, updates: Partial<TrainingSession>) => void;
  removeSession: (id: string) => void;
  
  addTrade: (trade: Trade) => void;
  setSessionTrades: (sessionId: string, trades: Trade[]) => void;
  
  updatePerformanceMetrics: (metrics: PerformanceMetrics) => void;
  addTrainingMetric: (metric: TrainingMetrics) => void;
  clearTrainingMetrics: () => void;
  
  setIsTraining: (isTraining: boolean) => void;
  setIsConnected: (isConnected: boolean) => void;
  
  reset: () => void;
}

const initialState = {
  sessions: [],
  activeSession: null,
  trades: [],
  sessionTrades: {},
  performanceMetrics: null,
  trainingMetrics: [],
  isTraining: false,
  isConnected: false,
};

export const useTradingStore = create<TradingState>()(
  devtools(
    subscribeWithSelector((set) => ({
      ...initialState,
      
      setActiveSession: (session) => set({ activeSession: session }),
      
      addSession: (session) => 
        set((state) => ({ 
          sessions: [...state.sessions, session] 
        })),
      
      updateSession: (id, updates) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, ...updates } : s
          ),
          activeSession:
            state.activeSession?.id === id
              ? { ...state.activeSession, ...updates }
              : state.activeSession,
        })),
      
      removeSession: (id) =>
        set((state) => ({
          sessions: state.sessions.filter((s) => s.id !== id),
          activeSession:
            state.activeSession?.id === id ? null : state.activeSession,
        })),
      
      addTrade: (trade) =>
        set((state) => {
          const sessionTrades = state.sessionTrades[trade.sessionId] || [];
          return {
            trades: [...state.trades, trade],
            sessionTrades: {
              ...state.sessionTrades,
              [trade.sessionId]: [...sessionTrades, trade],
            },
          };
        }),
      
      setSessionTrades: (sessionId, trades) =>
        set((state) => ({
          sessionTrades: {
            ...state.sessionTrades,
            [sessionId]: trades,
          },
        })),
      
      updatePerformanceMetrics: (metrics) =>
        set({ performanceMetrics: metrics }),
      
      addTrainingMetric: (metric) =>
        set((state) => ({
          trainingMetrics: [...state.trainingMetrics, metric].slice(-100), // Keep last 100
        })),
      
      clearTrainingMetrics: () => set({ trainingMetrics: [] }),
      
      setIsTraining: (isTraining) => set({ isTraining }),
      setIsConnected: (isConnected) => set({ isConnected }),
      
      reset: () => set(initialState),
    })),
    {
      name: 'trading-store',
    }
  )
);

// Selectors
export const selectActiveSession = (state: TradingState) => state.activeSession;
export const selectIsTraining = (state: TradingState) => state.isTraining;
export const selectSessionTrades = (sessionId: string) => (state: TradingState) => 
  state.sessionTrades[sessionId] || [];
export const selectLatestMetrics = (state: TradingState) => 
  state.trainingMetrics[state.trainingMetrics.length - 1];