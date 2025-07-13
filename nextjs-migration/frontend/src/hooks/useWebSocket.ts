/**
 * Custom WebSocket hook for real-time communication
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { 
  WSMessage, 
  TrainingMetrics, 
  Trade, 
  PerformanceMetrics 
} from '@/types';

interface UseWebSocketOptions {
  url?: string;
  onTrainingUpdate?: (data: TrainingMetrics) => void;
  onTradeUpdate?: (data: Trade) => void;
  onPerformanceUpdate?: (data: PerformanceMetrics) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

export const useWebSocket = ({
  url = 'ws://localhost:8000/ws',
  onTrainingUpdate,
  onTradeUpdate,
  onPerformanceUpdate,
  onConnect,
  onDisconnect,
}: UseWebSocketOptions = {}) => {
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef<Socket | null>(null);

  const sendMessage = useCallback((message: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.send(JSON.stringify(message));
    }
  }, []);

  useEffect(() => {
    // For Socket.IO compatibility, we'll use the Socket.IO client
    // In a pure WebSocket implementation, you'd use new WebSocket(url)
    const socket = io(url.replace('/ws', ''), {
      transports: ['websocket'],
      path: '/ws',
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      setIsConnected(true);
      onConnect?.();
      console.log('🔌 WebSocket connected');
    });

    socket.on('disconnect', () => {
      setIsConnected(false);
      onDisconnect?.();
      console.log('🔌 WebSocket disconnected');
    });

    // Handle different message types
    socket.on('message', (data: WSMessage) => {
      switch (data.type) {
        case 'training_update':
          onTrainingUpdate?.(data.data as TrainingMetrics);
          break;
        case 'trade_update':
          onTradeUpdate?.(data.data as Trade);
          break;
        case 'performance_update':
          onPerformanceUpdate?.(data.data as PerformanceMetrics);
          break;
        default:
          console.log('Unknown message type:', data.type);
      }
    });

    // Legacy event handlers for compatibility
    socket.on('training_update', (data: TrainingMetrics) => {
      onTrainingUpdate?.(data);
    });

    socket.on('trade_update', (data: Trade) => {
      onTradeUpdate?.(data);
    });

    socket.on('performance_update', (data: PerformanceMetrics) => {
      onPerformanceUpdate?.(data);
    });

    return () => {
      socket.disconnect();
    };
  }, [url, onTrainingUpdate, onTradeUpdate, onPerformanceUpdate, onConnect, onDisconnect]);

  return {
    isConnected,
    sendMessage,
    socket: socketRef.current,
  };
};