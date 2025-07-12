// Type definitions for global classes and functions used across the trading dashboard

// Three.js related
declare const THREE: any;
declare const TWEEN: any;

// Chart.js
declare const Chart: any;

// Socket.IO
declare const io: any;

// Bootstrap
declare const bootstrap: any;

// Classes
declare class NeuralNetworkVisualization {
    constructor(canvasId: string);
    update(data: any): void;
    render(): void;
    init(): void;
    updateActivations(activations: any): void;
    updateWeights(weights: any): void;
    updateTrainingMetrics(metrics: any): void;
    toggleWireframe(): void;
    resetCamera(): void;
    clear(): void;
    animate(): void;
    updateIndicators(data: any): void;
}

declare class Portfolio3DVisualization {
    constructor(containerId: string);
    init(): void;
    updateVisualization(data: any): void;
    // updatePrice(price: number): void;
    // setPerformanceData(data: any): void;
    // setRiskData(data: any): void;
    // setPositions(positions: any[]): void;
    // toggleWireframe(): void;
    // resetCamera(): void;
    dispose(): void;
}

declare class TradingCharts {
    constructor(canvasId?: string);
    initPriceChart(canvasId: string): void;
    updatePriceChart(data: any[]): void;
    updateIndicators(data: any): void;
    // addTradeMarker(trade: any): void;
    // highlightRegime(regime: string): void;
    // exportChart(chartType?: string): void;
    clear(): void;
    // destroy(): void;
    updateSMA(data: any[], period?: number): void;
    updateBollingerBands(data: any[]): void;
    updateRSI(data: any[]): void;
    updateMACD(data: any[]): void;
    setTimeframe(timeframe: string): void;
    toggleIndicator(indicator: string): void;
}

declare class TradingDashboard {
    constructor();
    init(): void;
    loadSession(sessionId: string): void;
    startNewSession(): void;
    stopTraining(): void;
    updatePerformanceMetrics(data: any): void;
    updateNeuralNetworkVisualization(data: any): void;
    update3DPortfolio(data: any): void;
    updateCharts(data: any): void;
}

// Utility functions
declare function showAlert(message: string, type: string): void;
declare function showNotification(message: string, type: string): void;
declare function loadAvailableSessions(): void;
declare function showNewSessionModal(): void;
declare function createNewSession(): void;

// Global objects
declare const io: any;
declare const socket: any;
declare const performanceMetrics: any;
declare const Quadratic: any;
declare const InOut: any;
declare const Easing: any;

// Global instances
declare let dashboard: TradingDashboard;
declare let neuralNetworkViz: NeuralNetworkVisualization;
declare let portfolio3D: Portfolio3DVisualization;
declare let tradingCharts: TradingCharts;