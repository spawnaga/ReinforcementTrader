// Type definitions for global classes and functions used across the trading dashboard

declare class NeuralNetworkViz {
    constructor();
    update(data: any): void;
    render(): void;
}

declare class Portfolio3D {
    constructor();
    update(data: any): void;
    animate(): void;
}

declare class TradingCharts {
    constructor();
    updatePriceChart(data: any[]): void;
    updateIndicators(data: any): void;
    clear(): void;
}

declare function showAlert(message: string, type: string): void;
declare function showNotification(message: string, type: string): void;

declare const io: any;
declare const performanceMetrics: any;

// Global instances
declare let neuralNetworkViz: NeuralNetworkViz;
declare let portfolio3D: Portfolio3D;
declare let tradingCharts: TradingCharts;