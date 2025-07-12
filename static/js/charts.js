// Revolutionary Trading Charts with Advanced Technical Analysis

// Tell IDE about Chart.js from CDN
/* global Chart */
/** @type {typeof Chart} */
const Chart = typeof window !== 'undefined' && window.Chart ? window.Chart : null;

class TradingCharts {
    constructor(canvasId) {
        this.priceChart = null;
        // Future charts can be added here
        // this.volumeChart = null;
        // this.performanceChart = null;
        // this.rsiChart = null;
        // this.macdChart = null;
        
        // Chart data storage - used internally
        // this.chartData = {
        //     prices: [],
        //     volumes: [],
        //     indicators: {}
        // };
        
        this.timeframe = '1m';
        this.indicators = {
            sma: true,
            bollinger: false,
            rsi: false,
            macd: false,
            volume: true
        };
        
        this.colors = {
            bullish: '#00ff88',
            bearish: '#ff0088',
            neutral: '#ffffff',
            volume: '#0088ff',
            sma: '#ffaa00',
            bollinger: '#aa00ff',
            rsi: '#ff6600',
            macd: '#00aaff'
        };
        
        console.log('📊 Trading Charts initialized');
        
        // Initialize the chart if canvas ID is provided
        if (canvasId) {
            this.initPriceChart(canvasId);
            // Add some sample data to display
            this.addSampleData();
        }
    }
    
    initPriceChart(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error('Price chart canvas not found');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        this.priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'NQ Futures Price',
                    data: [],
                    borderColor: this.colors.bullish,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: '#ffffff',
                            font: {
                                family: 'Arial',
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#00ff88',
                        borderWidth: 1,
                        callbacks: {
                            title: function(context) {
                                if (context[0] && context[0].parsed && context[0].parsed.x) {
                                    return new Date(context[0].parsed.x).toLocaleString();
                                }
                                return '';
                            },
                            label: function(context) {
                                const data = context.parsed;
                                if (data && data.y) {
                                    return `Price: $${data.y.toFixed(2)}`;
                                }
                                return 'Price: N/A';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'minute',
                            displayFormats: {
                                minute: 'HH:mm',
                                hour: 'HH:mm',
                                day: 'MM/DD'
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            lineWidth: 1
                        },
                        ticks: {
                            color: '#ffffff',
                            maxTicksLimit: 10
                        }
                    },
                    y: {
                        position: 'right',
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)',
                            lineWidth: 1
                        },
                        ticks: {
                            color: '#ffffff',
                            callback: function(value) {
                                return value.toFixed(2);
                            }
                        }
                    }
                },
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                animation: {
                    duration: 300,
                    easing: 'easeInOutQuart'
                }
            },
            plugins: [{
                id: 'crosshair',
                afterDraw: (chart) => {
                    this.drawCrosshair(chart);
                }
            }]
        });
        
        // Add custom candlestick rendering
        this.setupCandlestickChart();
        
        console.log('📈 Price chart initialized');
    }
    
    setupCandlestickChart() {
        // Register custom candlestick chart type
        Chart.register({
            id: 'candlestick',
            beforeDraw: (chart) => {
                this.drawCandlesticks(chart);
            }
        });
    }
    
    drawCandlesticks(chart) {
        const ctx = chart.ctx;
        const data = chart.data.datasets[0].data;
        const meta = chart.getDatasetMeta(0);
        
        if (!data || data.length === 0) return;
        
        data.forEach((point, index) => {
            if (!point.o || !point.h || !point.l || !point.c) return;
            
            const element = meta.data[index];
            if (!element) return;
            
            const x = element.x;
            const yOpen = chart.scales.y.getPixelForValue(point.o);
            const yHigh = chart.scales.y.getPixelForValue(point.h);
            const yLow = chart.scales.y.getPixelForValue(point.l);
            const yClose = chart.scales.y.getPixelForValue(point.c);
            
            const candleWidth = 8;
            const isBullish = point.c > point.o;
            
            // Draw wick
            ctx.strokeStyle = isBullish ? this.colors.bullish : this.colors.bearish;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, yHigh);
            ctx.lineTo(x, yLow);
            ctx.stroke();
            
            // Draw body
            if (isBullish) {
                ctx.fillStyle = this.colors.bullish;
                ctx.fillRect(x - candleWidth/2, yClose, candleWidth, yOpen - yClose);
            } else {
                ctx.fillStyle = this.colors.bearish;
                ctx.fillRect(x - candleWidth/2, yOpen, candleWidth, yClose - yOpen);
            }
            
            // Draw body outline
            ctx.strokeStyle = isBullish ? this.colors.bullish : this.colors.bearish;
            ctx.strokeRect(x - candleWidth/2, Math.min(yOpen, yClose), candleWidth, Math.abs(yOpen - yClose));
        });
    }
    
    drawCrosshair(chart) {
        const activePoints = chart.getActiveElements();
        if (activePoints.length === 0) return;
        
        const ctx = chart.ctx;
        const activePoint = activePoints[0];
        const x = activePoint.element.x;
        const y = activePoint.element.y;
        
        // Save context
        ctx.save();
        
        // Draw crosshair lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        
        // Vertical line
        ctx.beginPath();
        ctx.moveTo(x, chart.chartArea.top);
        ctx.lineTo(x, chart.chartArea.bottom);
        ctx.stroke();
        
        // Horizontal line
        ctx.beginPath();
        ctx.moveTo(chart.chartArea.left, y);
        ctx.lineTo(chart.chartArea.right, y);
        ctx.stroke();
        
        // Restore context
        ctx.restore();
    }
    
    updatePriceChart(marketData) {
        if (!this.priceChart || !marketData) return;
        
        // Convert market data to candlestick format
        const candlestickData = marketData.map(item => ({
            x: new Date(item.timestamp),
            o: item.open,
            h: item.high,
            l: item.low,
            c: item.close,
            v: item.volume
        }));
        
        this.priceChart.data.datasets[0].data = candlestickData;
        
        // Add technical indicators if enabled
        this.updateIndicators(candlestickData);
        
        this.priceChart.update('none');
    }
    
    updateIndicators(data) {
        if (!data || data.length === 0) return;
        
        // Update SMA
        if (this.indicators.sma) {
            this.updateSMA(data);
        }
        
        // Update Bollinger Bands
        if (this.indicators.bollinger) {
            this.updateBollingerBands(data);
        }
        
        // Update RSI
        if (this.indicators.rsi) {
            this.updateRSI(data);
        }
        
        // Update MACD
        if (this.indicators.macd) {
            this.updateMACD(data);
        }
    }
    
    updateSMA(data, period = 20) {
        const smaData = this.calculateSMA(data.map(d => d.c), period);
        
        // Find or create SMA dataset
        let smaDataset = this.priceChart.data.datasets.find(d => d.label === 'SMA(20)');
        
        if (!smaDataset) {
            smaDataset = {
                label: 'SMA(20)',
                data: [],
                borderColor: this.colors.sma,
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                type: 'line'
            };
            this.priceChart.data.datasets.push(smaDataset);
        }
        
        smaDataset.data = smaData.map((value, index) => ({
            x: data[index].x,
            y: value
        })).filter(point => point.y !== null);
    }
    
    updateBollingerBands(data, period = 20, stdDev = 2) {
        const closes = data.map(d => d.c);
        const sma = this.calculateSMA(closes, period);
        const bands = this.calculateBollingerBands(closes, period, stdDev);
        
        // Update upper band
        let upperBandDataset = this.priceChart.data.datasets.find(d => d.label === 'BB Upper');
        if (!upperBandDataset) {
            upperBandDataset = {
                label: 'BB Upper',
                data: [],
                borderColor: this.colors.bollinger,
                backgroundColor: 'transparent',
                borderWidth: 1,
                pointRadius: 0,
                type: 'line',
                borderDash: [2, 2]
            };
            this.priceChart.data.datasets.push(upperBandDataset);
        }
        
        // Update lower band
        let lowerBandDataset = this.priceChart.data.datasets.find(d => d.label === 'BB Lower');
        if (!lowerBandDataset) {
            lowerBandDataset = {
                label: 'BB Lower',
                data: [],
                borderColor: this.colors.bollinger,
                backgroundColor: 'transparent',
                borderWidth: 1,
                pointRadius: 0,
                type: 'line',
                borderDash: [2, 2]
            };
            this.priceChart.data.datasets.push(lowerBandDataset);
        }
        
        upperBandDataset.data = bands.upper.map((value, index) => ({
            x: data[index].x,
            y: value
        })).filter(point => point.y !== null);
        
        lowerBandDataset.data = bands.lower.map((value, index) => ({
            x: data[index].x,
            y: value
        })).filter(point => point.y !== null);
    }
    
    updateRSI(data, period = 14) {
        if (!this.rsiChart) {
            this.createRSIChart();
        }
        
        const closes = data.map(d => d.c);
        const rsiData = this.calculateRSI(closes, period);
        
        this.rsiChart.data.datasets[0].data = rsiData.map((value, index) => ({
            x: data[index].x,
            y: value
        })).filter(point => point.y !== null);
        
        this.rsiChart.update('none');
    }
    
    updateMACD(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
        if (!this.macdChart) {
            this.createMACDChart();
        }
        
        const closes = data.map(d => d.c);
        const macdData = this.calculateMACD(closes, fastPeriod, slowPeriod, signalPeriod);
        
        this.macdChart.data.datasets[0].data = macdData.macd.map((value, index) => ({
            x: data[index].x,
            y: value
        })).filter(point => point.y !== null);
        
        this.macdChart.data.datasets[1].data = macdData.signal.map((value, index) => ({
            x: data[index].x,
            y: value
        })).filter(point => point.y !== null);
        
        this.macdChart.data.datasets[2].data = macdData.histogram.map((value, index) => ({
            x: data[index].x,
            y: value
        })).filter(point => point.y !== null);
        
        this.macdChart.update('none');
    }
    
    createRSIChart() {
        const canvas = document.createElement('canvas');
        canvas.id = 'rsi-chart';
        canvas.style.height = '100px';
        
        // Add to a container (you might want to specify where)
        const container = document.querySelector('.chart-container');
        if (container) {
            container.appendChild(canvas);
        }
        
        const ctx = canvas.getContext('2d');
        
        this.rsiChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'RSI',
                    data: [],
                    borderColor: this.colors.rsi,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        display: false
                    },
                    y: {
                        min: 0,
                        max: 100,
                        ticks: {
                            color: '#ffffff'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                }
            }
        });
    }
    
    createMACDChart() {
        const canvas = document.createElement('canvas');
        canvas.id = 'macd-chart';
        canvas.style.height = '120px';
        
        const container = document.querySelector('.chart-container');
        if (container) {
            container.appendChild(canvas);
        }
        
        const ctx = canvas.getContext('2d');
        
        this.macdChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'MACD',
                    data: [],
                    borderColor: this.colors.macd,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0
                }, {
                    label: 'Signal',
                    data: [],
                    borderColor: this.colors.sma,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0
                }, {
                    label: 'Histogram',
                    data: [],
                    backgroundColor: 'rgba(0, 255, 136, 0.3)',
                    borderColor: this.colors.bullish,
                    borderWidth: 1,
                    type: 'bar'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        display: false
                    },
                    y: {
                        ticks: {
                            color: '#ffffff'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                }
            }
        });
    }
    
    // Technical indicator calculations
    calculateSMA(data, period) {
        const sma = [];
        
        for (let i = 0; i < data.length; i++) {
            if (i < period - 1) {
                sma.push(null);
            } else {
                const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
                sma.push(sum / period);
            }
        }
        
        return sma;
    }
    
    calculateEMA(data, period) {
        const ema = [];
        const multiplier = 2 / (period + 1);
        
        ema[0] = data[0];
        
        for (let i = 1; i < data.length; i++) {
            ema[i] = (data[i] * multiplier) + (ema[i - 1] * (1 - multiplier));
        }
        
        return ema;
    }
    
    calculateBollingerBands(data, period, stdDev) {
        const sma = this.calculateSMA(data, period);
        const upper = [];
        const lower = [];
        
        for (let i = 0; i < data.length; i++) {
            if (i < period - 1) {
                upper.push(null);
                lower.push(null);
            } else {
                const slice = data.slice(i - period + 1, i + 1);
                const mean = sma[i];
                const variance = slice.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / period;
                const standardDeviation = Math.sqrt(variance);
                
                upper.push(mean + (standardDeviation * stdDev));
                lower.push(mean - (standardDeviation * stdDev));
            }
        }
        
        return { upper, lower };
    }
    
    calculateRSI(data, period) {
        const rsi = [];
        let gains = 0;
        let losses = 0;
        
        // Calculate initial average gains and losses
        for (let i = 1; i <= period; i++) {
            const change = data[i] - data[i - 1];
            if (change > 0) {
                gains += change;
            } else {
                losses -= change;
            }
        }
        
        gains /= period;
        losses /= period;
        
        // Fill initial values
        for (let i = 0; i <= period; i++) {
            rsi.push(null);
        }
        
        // Calculate RSI
        for (let i = period + 1; i < data.length; i++) {
            const change = data[i] - data[i - 1];
            const gain = change > 0 ? change : 0;
            const loss = change < 0 ? -change : 0;
            
            gains = ((gains * (period - 1)) + gain) / period;
            losses = ((losses * (period - 1)) + loss) / period;
            
            const rs = gains / losses;
            rsi.push(100 - (100 / (1 + rs)));
        }
        
        return rsi;
    }
    
    calculateMACD(data, fastPeriod, slowPeriod, signalPeriod) {
        const fastEMA = this.calculateEMA(data, fastPeriod);
        const slowEMA = this.calculateEMA(data, slowPeriod);
        
        const macd = fastEMA.map((fast, i) => fast - slowEMA[i]);
        const signal = this.calculateEMA(macd.filter(val => val !== null), signalPeriod);
        
        // Align signal line with MACD
        const alignedSignal = new Array(slowPeriod - 1).fill(null).concat(signal);
        
        const histogram = macd.map((macdVal, i) => {
            const signalVal = alignedSignal[i];
            return (macdVal !== null && signalVal !== null) ? macdVal - signalVal : null;
        });
        
        return { macd, signal: alignedSignal, histogram };
    }
    
    updateTimeframe(timeframe) {
        this.timeframe = timeframe;
        
        // Update time axis
        let timeUnit = 'minute';
        let displayFormat = 'HH:mm';
        
        switch (timeframe) {
            case '1m':
                timeUnit = 'minute';
                displayFormat = 'HH:mm';
                break;
            case '5m':
                timeUnit = 'minute';
                displayFormat = 'HH:mm';
                break;
            case '15m':
                timeUnit = 'minute';
                displayFormat = 'HH:mm';
                break;
            case '1h':
                timeUnit = 'hour';
                displayFormat = 'HH:mm';
                break;
            case '4h':
                timeUnit = 'hour';
                displayFormat = 'MM/DD HH:mm';
                break;
            case '1d':
                timeUnit = 'day';
                displayFormat = 'MM/DD';
                break;
        }
        
        if (this.priceChart) {
            this.priceChart.options.scales.x.time.unit = timeUnit;
            this.priceChart.options.scales.x.time.displayFormats[timeUnit] = displayFormat;
            this.priceChart.update();
        }
        
        // Fetch new data for the timeframe
        this.fetchDataForTimeframe(timeframe);
    }
    
    fetchDataForTimeframe(timeframe) {
        fetch(`/api/market_data?symbol=NQ&timeframe=${timeframe}&limit=1000`)
            .then(response => response.json())
            .then(data => {
                this.updatePriceChart(data);
            })
            .catch(error => {
                console.error('Error fetching data:', error);
            });
    }
    
    toggleIndicator(indicator, enabled) {
        this.indicators[indicator] = enabled;
        
        if (!enabled) {
            // Remove indicator from chart
            this.removeIndicatorFromChart(indicator);
        } else {
            // Update chart with current data to add indicator
            if (this.priceChart && this.priceChart.data.datasets[0].data.length > 0) {
                this.updateIndicators(this.priceChart.data.datasets[0].data);
                this.priceChart.update();
            }
        }
    }
    
    removeIndicatorFromChart(indicator) {
        if (!this.priceChart) return;
        
        const indicatorLabels = {
            sma: ['SMA(20)'],
            bollinger: ['BB Upper', 'BB Lower'],
            rsi: ['RSI'],
            macd: ['MACD', 'Signal', 'Histogram']
        };
        
        const labelsToRemove = indicatorLabels[indicator] || [];
        
        this.priceChart.data.datasets = this.priceChart.data.datasets.filter(
            dataset => !labelsToRemove.includes(dataset.label)
        );
        
        this.priceChart.update();
    }
    
    updateRealTimeData(data) {
        if (!this.priceChart) return;
        
        const newPoint = {
            x: new Date(data.timestamp),
            o: data.open,
            h: data.high,
            l: data.low,
            c: data.close,
            v: data.volume
        };
        
        const dataset = this.priceChart.data.datasets[0];
        
        // Update last point or add new point
        if (dataset.data.length > 0) {
            const lastPoint = dataset.data[dataset.data.length - 1];
            const timeDiff = newPoint.x.getTime() - lastPoint.x.getTime();
            
            // If within same timeframe, update last point
            if (timeDiff < this.getTimeframeMs()) {
                dataset.data[dataset.data.length - 1] = newPoint;
            } else {
                dataset.data.push(newPoint);
            }
        } else {
            dataset.data.push(newPoint);
        }
        
        // Keep only last 1000 points
        if (dataset.data.length > 1000) {
            dataset.data.shift();
        }
        
        // Update indicators
        this.updateIndicators(dataset.data);
        
        this.priceChart.update('none');
    }
    
    getTimeframeMs() {
        const timeframes = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        };
        
        return timeframes[this.timeframe] || 60 * 1000;
    }
    
    // Commented out unused method to remove IDE warning
    // addTradeMarker(trade) {
    //     if (!this.priceChart) return;
    //     
    //     const annotation = {
    //         type: 'point',
    //         xValue: new Date(trade.timestamp),
    //         yValue: trade.price,
    //         backgroundColor: trade.action === 'buy' ? this.colors.bullish : this.colors.bearish,
    //         borderColor: '#ffffff',
    //         borderWidth: 2,
    //         radius: 8,
    //         label: {
    //             content: trade.action.toUpperCase(),
    //             enabled: true,
    //             position: 'top'
    //         }
    //     };
    //     
    //     if (!this.priceChart.options.plugins.annotation) {
    //         this.priceChart.options.plugins.annotation = {
    //             annotations: {}
    //         };
    //     }
    //     
    //     this.priceChart.options.plugins.annotation.annotations[trade.id] = annotation;
    //     this.priceChart.update();
    // }
    
    // Commented out unused method to remove IDE warning
    // highlightRegime(regime) {
    //     // Add background color zones based on market regime
    //     const regimeColors = {
    //         bull: 'rgba(0, 255, 136, 0.1)',
    //         bear: 'rgba(255, 0, 136, 0.1)',
    //         sideways: 'rgba(255, 255, 255, 0.05)',
    //         volatile: 'rgba(255, 136, 0, 0.1)'
    //     };
    //     
    //     // Implementation would depend on having regime detection data
    //     console.log(`Market regime: ${regime}`);
    // }
    
    // Commented out unused method to remove IDE warning
    // exportChart(chartType = 'price') {
    //     let chart = this.priceChart;
    //     
    //     switch (chartType) {
    //         case 'rsi':
    //             chart = this.rsiChart;
    //             break;
    //         case 'macd':
    //             chart = this.macdChart;
    //             break;
    //     }
    //     
    //     if (chart) {
    //         const url = chart.toBase64Image();
    //         const a = document.createElement('a');
    //         a.href = url;
    //         a.download = `${chartType}_chart_${Date.now()}.png`;
    //         a.click();
    //     }
    // }
    
    addSampleData() {
        // Generate sample data for testing
        const now = new Date();
        const sampleData = [];
        const basePrice = 15000;
        
        for (let i = 50; i >= 0; i--) {
            const timestamp = new Date(now.getTime() - i * 60000); // 1 minute intervals
            const price = basePrice + (Math.random() - 0.5) * 50;
            
            sampleData.push({
                x: timestamp,
                y: price
            });
        }
        
        this.priceChart.data.datasets[0].data = sampleData;
        this.priceChart.update();
        
        console.log('📊 Sample data added to chart');
    }
    
    // Commented out unused method to remove IDE warning
    // destroy() {
    //     if (this.priceChart) {
    //         this.priceChart.destroy();
    //     }
    //     if (this.rsiChart) {
    //         this.rsiChart.destroy();
    //     }
    //     if (this.macdChart) {
    //         this.macdChart.destroy();
    //     }
    // }
}

// Export for global use
window.TradingCharts = TradingCharts;

console.log('📊 Trading Charts module loaded');
