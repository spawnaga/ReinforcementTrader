// Revolutionary AI Trading Dashboard

/* global NeuralNetworkViz, Portfolio3D, TradingCharts, io, showAlert, showNotification, performanceMetrics */

/**
 * @typedef {Object} PerformanceMetrics
 * @property {number} totalReturn
 * @property {number} sharpeRatio
 * @property {number} maxDrawdown
 * @property {number} winRate
 * @property {number} totalTrades
 * @property {number} profitableTrades
 */

/**
 * @typedef {Object} Trade
 * @property {string} position_type
 * @property {number} profit_loss
 * @property {number} entry_price
 * @property {number} exit_price
 * @property {string} entry_time
 * @property {string} action
 * @property {string} symbol
 * @property {number} position_size
 * @property {string} risk_level
 * @property {number} current_price
 * @property {string} timestamp
 */

/**
 * @typedef {Object} TrainingUpdate
 * @property {number} episode
 * @property {number} reward
 * @property {number} sharpe_ratio
 * @property {number} total_return
 * @property {number} win_rate
 * @property {number} total_trades
 * @property {boolean} success
 * @property {string} error
 * @property {string} session_id
 */

// Declare global functions that are defined in other files
/**
 * @function showAlert
 * @param {string} message
 * @param {string} type
 */

/**
 * @function showNotification
 * @param {string} message
 * @param {string} type
 */

/**
 * @var {Function} io - Socket.IO client factory
 */

/**
 * @var {PerformanceMetrics} performanceMetrics - Global performance metrics
 */

/**
 * @class NeuralNetworkViz
 */

/**
 * @class Portfolio3D
 */

/**
 * @class TradingCharts
 */

// Initialize global instances that will be created after page load
let neuralNetworkViz = null;
let tradingCharts = null;
let portfolio3D = null;

/**
 * @typedef {Object} SessionData
 * @property {Object} session
 * @property {string} session.name
 * @property {string} session.algorithm_type
 * @property {number} session.current_episode
 * @property {number} session.total_episodes
 */

/**
 * @typedef {Object} RiskData
 * @property {string} level
 * @property {number} drawdown
 * @property {string} severity
 * @property {string} message
 */

/**
 * @typedef {Object} AIRecommendation
 * @property {string} action
 * @property {number} confidence
 * @property {string} timestamp
 */

class TradingDashboard {
    constructor() {
        this.currentSession = null;
        this.isLive = false;
        this.marketData = [];
        this.tradingEnabled = false;
        this.riskLevel = 'moderate';
        this.positionSize = 1;
        this.currentPrice = 0;
        this.aiRecommendations = [];
        this.performanceMetrics = {
            totalReturn: 0,
            sharpeRatio: 0,
            maxDrawdown: 0,
            winRate: 0,
            totalTrades: 0,
            profitableTrades: 0
        };
        
        this.socket = io();
        this.setupSocketListeners();
        this.setupEventHandlers();
        
        console.log('🎯 Trading Dashboard initialized');
    }
    
    init() {
        // Initialize global visualization instances
        if (typeof TradingCharts !== 'undefined') {
            tradingCharts = new TradingCharts('price-chart');
            window.tradingCharts = tradingCharts;
        }
        
        if (typeof NeuralNetworkVisualization !== 'undefined') {
            const nnContainer = document.getElementById('neural-network-viz');
            if (nnContainer) {
                neuralNetworkViz = new NeuralNetworkVisualization('neural-network-viz');
                neuralNetworkViz.init();
                window.neuralNetworkViz = neuralNetworkViz;
            }
        }
        
        if (typeof Portfolio3DVisualization !== 'undefined') {
            const p3dContainer = document.getElementById('portfolio-3d');
            if (p3dContainer) {
                portfolio3D = new Portfolio3DVisualization('portfolio-3d');
                portfolio3D.init();
                window.portfolio3D = portfolio3D;
            }
        }
        
        this.loadInitialData();
        this.loadRecentTrades();
        this.startRealTimeUpdates();
        this.setupKeyboardShortcuts();
        this.initializeAIAssistant();
        this.loadUserPreferences();
    }
    
    setupSocketListeners() {
        this.socket.on('training_update', (data) => {
            this.handleTrainingUpdate(data);
        });
        
        this.socket.on('market_data', (data) => {
            this.handleMarketData(data);
        });
        
        this.socket.on('trade_execution', (data) => {
            this.handleTradeExecution(data);
        });
        
        this.socket.on('ai_recommendation', (data) => {
            this.handleAIRecommendation(data);
        });
        
        this.socket.on('risk_alert', (data) => {
            this.handleRiskAlert(data);
        });
        
        this.socket.on('recent_trades_update', (data) => {
            this.updateRecentTrades(data);
        });
    }
    
    setupEventHandlers() {
        // Manual trading controls
        document.getElementById('manual-buy')?.addEventListener('click', () => {
            this.placeTrade('buy');
        });
        
        document.getElementById('manual-sell')?.addEventListener('click', () => {
            this.placeTrade('sell');
        });
        
        document.getElementById('manual-hold')?.addEventListener('click', () => {
            this.placeTrade('hold');
        });
        
        // Position size control
        document.getElementById('position-size')?.addEventListener('input', (e) => {
            const target = /** @type {HTMLInputElement} */ (e.target);
            this.positionSize = parseInt(target.value);
            this.updatePositionSizeDisplay();
        });
        
        // Risk level control
        document.getElementById('risk-level')?.addEventListener('change', (e) => {
            const target = /** @type {HTMLSelectElement} */ (e.target);
            this.riskLevel = target.value;
            this.updateRiskParameters();
        });
        
        // AI toggle
        document.getElementById('toggle-ai')?.addEventListener('click', () => {
            this.toggleAI();
        });
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey) {
                switch(e.key) {
                    case 'b':
                        e.preventDefault();
                        this.placeTrade('buy');
                        break;
                    case 's':
                        e.preventDefault();
                        this.placeTrade('sell');
                        break;
                    case 'h':
                        e.preventDefault();
                        this.placeTrade('hold');
                        break;
                    case 'a':
                        e.preventDefault();
                        this.toggleAI();
                        break;
                }
            }
        });
    }
    
    loadInitialData() {
        // Load market data
        fetch('/api/market_data?symbol=NQ&limit=1000')
            .then(response => response.json())
            .then(data => {
                this.marketData = data;
                this.updatePriceChart();
            })
            .catch(error => {
                console.error('Error loading market data:', error);
                showAlert('Failed to load market data', 'danger');
            });
        
        // Load current session if specified
        const urlParams = new URLSearchParams(window.location.search);
        const sessionId = urlParams.get('session_id');
        if (sessionId) {
            this.loadSession(sessionId);
        }
    }
    
    loadSession(sessionId) {
        this.currentSession = sessionId;
        
        fetch(`/api/session_status/${sessionId}`)
            .then(response => response.json())
            .then(data => {
                this.updateSessionDisplay(data);
                this.loadSessionMetrics(sessionId);
                this.loadRecentTrades();  // Load recent trades for this session
            })
            .catch(error => {
                console.error('Error loading session:', error);
                showAlert('Failed to load session data', 'danger');
            });
    }
    
    loadSessionMetrics(sessionId) {
        fetch(`/api/training_metrics/${sessionId}`)
            .then(response => response.json())
            .then(metrics => {
                this.updatePerformanceMetrics(metrics);
                this.updateRiskMetrics(metrics);
            })
            .catch(error => {
                console.error('Error loading metrics:', error);
            });
    }
    
    loadRecentTrades() {
        // Load recent trades for the current session
        if (this.currentSession) {
            fetch(`/api/recent_trades?session_id=${this.currentSession}`)
                .then(response => response.json())
                .then(trades => {
                    this.displayRecentTrades(trades);
                })
                .catch(error => {
                    console.error('Error loading recent trades:', error);
                });
        }
    }
    
    /**
     * @param {Trade[]} trades
     */
    displayRecentTrades(trades) {
        const tradesContainer = document.getElementById('recent-trades-list');
        const noTradesMessage = document.getElementById('no-trades-message');
        
        if (!tradesContainer || !noTradesMessage) return;
        
        if (trades && trades.length > 0) {
            // Hide no trades message
            noTradesMessage.style.display = 'none';
            
            // Clear existing trades
            tradesContainer.innerHTML = '';
            
            // Display trades
            trades.forEach((trade, index) => {
                const tradeItem = document.createElement('div');
                tradeItem.className = `list-group-item list-group-item-action ${index === 0 ? 'active' : ''}`;
                
                const profitClass = trade.profit_loss > 0 ? 'text-success' : 'text-danger';
                const profitPrefix = trade.profit_loss > 0 ? '+' : '';
                
                tradeItem.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <h6 class="mb-1">${trade.position_type || 'Unknown'}</h6>
                            <small class="text-muted">
                                Entry: $${trade.entry_price ? trade.entry_price.toFixed(2) : 'N/A'} | 
                                Exit: $${trade.exit_price ? trade.exit_price.toFixed(2) : 'N/A'}
                            </small>
                            <br>
                            <small class="text-muted">${this.formatTime(trade.entry_time)}</small>
                        </div>
                        <div class="text-end">
                            <span class="${profitClass} fw-bold">
                                ${profitPrefix}$${Math.abs(trade.profit_loss || 0).toFixed(2)}
                            </span>
                        </div>
                    </div>
                `;
                
                tradesContainer.appendChild(tradeItem);
            });
        } else {
            // Show no trades message
            noTradesMessage.style.display = 'block';
            tradesContainer.innerHTML = '';
        }
    }
    
    updateRecentTrades(data) {
        // Handle real-time trade updates
        if (data.trades) {
            this.displayRecentTrades(data.trades);
        }
    }
    
    formatTime(timestamp) {
        if (!timestamp) return 'N/A';
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit', 
            second: '2-digit' 
        });
    }
    
    placeTrade(action) {
        if (!this.tradingEnabled) {
            showAlert('Trading is disabled. Enable AI mode to trade.', 'warning');
            return;
        }
        
        const trade = {
            action: action,
            symbol: 'NQ',
            position_size: this.positionSize,
            risk_level: this.riskLevel,
            current_price: this.currentPrice,
            timestamp: new Date().toISOString()
        };
        
        // Validate trade
        if (!this.validateTrade(trade)) {
            return;
        }
        
        // Send trade to server
        fetch('/api/place_trade', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(trade)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showAlert(`${action.toUpperCase()} order placed successfully`, 'success');
                this.updateTradingInterface(data);
            } else {
                showAlert(`Failed to place ${action} order: ${data.error}`, 'danger');
            }
        })
        .catch(error => {
            console.error('Error placing trade:', error);
            showAlert('Error placing trade', 'danger');
        });
    }
    
    /**
     * @param {Trade} trade
     * @returns {boolean}
     */
    validateTrade(trade) {
        // Check if market is open
        if (!this.isMarketOpen()) {
            showAlert('Market is closed. Cannot place trades.', 'warning');
            return false;
        }
        
        // Check position size
        if (trade.position_size <= 0 || trade.position_size > 10) {
            showAlert('Invalid position size. Must be between 1 and 10.', 'warning');
            return false;
        }
        
        // Check risk limits
        if (!this.checkRiskLimits(trade)) {
            showAlert('Trade exceeds risk limits.', 'danger');
            return false;
        }
        
        return true;
    }
    
    checkRiskLimits(trade) {
        // Implement risk checking logic
        const maxRisk = this.calculateMaxRisk();
        const tradeRisk = this.calculateTradeRisk(trade);
        
        return tradeRisk <= maxRisk;
    }
    
    calculateMaxRisk() {
        const riskLevels = {
            conservative: 0.01, // 1%
            moderate: 0.02,     // 2%
            aggressive: 0.05    // 5%
        };
        
        return riskLevels[this.riskLevel] || 0.02;
    }
    
    calculateTradeRisk(trade) {
        // Simplified risk calculation
        const accountValue = 100000; // Assume $100k account
        const contractValue = trade.current_price * 20; // NQ contract size
        const riskPerContract = contractValue * 0.01; // 1% risk per contract
        
        return (riskPerContract * trade.position_size) / accountValue;
    }
    
    isMarketOpen() {
        const now = new Date();
        const currentHour = now.getHours();
        
        // NQ futures trade nearly 24/7, simplified check
        return currentHour >= 6 && currentHour <= 17; // 6 AM to 5 PM
    }
    
    toggleAI() {
        this.tradingEnabled = !this.tradingEnabled;
        
        const toggleBtn = document.getElementById('toggle-ai');
        const aiRecommendations = document.getElementById('ai-recommendations');
        
        if (this.tradingEnabled) {
            toggleBtn.innerHTML = '<i class="fas fa-power-off me-1"></i>AI Active';
            toggleBtn.className = 'btn btn-sm btn-success w-100';
            this.startAIMode();
            
            // Update AI recommendations display
            if (aiRecommendations) {
                aiRecommendations.innerHTML = `
                    <div class="text-success">AI System Active</div>
                    <div class="text-info">Analyzing market patterns...</div>
                    <div class="text-warning">Generating trading signals...</div>
                `;
            }
        } else {
            toggleBtn.innerHTML = '<i class="fas fa-power-off me-1"></i>AI Inactive';
            toggleBtn.className = 'btn btn-sm btn-outline-primary w-100';
            this.stopAIMode();
            
            if (aiRecommendations) {
                aiRecommendations.innerHTML = `
                    <div class="text-muted">AI System Offline</div>
                    <div class="text-muted">Enable AI to receive recommendations</div>
                `;
            }
        }
        
        showNotification(`AI mode ${this.tradingEnabled ? 'enabled' : 'disabled'}`, 'info');
    }
    
    startAIMode() {
        // Start AI analysis
        this.socket.emit('start_ai_analysis', {
            session_id: this.currentSession,
            risk_level: this.riskLevel,
            position_size: this.positionSize
        });
        
        // Start real-time monitoring
        this.aiUpdateInterval = setInterval(() => {
            this.requestAIUpdate();
        }, 1000);
    }
    
    stopAIMode() {
        // Stop AI analysis
        this.socket.emit('stop_ai_analysis');
        
        // Stop real-time monitoring
        if (this.aiUpdateInterval) {
            clearInterval(this.aiUpdateInterval);
            this.aiUpdateInterval = null;
        }
    }
    
    requestAIUpdate() {
        if (this.currentSession) {
            this.socket.emit('request_ai_update', {
                session_id: this.currentSession,
                current_price: this.currentPrice
            });
        }
    }
    
    /**
     * @param {TrainingUpdate} data
     */
    handleTrainingUpdate(data) {
        if (data.session_id === this.currentSession) {
            // Update session progress
            this.updateSessionProgress(data);
            
            // Update performance metrics if metrics are provided
            if (data.metrics) {
                this.updatePerformanceMetrics(data.metrics);
            }
            
            // Update neural network visualization if network state is provided
            if (window.neuralNetworkViz && data.network_state) {
                window.neuralNetworkViz.updateWeights(data.network_state);
            }
        }
    }
    
    handleMarketData(data) {
        this.currentPrice = data.close;
        this.marketData.push(data);
        
        // Keep only last 1000 data points
        if (this.marketData.length > 1000) {
            this.marketData.shift();
        }
        
        // Update price display
        this.updatePriceDisplay(data);
        
        // Update charts
        this.updatePriceChart();
        
        // Update 3D visualization
        if (window.portfolio3D) {
            window.portfolio3D.updatePrice(data.close);
        }
    }
    
    handleTradeExecution(data) {
        showNotification(`Trade executed: ${data.action} ${data.quantity} contracts at ${data.price}`, 'success');
        
        // Update trading interface
        this.updateTradingInterface(data);
        
        // Update performance metrics
        this.updatePerformanceMetrics(data.metrics);
    }
    
    handleAIRecommendation(data) {
        this.aiRecommendations.push(data);
        
        // Keep only last 10 recommendations
        if (this.aiRecommendations.length > 10) {
            this.aiRecommendations.shift();
        }
        
        // Update AI recommendations display
        this.updateAIRecommendationsDisplay();
        
        // Show notification for strong recommendations
        if (data.confidence > 0.8) {
            showNotification(`AI Recommendation: ${data.action.toUpperCase()} (${(data.confidence * 100).toFixed(1)}% confidence)`, 'info');
        }
    }
    
    handleRiskAlert(data) {
        showAlert(`Risk Alert: ${data.message}`, 'warning');
        
        // Update risk metrics display
        this.updateRiskDisplay(data);
        
        // Auto-disable trading if risk is too high
        if (data.severity === 'high') {
            this.tradingEnabled = false;
            this.toggleAI();
        }
    }
    
    /**
     * @param {SessionData} sessionData
     */
    updateSessionDisplay(sessionData) {
        // Update session info
        const sessionName = document.querySelector('.session-name');
        if (sessionName) sessionName.textContent = sessionData.session.name;
        
        const sessionAlgorithm = document.querySelector('.session-algorithm');
        if (sessionAlgorithm) sessionAlgorithm.textContent = sessionData.session.algorithm_type;
        
        const sessionEpisode = document.querySelector('.session-episode');
        if (sessionEpisode) sessionEpisode.textContent = 
            `${sessionData.session.current_episode}/${sessionData.session.total_episodes}`;
        
        // Update progress bar
        const progressBar = document.querySelector('.session-progress');
        if (progressBar) {
            const progress = (sessionData.session.current_episode / sessionData.session.total_episodes) * 100;
            progressBar.style.width = progress + '%';
        }
    }
    
    updateSessionProgress(data) {
        // Update episode counter
        const episodeElement = document.querySelector('.current-episode');
        if (episodeElement) episodeElement.textContent = data.episode;
        
        // Update progress bar
        const progressBar = document.querySelector('.episode-progress');
        if (progressBar) {
            const progress = (data.episode / data.total_episodes) * 100;
            progressBar.style.width = progress + '%';
        }
        
        // Update reward display
        const rewardElement = document.querySelector('.current-reward');
        if (rewardElement) rewardElement.textContent = data.reward.toFixed(2);
    }
    
    updatePerformanceMetrics(metrics) {
        if (!metrics) return;
        
        // Update display elements
        const sharpeElement = document.getElementById('sharpe-ratio');
        if (sharpeElement) sharpeElement.textContent = metrics.sharpe_ratio?.toFixed(2) || '0.00';
        
        const totalReturnElement = document.getElementById('total-return');
        if (totalReturnElement) totalReturnElement.textContent = ((metrics.total_return * 100)?.toFixed(2) || '0.00') + '%';
        
        const maxDrawdownElement = document.getElementById('max-drawdown');
        if (maxDrawdownElement) maxDrawdownElement.textContent = ((metrics.max_drawdown * 100)?.toFixed(2) || '0.00') + '%';
        
        const winRateElement = document.getElementById('win-rate');
        if (winRateElement) winRateElement.textContent = ((metrics.win_rate * 100)?.toFixed(2) || '0.00') + '%';
        
        const totalTradesElement = document.getElementById('total-trades');
        if (totalTradesElement) totalTradesElement.textContent = metrics.total_trades || '0';
        
        const profitableTradesElement = document.getElementById('profitable-trades');
        if (profitableTradesElement) profitableTradesElement.textContent = metrics.profitable_trades || '0';
        
        // Update performance rings
        this.updatePerformanceRings(metrics);
    }
    
    updatePerformanceRings(metrics) {
        const sharpeRing = document.querySelector('.performance-ring #sharpe-ratio');
        const returnRing = document.querySelector('.performance-ring #total-return');
        
        if (sharpeRing && metrics.sharpe_ratio) {
            const sharpeAngle = Math.min(Math.max(metrics.sharpe_ratio * 180, -180), 180);
            sharpeRing.style.background = `conic-gradient(#00ff88 ${sharpeAngle}deg, #333 ${sharpeAngle}deg)`;
        }
        
        if (returnRing && metrics.total_return) {
            const returnAngle = Math.min(Math.max(metrics.total_return * 1800, -180), 180);
            returnRing.style.background = `conic-gradient(#00ff88 ${returnAngle}deg, #333 ${returnAngle}deg)`;
        }
    }
    
    updateRiskMetrics(metrics) {
        if (!metrics) return;
        
        // Update drawdown bar
        const drawdownBar = document.getElementById('drawdown-bar');
        const drawdownValue = document.getElementById('drawdown-value');
        
        if (drawdownBar && metrics.max_drawdown) {
            const drawdownPercent = Math.abs(metrics.max_drawdown * 100);
            drawdownBar.style.width = Math.min(drawdownPercent, 100) + '%';
            drawdownValue.textContent = drawdownPercent.toFixed(2) + '%';
        }
        
        // Update VaR bar
        const varBar = document.getElementById('var-bar');
        const varValue = document.getElementById('var-value');
        
        if (varBar && metrics.var_95) {
            const varPercent = (metrics.var_95 / 10000) * 100; // Assuming $10k max
            varBar.style.width = Math.min(varPercent, 100) + '%';
            varValue.textContent = '$' + metrics.var_95.toFixed(2);
        }
    }
    
    updatePriceDisplay(data) {
        const priceElements = document.querySelectorAll('.current-price');
        priceElements.forEach(element => {
            element.textContent = data.close.toFixed(2);
            
            // Add price change indicator
            const changeClass = data.change > 0 ? 'price-up' : 'price-down';
            element.className = `current-price ${changeClass}`;
        });
    }
    
    updatePriceChart() {
        if (window.tradingCharts) {
            window.tradingCharts.updatePriceChart(this.marketData);
        }
    }
    
    updateTradingInterface(data) {
        // Update position display
        const positionElement = document.querySelector('.current-position');
        if (positionElement) positionElement.textContent = data.position || '0';
        
        // Update P&L display
        const pnlElement = document.querySelector('.unrealized-pnl');
        if (pnlElement) pnlElement.textContent = (data.unrealized_pnl || 0).toFixed(2);
        
        // Update trade history
        this.addTradeToHistory(data);
    }
    
    addTradeToHistory(trade) {
        const historyContainer = document.querySelector('.trade-history');
        if (!historyContainer) return;
        
        const tradeElement = document.createElement('div');
        tradeElement.className = 'trade-item';
        tradeElement.innerHTML = `
            <div class="trade-time">${new Date(trade.timestamp).toLocaleTimeString()}</div>
            <div class="trade-action ${trade.action}">${trade.action.toUpperCase()}</div>
            <div class="trade-price">${trade.price.toFixed(2)}</div>
            <div class="trade-pnl ${trade.pnl > 0 ? 'profit' : 'loss'}">
                ${trade.pnl > 0 ? '+' : ''}${trade.pnl.toFixed(2)}
            </div>
        `;
        
        historyContainer.insertBefore(tradeElement, historyContainer.firstChild);
        
        // Keep only last 20 trades
        const trades = historyContainer.querySelectorAll('.trade-item');
        if (trades.length > 20) {
            trades[trades.length - 1].remove();
        }
    }
    
    updateAIRecommendationsDisplay() {
        const container = document.getElementById('ai-recommendations');
        if (!container) return;
        
        const recommendations = this.aiRecommendations.slice(-5); // Last 5 recommendations
        
        container.innerHTML = recommendations.map(rec => `
            <div class="ai-recommendation ${rec.action}">
                <span class="recommendation-time">${new Date(rec.timestamp).toLocaleTimeString()}</span>
                <span class="recommendation-action">${rec.action.toUpperCase()}</span>
                <span class="recommendation-confidence">${(rec.confidence * 100).toFixed(1)}%</span>
            </div>
        `).join('');
    }
    
    updatePositionSizeDisplay() {
        const display = document.getElementById('position-size-value');
        if (display) {
            display.textContent = this.positionSize + ' Contract' + (this.positionSize > 1 ? 's' : '');
        }
    }
    
    updateRiskParameters() {
        // Update risk-related parameters based on selected level
        const riskParams = {
            conservative: {
                maxDrawdown: 0.05,
                maxPosition: 2,
                stopLoss: 0.01
            },
            moderate: {
                maxDrawdown: 0.10,
                maxPosition: 5,
                stopLoss: 0.02
            },
            aggressive: {
                maxDrawdown: 0.20,
                maxPosition: 10,
                stopLoss: 0.03
            }
        };
        
        this.riskParams = riskParams[this.riskLevel];
        
        // Update position size limits
        const positionSlider = document.getElementById('position-size');
        if (positionSlider) {
            const slider = /** @type {HTMLInputElement} */ (positionSlider);
            slider.max = this.riskParams.maxPosition.toString();
            if (this.positionSize > this.riskParams.maxPosition) {
                this.positionSize = this.riskParams.maxPosition;
                slider.value = this.positionSize.toString();
                this.updatePositionSizeDisplay();
            }
        }
    }
    
    /**
     * @param {RiskData} riskData
     */
    updateRiskDisplay(riskData) {
        // Update risk indicators
        const riskIndicators = document.querySelectorAll('.risk-indicator');
        riskIndicators.forEach(indicator => {
            indicator.className = `risk-indicator ${riskData.level}`;
        });
        
        // Update risk metrics
        if (riskData.drawdown) {
            const drawdownBar = document.getElementById('drawdown-bar');
            const drawdownValue = document.getElementById('drawdown-value');
            
            if (drawdownBar) {
                const drawdownPercent = Math.abs(riskData.drawdown * 100);
                drawdownBar.style.width = Math.min(drawdownPercent, 100) + '%';
                drawdownValue.textContent = drawdownPercent.toFixed(2) + '%';
            }
        }
    }
    
    startRealTimeUpdates() {
        // Request real-time updates every second
        this.updateInterval = setInterval(() => {
            if (this.currentSession) {
                this.socket.emit('request_real_time_update', {
                    session_id: this.currentSession
                });
            }
        }, 1000);
    }
    
    initializeAIAssistant() {
        // Initialize AI assistant interface
        const assistant = document.getElementById('ai-assistant');
        if (assistant) {
            // Add drag functionality
            this.makeDraggable(assistant);
            
            // Add minimize/maximize functionality
            const minimizeBtn = document.createElement('button');
            minimizeBtn.className = 'btn btn-sm btn-outline-light';
            minimizeBtn.innerHTML = '<i class="fas fa-minus"></i>';
            minimizeBtn.style.position = 'absolute';
            minimizeBtn.style.top = '10px';
            minimizeBtn.style.right = '10px';
            
            minimizeBtn.addEventListener('click', () => {
                assistant.classList.toggle('minimized');
            });
            
            assistant.appendChild(minimizeBtn);
        }
    }
    
    makeDraggable(element) {
        let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
        
        element.style.cursor = 'move';
        element.addEventListener('mousedown', dragMouseDown);
        
        function dragMouseDown(e) {
            e.preventDefault();
            pos3 = e.clientX;
            pos4 = e.clientY;
            document.addEventListener('mouseup', closeDragElement);
            document.addEventListener('mousemove', elementDrag);
        }
        
        function elementDrag(e) {
            e.preventDefault();
            pos1 = pos3 - e.clientX;
            pos2 = pos4 - e.clientY;
            pos3 = e.clientX;
            pos4 = e.clientY;
            
            element.style.top = (element.offsetTop - pos2) + "px";
            element.style.left = (element.offsetLeft - pos1) + "px";
        }
        
        function closeDragElement() {
            document.removeEventListener('mouseup', closeDragElement);
            document.removeEventListener('mousemove', elementDrag);
        }
    }
    
    loadUserPreferences() {
        const preferences = localStorage.getItem('trading_preferences');
        if (preferences) {
            try {
                const prefs = JSON.parse(preferences);
                this.riskLevel = prefs.riskLevel || 'moderate';
                this.positionSize = prefs.positionSize || 1;
                this.tradingEnabled = prefs.tradingEnabled || false;
                
                // Apply preferences to UI
                const riskLevelElement = document.getElementById('risk-level');
                if (riskLevelElement) riskLevelElement.value = this.riskLevel;
                
                const positionSizeElement = document.getElementById('position-size');
                if (positionSizeElement) positionSizeElement.value = this.positionSize;
                
                this.updateRiskParameters();
                this.updatePositionSizeDisplay();
                
            } catch (error) {
                console.error('Error loading preferences:', error);
            }
        }
    }
    
    saveUserPreferences() {
        const preferences = {
            riskLevel: this.riskLevel,
            positionSize: this.positionSize,
            tradingEnabled: this.tradingEnabled
        };
        
        localStorage.setItem('trading_preferences', JSON.stringify(preferences));
    }
    
    // Cleanup when leaving page
    cleanup() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        if (this.aiUpdateInterval) {
            clearInterval(this.aiUpdateInterval);
        }
        
        this.socket.disconnect();
        this.saveUserPreferences();
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', function() {
    window.tradingDashboard = new TradingDashboard();
    window.tradingDashboard.init();
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (window.tradingDashboard) {
        window.tradingDashboard.cleanup();
    }
});

console.log('🎯 Trading Dashboard module loaded');
