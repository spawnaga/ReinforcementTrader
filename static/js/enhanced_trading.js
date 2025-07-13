// Enhanced Trading Dashboard JavaScript
class EnhancedTradingDashboard {
    constructor() {
        this.socket = io();
        this.isTraining = false;
        this.currentEpisode = 0;
        this.totalEpisodes = 1000;
        this.selectedIndicators = ['sma', 'ema', 'rsi', 'macd'];
        this.neuralNetwork = null;
        this.trainingChart = null;
        this.dataStreamInterval = null;
        
        this.initializeComponents();
        this.setupEventListeners();
        this.setupWebSocketHandlers();
        this.startDataStream();
        this.initializeNeuralNetworkViz();
    }
    
    initializeComponents() {
        // Initialize training chart
        const ctx = document.getElementById('trainingChart')?.getContext('2d');
        if (ctx) {
            this.trainingChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Reward',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        tension: 0.1
                    }, {
                        label: 'Loss',
                        data: [],
                        borderColor: '#ff4444',
                        backgroundColor: 'rgba(255, 68, 68, 0.1)',
                        tension: 0.1,
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#e0e0e0'
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                color: '#333'
                            },
                            ticks: {
                                color: '#e0e0e0'
                            }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            grid: {
                                color: '#333'
                            },
                            ticks: {
                                color: '#e0e0e0'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            grid: {
                                drawOnChartArea: false,
                                color: '#333'
                            },
                            ticks: {
                                color: '#e0e0e0'
                            }
                        }
                    }
                }
            });
        }
    }
    
    setupEventListeners() {
        // Training controls
        document.getElementById('startTraining')?.addEventListener('click', () => this.startTraining());
        document.getElementById('pauseTraining')?.addEventListener('click', () => this.pauseTraining());
        document.getElementById('stopTraining')?.addEventListener('click', () => this.stopTraining());
        
        // Neural network controls
        document.getElementById('hiddenLayers')?.addEventListener('input', (e) => {
            document.getElementById('hiddenLayersValue').textContent = e.target.value;
            this.updateNeuralNetwork();
        });
        
        document.getElementById('neuronsPerLayer')?.addEventListener('input', (e) => {
            document.getElementById('neuronsValue').textContent = e.target.value;
            this.updateNeuralNetwork();
        });
        
        document.getElementById('learningRate')?.addEventListener('input', (e) => {
            document.getElementById('learningRateValue').textContent = e.target.value;
        });
        
        document.getElementById('dropoutRate')?.addEventListener('input', (e) => {
            document.getElementById('dropoutValue').textContent = e.target.value;
        });
        
        document.getElementById('batchSize')?.addEventListener('input', (e) => {
            document.getElementById('batchSizeValue').textContent = e.target.value;
        });
        
        // Indicator selection
        document.querySelectorAll('.indicator-item').forEach(item => {
            item.addEventListener('click', (e) => {
                if (e.target.tagName !== 'INPUT') {
                    const checkbox = item.querySelector('input[type="checkbox"]');
                    checkbox.checked = !checkbox.checked;
                }
                
                item.classList.toggle('selected');
                this.updateSelectedIndicators();
            });
        });
        
        // Optimization buttons
        document.querySelectorAll('.optimization-controls button').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const optimizationType = e.currentTarget.textContent.trim();
                this.runOptimization(optimizationType);
            });
        });
        
        // Timeframe selection
        document.getElementById('timeframeSelect')?.addEventListener('change', (e) => {
            this.updateTimeframe(e.target.value);
        });
    }
    
    setupWebSocketHandlers() {
        this.socket.on('connect', () => {
            console.log('Connected to trading server');
            this.updateStatus('ONLINE', true);
            
            // Join session room if we have a session
            if (this.sessionId) {
                this.socket.emit('join_session', {session_id: this.sessionId});
            }
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from trading server');
            this.updateStatus('OFFLINE', false);
        });
        
        this.socket.on('training_update', (data) => {
            this.handleTrainingUpdate(data);
        });
        
        this.socket.on('trade_update', (data) => {
            this.handleTradeUpdate(data);
        });
        
        this.socket.on('performance_metrics', (data) => {
            this.updatePerformanceMetrics(data);
        });
        
        this.socket.on('neural_network_update', (data) => {
            this.updateNeuralNetworkVisualization(data);
        });
    }
    
    async startTraining() {
        // Check if there's an existing session
        const existingSessions = await this.checkExistingSessions();
        
        if (existingSessions.length > 0) {
            // Ask user what to do with existing sessions
            const userChoice = await this.showSessionOptions(existingSessions);
            
            if (userChoice === 'continue') {
                // Continue with the existing session
                this.sessionId = existingSessions[0].id;
                await this.resumeTraining();
                return;
            } else if (userChoice === 'reset') {
                // Reset the existing session
                await this.resetSession(existingSessions[0].id);
                this.sessionId = existingSessions[0].id;
            } else {
                // Create new session (default behavior)
                await this.createNewSession();
            }
        } else {
            // No existing sessions, create new one
            await this.createNewSession();
        }
        
        // Start the training
        if (this.sessionId) {
            const startResponse = await fetch(`/api/sessions/${this.sessionId}/start`, {
                method: 'POST'
            });
            
            if (startResponse.ok) {
                this.isTraining = true;
                this.updateTrainingControls();
                this.resetMetricsDisplay();
                console.log('Training started successfully');
                
                // Join session room for WebSocket updates
                if (this.socket.connected) {
                    this.socket.emit('join_session', {session_id: this.sessionId});
                }
                
                // Load existing trades for this session
                this.loadSessionTrades();
            }
        }
    }
    
    async checkExistingSessions() {
        try {
            const response = await fetch('/api/sessions');
            if (response.ok) {
                const sessions = await response.json();
                return sessions.filter(s => s.status === 'active' || s.status === 'paused');
            }
        } catch (error) {
            console.error('Error checking sessions:', error);
        }
        return [];
    }
    
    async createNewSession() {
        const params = {
            session_name: `Training Session ${new Date().toLocaleString()}`,
            algorithm_type: 'ANE_PPO',
            parameters: {
                hidden_layers: parseInt(document.getElementById('hiddenLayers').value),
                neurons_per_layer: parseInt(document.getElementById('neuronsPerLayer').value),
                learning_rate: parseFloat(document.getElementById('learningRate').value),
                dropout_rate: parseFloat(document.getElementById('dropoutRate').value),
                batch_size: parseInt(document.getElementById('batchSize').value),
                indicators: this.selectedIndicators,
                timeframe: parseInt(document.getElementById('timeframeSelect').value)
            }
        };
        
        try {
            const response = await fetch('/api/sessions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(params)
            });
            
            if (response.ok) {
                const data = await response.json();
                this.sessionId = data.session_id;
            }
        } catch (error) {
            console.error('Error creating session:', error);
        }
    }
    
    async resetSession(sessionId) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}/reset`, {
                method: 'POST'
            });
            
            if (response.ok) {
                console.log('Session reset successfully');
                this.currentEpisode = 0;
                this.updateTrainingProgress(0);
                this.resetMetricsDisplay();
                this.clearTradeList();
            }
        } catch (error) {
            console.error('Error resetting session:', error);
        }
    }
    
    async resumeTraining() {
        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/start`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.isTraining = true;
                this.updateTrainingControls();
                console.log('Training resumed');
            }
        } catch (error) {
            console.error('Error resuming training:', error);
        }
    }
    
    async showSessionOptions(sessions) {
        // Create a simple modal to ask user
        const modal = document.createElement('div');
        modal.innerHTML = `
            <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); display: flex; align-items: center; justify-content: center; z-index: 9999;">
                <div style="background: #1a1a2e; padding: 30px; border-radius: 10px; max-width: 500px;">
                    <h3 style="color: #00ff88; margin-bottom: 20px;">Active Session Found</h3>
                    <p style="color: #e0e0e0; margin-bottom: 20px;">
                        Session: ${sessions[0].name}<br>
                        Progress: ${sessions[0].current_episode}/${sessions[0].total_episodes} episodes<br>
                        Total Profit: $${sessions[0].total_profit.toFixed(2)}<br>
                        Total Trades: ${sessions[0].total_trades}
                    </p>
                    <p style="color: #e0e0e0; margin-bottom: 20px;">What would you like to do?</p>
                    <div style="display: flex; gap: 10px;">
                        <button id="continueBtn" style="padding: 10px 20px; background: #00ff88; color: #000; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">Continue Training</button>
                        <button id="resetBtn" style="padding: 10px 20px; background: #ff8800; color: #fff; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">Reset & Start Fresh</button>
                        <button id="newBtn" style="padding: 10px 20px; background: #0088ff; color: #fff; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">Create New Session</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        return new Promise(resolve => {
            document.getElementById('continueBtn').onclick = () => {
                document.body.removeChild(modal);
                resolve('continue');
            };
            document.getElementById('resetBtn').onclick = () => {
                document.body.removeChild(modal);
                resolve('reset');
            };
            document.getElementById('newBtn').onclick = () => {
                document.body.removeChild(modal);
                resolve('new');
            };
        });
    }
    
    resetMetricsDisplay() {
        document.getElementById('winRate').textContent = '0%';
        document.getElementById('sharpeRatio').textContent = '0.00';
        document.getElementById('totalProfit').textContent = '$0';
        document.getElementById('maxDrawdown').textContent = '0%';
        document.getElementById('currentEpisode').textContent = '0';
        document.getElementById('totalTrades').textContent = '0';
    }
    
    clearTradeList() {
        document.getElementById('tradeList').innerHTML = '';
    }
    
    async loadSessionTrades() {
        if (!this.sessionId) return;
        
        try {
            const response = await fetch(`/api/recent_trades?session_id=${this.sessionId}&limit=20`);
            if (response.ok) {
                const trades = await response.json();
                
                // Clear existing trades
                this.clearTradeList();
                
                // Add trades in reverse order (oldest first) so newest appear at top
                trades.reverse().forEach(trade => {
                    this.handleTradeUpdate(trade);
                });
            }
        } catch (error) {
            console.error('Error loading session trades:', error);
        }
    }
    
    async pauseTraining() {
        if (!this.sessionId) return;
        
        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/pause`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.isTraining = false;
                this.updateTrainingControls();
            }
        } catch (error) {
            console.error('Error pausing training:', error);
        }
    }
    
    async stopTraining() {
        if (!this.sessionId) return;
        
        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/stop`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.isTraining = false;
                this.currentEpisode = 0;
                this.updateTrainingControls();
                this.updateTrainingProgress(0);
            }
        } catch (error) {
            console.error('Error stopping training:', error);
        }
    }
    
    updateTrainingControls() {
        document.getElementById('startTraining').disabled = this.isTraining;
        document.getElementById('pauseTraining').disabled = !this.isTraining;
        document.getElementById('stopTraining').disabled = !this.isTraining;
        
        document.getElementById('trainingMode').textContent = this.isTraining ? 'LIVE' : 'STOPPED';
    }
    
    handleTrainingUpdate(data) {
        this.currentEpisode = data.episode;
        this.updateTrainingProgress(data.episode / this.totalEpisodes * 100);
        
        // Update chart
        if (this.trainingChart) {
            this.trainingChart.data.labels.push(data.episode);
            this.trainingChart.data.datasets[0].data.push(data.reward);
            this.trainingChart.data.datasets[1].data.push(data.loss);
            
            // Keep only last 100 points
            if (this.trainingChart.data.labels.length > 100) {
                this.trainingChart.data.labels.shift();
                this.trainingChart.data.datasets.forEach(dataset => dataset.data.shift());
            }
            
            this.trainingChart.update();
        }
        
        // Update metrics
        if (data.metrics) {
            // Safely update win rate
            if (data.metrics.win_rate !== undefined && data.metrics.win_rate !== null) {
                document.getElementById('winRate').textContent = `${(data.metrics.win_rate * 100).toFixed(1)}%`;
            }
            
            // Safely update sharpe ratio
            if (data.metrics.sharpe_ratio !== undefined && data.metrics.sharpe_ratio !== null) {
                document.getElementById('sharpeRatio').textContent = data.metrics.sharpe_ratio.toFixed(2);
            }
            
            // Safely update total profit
            if (data.metrics.total_profit !== undefined && data.metrics.total_profit !== null) {
                document.getElementById('totalProfit').textContent = `$${data.metrics.total_profit.toFixed(2)}`;
            }
            
            // Safely update max drawdown
            if (data.metrics.max_drawdown !== undefined && data.metrics.max_drawdown !== null) {
                document.getElementById('maxDrawdown').textContent = `${(data.metrics.max_drawdown * 100).toFixed(1)}%`;
            }
            
            // Safely update total trades
            if (data.metrics.total_trades !== undefined && data.metrics.total_trades !== null) {
                document.getElementById('totalTrades').textContent = data.metrics.total_trades.toString();
            }
        }
    }
    
    handleTradeUpdate(trade) {
        // Only show trades for the current session
        if (trade.session_id !== this.sessionId) {
            return;
        }
        
        const tradeList = document.getElementById('tradeList');
        const tradeItem = document.createElement('div');
        tradeItem.className = `trade-item ${trade.profit_loss >= 0 ? 'profit' : 'loss'}`;
        tradeItem.innerHTML = `
            <span>${trade.position_type.toUpperCase()} @ $${trade.entry_price.toFixed(2)}</span>
            <span class="${trade.profit_loss >= 0 ? 'text-success' : 'text-danger'}">
                ${trade.profit_loss >= 0 ? '+' : ''}$${trade.profit_loss.toFixed(2)}
            </span>
        `;
        
        // Prepend to list (newest first)
        tradeList.insertBefore(tradeItem, tradeList.firstChild);
        
        // Keep only last 20 trades
        while (tradeList.children.length > 20) {
            tradeList.removeChild(tradeList.lastChild);
        }
    }
    
    updateTrainingProgress(percentage) {
        const progressBar = document.getElementById('trainingProgress');
        progressBar.style.width = `${percentage}%`;
        progressBar.textContent = `${Math.round(percentage)}%`;
        
        document.getElementById('currentEpisode').textContent = this.currentEpisode;
        document.getElementById('totalEpisodes').textContent = this.totalEpisodes;
    }
    
    updateSelectedIndicators() {
        this.selectedIndicators = [];
        document.querySelectorAll('.indicator-item.selected').forEach(item => {
            this.selectedIndicators.push(item.dataset.indicator);
        });
        console.log('Selected indicators:', this.selectedIndicators);
    }
    
    updateNeuralNetwork() {
        const layers = parseInt(document.getElementById('hiddenLayers').value);
        const neurons = parseInt(document.getElementById('neuronsPerLayer').value);
        
        // Redraw neural network visualization
        if (this.neuralNetwork) {
            this.neuralNetwork.updateArchitecture(layers, neurons);
        }
    }
    
    startDataStream() {
        const dataStream = document.getElementById('dataStream');
        
        this.dataStreamInterval = setInterval(() => {
            const timestamp = new Date().toISOString().substr(11, 8);
            const price = (15000 + Math.random() * 100).toFixed(2);
            const volume = Math.floor(Math.random() * 1000);
            
            const line = `[${timestamp}] NQ: $${price} Vol: ${volume}\n`;
            dataStream.textContent = line + dataStream.textContent;
            
            // Keep only last 10 lines
            const lines = dataStream.textContent.split('\n');
            if (lines.length > 10) {
                dataStream.textContent = lines.slice(0, 10).join('\n');
            }
        }, 1000);
    }
    
    updateStatus(status, isActive) {
        document.getElementById('aiStatus').textContent = status;
        const indicator = document.querySelector('.status-indicator');
        if (isActive) {
            indicator.classList.add('active');
            indicator.classList.remove('inactive');
        } else {
            indicator.classList.add('inactive');
            indicator.classList.remove('active');
        }
    }
    
    async runOptimization(type) {
        console.log(`Running ${type} optimization...`);
        // This would trigger the optimization process
        try {
            const response = await fetch('/api/optimize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    type: type,
                    parameters: {
                        indicators: this.selectedIndicators,
                        timeframe: document.getElementById('timeframeSelect').value
                    }
                })
            });
            
            if (response.ok) {
                console.log('Optimization started');
            }
        } catch (error) {
            console.error('Error starting optimization:', error);
        }
    }
    
    updateTimeframe(timeframe) {
        console.log(`Timeframe changed to ${timeframe} minutes`);
        // This would update the data fetching interval
    }
    
    updatePerformanceMetrics(data) {
        // Update GPU status if available
        if (data.gpu_usage !== undefined) {
            document.getElementById('gpuStatus').textContent = 
                data.gpu_usage > 0 ? `4x RTX 3090 (${data.gpu_usage}%)` : 'CPU Mode';
        }
    }
    
    initializeNeuralNetworkViz() {
        const canvas = document.getElementById('neural-network-viz');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        
        // Simple neural network visualization
        this.neuralNetwork = {
            ctx: ctx,
            width: canvas.width,
            height: canvas.height,
            
            updateArchitecture: function(layers, neurons) {
                this.draw(layers, neurons);
            },
            
            draw: function(layers = 3, neuronsPerLayer = 128) {
                ctx.clearRect(0, 0, this.width, this.height);
                
                const layerWidth = this.width / (layers + 2);
                const maxNeuronsToShow = 10; // Show max 10 neurons per layer for visualization
                
                // Draw connections
                ctx.strokeStyle = 'rgba(0, 255, 136, 0.1)';
                ctx.lineWidth = 1;
                
                for (let i = 0; i < layers + 1; i++) {
                    const x1 = layerWidth * (i + 0.5);
                    const x2 = layerWidth * (i + 1.5);
                    const neurons1 = Math.min(neuronsPerLayer, maxNeuronsToShow);
                    const neurons2 = Math.min(neuronsPerLayer, maxNeuronsToShow);
                    
                    for (let j = 0; j < neurons1; j++) {
                        const y1 = (this.height / (neurons1 + 1)) * (j + 1);
                        for (let k = 0; k < neurons2; k++) {
                            const y2 = (this.height / (neurons2 + 1)) * (k + 1);
                            ctx.beginPath();
                            ctx.moveTo(x1, y1);
                            ctx.lineTo(x2, y2);
                            ctx.stroke();
                        }
                    }
                }
                
                // Draw neurons
                ctx.fillStyle = '#00ff88';
                for (let i = 0; i <= layers + 1; i++) {
                    const x = layerWidth * (i + 0.5);
                    const neuronsToShow = Math.min(neuronsPerLayer, maxNeuronsToShow);
                    
                    for (let j = 0; j < neuronsToShow; j++) {
                        const y = (this.height / (neuronsToShow + 1)) * (j + 1);
                        ctx.beginPath();
                        ctx.arc(x, y, 5, 0, Math.PI * 2);
                        ctx.fill();
                    }
                }
                
                // Labels
                ctx.fillStyle = '#e0e0e0';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('Input', layerWidth * 0.5, this.height - 10);
                ctx.fillText('Output', layerWidth * (layers + 1.5), this.height - 10);
            }
        };
        
        // Initial draw
        this.neuralNetwork.draw();
    }
    
    updateNeuralNetworkVisualization(data) {
        // This would update the neural network visualization with real-time activation data
        console.log('Neural network update:', data);
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.enhancedDashboard = new EnhancedTradingDashboard();
});