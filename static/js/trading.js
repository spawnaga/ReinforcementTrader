// Trading Dashboard JavaScript
// Visualization classes are loaded from separate files:
// - Portfolio3DVisualization from 3d_visualization.js
// - NeuralNetworkVisualization from neural_network_viz.js  
// - TradingCharts from charts.js

class TradingDashboard {
    constructor() {
        this.sessionId = null;
        this.isTraining = false;
        this.chart = null;
        this.metricsChart = null;
        this.priceChart = null;
        this.socket = window.socket;
        
        this.initializeEventListeners();
        this.initializeCharts();
        this.updateSessionsList();
        
        // Subscribe to real-time updates
        this.subscribeToUpdates();
    }
    
    initializeEventListeners() {
        // Session management
        document.getElementById('startNewSession')?.addEventListener('click', () => this.showNewSessionModal());
        document.getElementById('createSession')?.addEventListener('click', () => this.createNewSession());
        document.getElementById('stopTraining')?.addEventListener('click', () => this.stopTraining());
        
        // Algorithm selection
        document.getElementById('algorithmType')?.addEventListener('change', (e) => this.updateAlgorithmParameters(e.target.value));
        
        // Modal close buttons
        document.querySelectorAll('.close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.target.closest('.modal').style.display = 'none';
            });
        });
        
        // Click outside modal to close
        window.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                e.target.style.display = 'none';
            }
        });
    }
    
    initializeCharts() {
        // Performance metrics chart
        const metricsCtx = document.getElementById('metricsChart')?.getContext('2d');
        if (metricsCtx) {
            this.metricsChart = new Chart(metricsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Reward',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        tension: 0.1
                    }, {
                        label: 'Loss',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        tension: 0.1,
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Training Metrics',
                            color: '#e0e0e0'
                        },
                        legend: {
                            labels: {
                                color: '#e0e0e0'
                            }
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Episode',
                                color: '#e0e0e0'
                            },
                            grid: {
                                color: '#444'
                            },
                            ticks: {
                                color: '#e0e0e0'
                            }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Reward',
                                color: '#e0e0e0'
                            },
                            grid: {
                                color: '#444'
                            },
                            ticks: {
                                color: '#e0e0e0'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Loss',
                                color: '#e0e0e0'
                            },
                            grid: {
                                drawOnChartArea: false,
                                color: '#444'
                            },
                            ticks: {
                                color: '#e0e0e0'
                            }
                        }
                    }
                }
            });
        }
        
        // Price chart
        const priceCtx = document.getElementById('priceChart')?.getContext('2d');
        if (priceCtx) {
            this.priceChart = new Chart(priceCtx, {
                type: 'candlestick',
                data: {
                    datasets: [{
                        label: 'NQ Futures',
                        data: []
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'NQ Futures Price',
                            color: '#e0e0e0'
                        },
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'minute'
                            },
                            grid: {
                                color: '#444'
                            },
                            ticks: {
                                color: '#e0e0e0'
                            }
                        },
                        y: {
                            grid: {
                                color: '#444'
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
    
    subscribeToUpdates() {
        // Check if socket exists before subscribing
        if (!this.socket) {
            console.error('Socket is not defined, cannot subscribe to updates');
            return;
        }
        
        // Training updates
        this.socket.on('training_update', (data) => {
            this.updateTrainingMetrics(data);
        });
        
        // Market data updates
        this.socket.on('market_update', (data) => {
            this.updateMarketData(data);
        });
        
        // Session updates
        this.socket.on('session_update', (data) => {
            this.updateSessionInfo(data);
        });
        
        // Trade updates
        this.socket.on('trade_update', (data) => {
            this.updateTradeInfo(data);
        });
    }
    
    showNewSessionModal() {
        document.getElementById('newSessionModal').style.display = 'block';
        this.updateAlgorithmParameters('ane_ppo');
    }
    
    updateAlgorithmParameters(algorithm) {
        const paramsDiv = document.getElementById('algorithmParameters');
        if (!paramsDiv) return;
        
        let parametersHTML = '';
        
        switch(algorithm) {
            case 'ane_ppo':
                parametersHTML = `
                    <div class="form-group">
                        <label>Learning Rate</label>
                        <input type="number" id="param_learning_rate" value="0.0003" step="0.0001" min="0" max="1">
                    </div>
                    <div class="form-group">
                        <label>Batch Size</label>
                        <input type="number" id="param_batch_size" value="64" step="1" min="1">
                    </div>
                    <div class="form-group">
                        <label>Episodes</label>
                        <input type="number" id="param_episodes" value="1000" step="100" min="100">
                    </div>
                    <div class="form-group">
                        <label>Risk Level</label>
                        <select id="param_risk_level">
                            <option value="conservative">Conservative</option>
                            <option value="moderate" selected>Moderate</option>
                            <option value="aggressive">Aggressive</option>
                        </select>
                    </div>
                `;
                break;
            case 'dqn':
                parametersHTML = `
                    <div class="form-group">
                        <label>Learning Rate</label>
                        <input type="number" id="param_learning_rate" value="0.001" step="0.0001" min="0" max="1">
                    </div>
                    <div class="form-group">
                        <label>Memory Size</label>
                        <input type="number" id="param_memory_size" value="10000" step="1000" min="1000">
                    </div>
                    <div class="form-group">
                        <label>Episodes</label>
                        <input type="number" id="param_episodes" value="1000" step="100" min="100">
                    </div>
                `;
                break;
            case 'genetic':
                parametersHTML = `
                    <div class="form-group">
                        <label>Population Size</label>
                        <input type="number" id="param_population_size" value="50" step="10" min="10">
                    </div>
                    <div class="form-group">
                        <label>Generations</label>
                        <input type="number" id="param_generations" value="100" step="10" min="10">
                    </div>
                    <div class="form-group">
                        <label>Mutation Rate</label>
                        <input type="number" id="param_mutation_rate" value="0.1" step="0.01" min="0" max="1">
                    </div>
                `;
                break;
        }
        
        paramsDiv.innerHTML = parametersHTML;
    }
    
    async createNewSession() {
        const sessionName = document.getElementById('sessionName').value;
        const algorithmType = document.getElementById('algorithmType').value;
        
        if (!sessionName) {
            alert('Please enter a session name');
            return;
        }
        
        // Gather parameters
        const parameters = {};
        document.querySelectorAll('#algorithmParameters input, #algorithmParameters select').forEach(input => {
            const paramName = input.id.replace('param_', '');
            parameters[paramName] = input.type === 'number' ? parseFloat(input.value) : input.value;
        });
        
        try {
            const response = await fetch('/api/sessions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_name: sessionName,
                    algorithm_type: algorithmType,
                    parameters: parameters
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                this.sessionId = data.session_id;
                document.getElementById('newSessionModal').style.display = 'none';
                this.startTraining();
                this.updateSessionsList();
            } else {
                const error = await response.json();
                alert('Error creating session: ' + error.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to create session');
        }
    }
    
    async startTraining() {
        if (!this.sessionId) return;
        
        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/start`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.isTraining = true;
                document.getElementById('stopTraining').disabled = false;
                this.showNotification('Training started', 'success');
            } else {
                const error = await response.json();
                alert('Error starting training: ' + error.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to start training');
        }
    }
    
    async stopTraining() {
        if (!this.sessionId || !this.isTraining) return;
        
        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/stop`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.isTraining = false;
                document.getElementById('stopTraining').disabled = true;
                this.showNotification('Training stopped', 'info');
            } else {
                const error = await response.json();
                alert('Error stopping training: ' + error.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to stop training');
        }
    }
    
    async updateSessionsList() {
        try {
            const response = await fetch('/api/sessions');
            if (response.ok) {
                const sessions = await response.json();
                const sessionsList = document.getElementById('sessionsList');
                if (!sessionsList) return;
                
                sessionsList.innerHTML = sessions.map(session => `
                    <div class="session-item ${session.status === 'active' ? 'active' : ''}" 
                         onclick="tradingDashboard.selectSession(${session.id})">
                        <h4>${session.session_name}</h4>
                        <p>Algorithm: ${session.algorithm_type.toUpperCase()}</p>
                        <p>Status: <span class="status-${session.status}">${session.status}</span></p>
                        <p>Trades: ${session.total_trades} | Win Rate: ${(session.win_rate * 100).toFixed(1)}%</p>
                    </div>
                `).join('');
            }
        } catch (error) {
            console.error('Error fetching sessions:', error);
        }
    }
    
    async selectSession(sessionId) {
        this.sessionId = sessionId;
        
        // Load session details
        try {
            const response = await fetch(`/api/sessions/${sessionId}`);
            if (response.ok) {
                const session = await response.json();
                this.updateSessionInfo(session);
                
                // Load trades
                this.loadSessionTrades(sessionId);
            }
        } catch (error) {
            console.error('Error loading session:', error);
        }
    }
    
    async loadSessionTrades(sessionId) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}/trades`);
            if (response.ok) {
                const trades = await response.json();
                this.updateTradesTable(trades);
            }
        } catch (error) {
            console.error('Error loading trades:', error);
        }
    }
    
    updateSessionInfo(session) {
        document.getElementById('currentSessionName').textContent = session.session_name;
        document.getElementById('sessionProfit').textContent = `$${session.total_profit.toFixed(2)}`;
        document.getElementById('sessionWinRate').textContent = `${(session.win_rate * 100).toFixed(1)}%`;
        document.getElementById('sessionSharpe').textContent = session.sharpe_ratio.toFixed(2);
        document.getElementById('sessionDrawdown').textContent = `${(session.max_drawdown * 100).toFixed(1)}%`;
        
        this.isTraining = session.status === 'active';
        document.getElementById('stopTraining').disabled = !this.isTraining;
    }
    
    updateTrainingMetrics(data) {
        if (!this.metricsChart) return;
        
        // Add new data point
        this.metricsChart.data.labels.push(data.episode);
        this.metricsChart.data.datasets[0].data.push(data.reward);
        this.metricsChart.data.datasets[1].data.push(data.loss);
        
        // Keep only last 100 points
        if (this.metricsChart.data.labels.length > 100) {
            this.metricsChart.data.labels.shift();
            this.metricsChart.data.datasets.forEach(dataset => dataset.data.shift());
        }
        
        this.metricsChart.update();
        
        // Update episode counter
        document.getElementById('currentEpisode').textContent = data.episode;
    }
    
    updateMarketData(data) {
        // Update current price display
        document.getElementById('currentPrice').textContent = `$${data.price.toFixed(2)}`;
        
        // Update price chart
        if (this.priceChart && data.candle) {
            this.priceChart.data.datasets[0].data.push({
                x: new Date(data.timestamp),
                o: data.candle.open,
                h: data.candle.high,
                l: data.candle.low,
                c: data.candle.close
            });
            
            // Keep only last 100 candles
            if (this.priceChart.data.datasets[0].data.length > 100) {
                this.priceChart.data.datasets[0].data.shift();
            }
            
            this.priceChart.update();
        }
    }
    
    updateTradeInfo(trade) {
        // Add to trades table
        const tradesTableBody = document.querySelector('#tradesTable tbody');
        if (!tradesTableBody) return;
        
        const row = tradesTableBody.insertRow(0);
        row.innerHTML = `
            <td>${new Date(trade.entry_time).toLocaleTimeString()}</td>
            <td>${trade.position_type.toUpperCase()}</td>
            <td>$${trade.entry_price.toFixed(2)}</td>
            <td>${trade.exit_price ? '$' + trade.exit_price.toFixed(2) : '-'}</td>
            <td class="${trade.profit_loss >= 0 ? 'profit' : 'loss'}">
                $${trade.profit_loss.toFixed(2)}
            </td>
            <td><span class="status-${trade.status}">${trade.status}</span></td>
        `;
        
        // Keep only last 20 trades visible
        while (tradesTableBody.rows.length > 20) {
            tradesTableBody.deleteRow(tradesTableBody.rows.length - 1);
        }
        
        // Flash notification for new trade
        this.showNotification(`New ${trade.position_type} trade at $${trade.entry_price.toFixed(2)}`, 'info');
    }
    
    updateTradesTable(trades) {
        const tradesTableBody = document.querySelector('#tradesTable tbody');
        if (!tradesTableBody) return;
        
        tradesTableBody.innerHTML = trades.slice(0, 20).map(trade => `
            <tr>
                <td>${new Date(trade.entry_time).toLocaleTimeString()}</td>
                <td>${trade.position_type.toUpperCase()}</td>
                <td>$${trade.entry_price.toFixed(2)}</td>
                <td>${trade.exit_price ? '$' + trade.exit_price.toFixed(2) : '-'}</td>
                <td class="${trade.profit_loss >= 0 ? 'profit' : 'loss'}">
                    $${trade.profit_loss.toFixed(2)}
                </td>
                <td><span class="status-${trade.status}">${trade.status}</span></td>
            </tr>
        `).join('');
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('tradingDashboard')) {
        window.tradingDashboard = new TradingDashboard();
    }
});

// Export for use in other modules
window.TradingDashboard = TradingDashboard;