/**
 * Learning Visualizations for AI Trading System
 * Shows real-time animations and charts for PPO, Genetic Algorithm, and ANE model learning
 */

class LearningVisualizations {
    constructor() {
        this.charts = {};
        this.animations = {};
        this.geneticPopulation = [];
        this.ppoMetrics = {
            policyLoss: [],
            valueLoss: [],
            entropy: [],
            rewards: [],
            advantages: []
        };
        this.aneMetrics = {
            attentionWeights: [],
            transformerOutputs: [],
            marketRegimes: [],
            confidenceScores: []
        };
        
        this.initializeVisualizations();
        this.setupWebSocketHandlers();
    }
    
    initializeVisualizations() {
        // Create containers for visualizations
        this.createVisualizationContainers();
        
        // Initialize PPO Learning Chart
        this.initializePPOChart();
        
        // Initialize Genetic Algorithm Visualization
        this.initializeGeneticVisualization();
        
        // Initialize ANE Model Attention Heatmap
        this.initializeANEVisualization();
        
        // Initialize Learning Progress Animation
        this.initializeLearningAnimation();
    }
    
    createVisualizationContainers() {
        const container = document.getElementById('learningVisualizationsContainer');
        if (!container) {
            console.warn('Learning visualizations container not found');
            return;
        }
        
        container.innerHTML = `
            <div class="learning-viz-grid">
                <!-- Model Selection Panel -->
                <div class="viz-panel model-selection-panel">
                    <h3>Active Learning Model</h3>
                    <div id="modelSelectionAnimation" class="model-animation"></div>
                    <div class="model-info">
                        <div class="model-name" id="activeModelName">ANE-PPO</div>
                        <div class="model-description" id="modelDescription">
                            Advanced Neural Evolution with Proximal Policy Optimization
                        </div>
                    </div>
                </div>
                
                <!-- PPO Learning Visualization -->
                <div class="viz-panel ppo-panel">
                    <h3>PPO Learning Progress</h3>
                    <canvas id="ppoLearningChart"></canvas>
                    <div class="ppo-metrics">
                        <div class="metric">
                            <span class="label">Policy Loss:</span>
                            <span class="value" id="ppoLolicyLoss">0.000</span>
                        </div>
                        <div class="metric">
                            <span class="label">Value Loss:</span>
                            <span class="value" id="ppoValueLoss">0.000</span>
                        </div>
                        <div class="metric">
                            <span class="label">Entropy:</span>
                            <span class="value" id="ppoEntropy">0.000</span>
                        </div>
                    </div>
                </div>
                
                <!-- Genetic Algorithm Evolution -->
                <div class="viz-panel genetic-panel">
                    <h3>Genetic Algorithm Evolution</h3>
                    <div id="geneticEvolutionCanvas" class="genetic-canvas"></div>
                    <div class="genetic-info">
                        <div class="generation-counter">Generation: <span id="genCounter">0</span></div>
                        <div class="fitness-score">Best Fitness: <span id="bestFitness">0.000</span></div>
                        <div class="convergence-status" id="convergenceStatus">Evolving...</div>
                    </div>
                </div>
                
                <!-- ANE Attention Visualization -->
                <div class="viz-panel ane-panel">
                    <h3>ANE Transformer Attention</h3>
                    <div id="aneAttentionHeatmap" class="attention-heatmap"></div>
                    <div class="market-regimes">
                        <div class="regime" id="regime-bull" data-regime="bull">Bull</div>
                        <div class="regime" id="regime-bear" data-regime="bear">Bear</div>
                        <div class="regime" id="regime-sideways" data-regime="sideways">Sideways</div>
                        <div class="regime" id="regime-volatile" data-regime="volatile">Volatile</div>
                    </div>
                </div>
                
                <!-- Learning Animation -->
                <div class="viz-panel learning-animation-panel">
                    <h3>Neural Network Learning Flow</h3>
                    <div id="learningFlowCanvas"></div>
                </div>
                
                <!-- Real-time Insights -->
                <div class="viz-panel insights-panel">
                    <h3>AI Learning Insights</h3>
                    <div id="learningInsights" class="insights-feed"></div>
                </div>
            </div>
        `;
    }
    
    initializePPOChart() {
        const ctx = document.getElementById('ppoLearningChart');
        if (!ctx) return;
        
        this.charts.ppoChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Policy Loss',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Value Loss',
                        data: [],
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Reward',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#aaa' }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#aaa' }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#aaa' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#fff' }
                    }
                }
            }
        });
    }
    
    initializeGeneticVisualization() {
        const canvas = document.getElementById('geneticEvolutionCanvas');
        if (!canvas) return;
        
        // Create SVG for genetic algorithm visualization
        const svg = d3.select(canvas)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '300px');
        
        this.geneticViz = {
            svg: svg,
            width: canvas.offsetWidth,
            height: 300
        };
        
        // Initialize population dots
        this.updateGeneticPopulation([]);
    }
    
    initializeANEVisualization() {
        const container = document.getElementById('aneAttentionHeatmap');
        if (!container) return;
        
        // Create attention heatmap using D3
        const width = container.offsetWidth;
        const height = 200;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        this.aneViz = {
            svg: svg,
            width: width,
            height: height,
            cellSize: 20
        };
        
        // Initialize with empty heatmap
        this.updateAttentionHeatmap([]);
    }
    
    initializeLearningAnimation() {
        const canvas = document.getElementById('learningFlowCanvas');
        if (!canvas) return;
        
        // Create animated neural network flow
        this.learningFlow = new LearningFlowAnimation(canvas);
        this.learningFlow.start();
    }
    
    updatePPOMetrics(data) {
        // Update PPO chart
        const chart = this.charts.ppoChart;
        if (!chart) return;
        
        // Add new data points
        chart.data.labels.push(data.episode || chart.data.labels.length);
        chart.data.datasets[0].data.push(data.policyLoss || 0);
        chart.data.datasets[1].data.push(data.valueLoss || 0);
        chart.data.datasets[2].data.push(data.reward || 0);
        
        // Keep only last 50 points
        if (chart.data.labels.length > 50) {
            chart.data.labels.shift();
            chart.data.datasets.forEach(dataset => dataset.data.shift());
        }
        
        chart.update('none'); // No animation for performance
        
        // Update metric displays
        document.getElementById('ppoLolicyLoss').textContent = (data.policyLoss || 0).toFixed(4);
        document.getElementById('ppoValueLoss').textContent = (data.valueLoss || 0).toFixed(4);
        document.getElementById('ppoEntropy').textContent = (data.entropy || 0).toFixed(4);
        
        // Add insight
        this.addLearningInsight(`PPO Update: Policy improving by ${(data.improvement || 0).toFixed(2)}%`);
    }
    
    updateGeneticPopulation(populationData) {
        if (!this.geneticViz) return;
        
        const { svg, width, height } = this.geneticViz;
        
        // Update generation counter
        const generation = populationData.generation || 0;
        document.getElementById('genCounter').textContent = generation;
        
        // Update best fitness
        const bestFitness = populationData.bestFitness || 0;
        document.getElementById('bestFitness').textContent = bestFitness.toFixed(3);
        
        // Update convergence status
        if (populationData.converged) {
            document.getElementById('convergenceStatus').textContent = `Converged at Gen ${generation}!`;
            document.getElementById('convergenceStatus').classList.add('converged');
        }
        
        // Visualize population
        const individuals = populationData.individuals || [];
        
        const circles = svg.selectAll('circle')
            .data(individuals);
        
        circles.enter()
            .append('circle')
            .merge(circles)
            .transition()
            .duration(500)
            .attr('cx', d => Math.random() * width)
            .attr('cy', d => (1 - d.fitness) * height)
            .attr('r', d => 3 + d.fitness * 10)
            .attr('fill', d => d.isElite ? '#FFD700' : '#00CED1')
            .attr('opacity', d => 0.6 + d.fitness * 0.4);
        
        circles.exit().remove();
        
        // Add evolution trail
        if (generation % 10 === 0) {
            svg.append('line')
                .attr('x1', 0)
                .attr('y1', (1 - bestFitness) * height)
                .attr('x2', width)
                .attr('y2', (1 - bestFitness) * height)
                .attr('stroke', '#FFD700')
                .attr('stroke-width', 1)
                .attr('opacity', 0.3)
                .attr('stroke-dasharray', '5,5');
        }
        
        // Add insight
        if (generation > 0) {
            this.addLearningInsight(`Generation ${generation}: Population fitness improving, ${individuals.filter(i => i.isElite).length} elite individuals`);
        }
    }
    
    updateAttentionHeatmap(attentionData) {
        if (!this.aneViz) return;
        
        const { svg, cellSize } = this.aneViz;
        
        // Create heatmap cells
        const cells = svg.selectAll('rect')
            .data(attentionData.flat());
        
        const colorScale = d3.scaleSequential(d3.interpolateYlOrRd)
            .domain([0, 1]);
        
        cells.enter()
            .append('rect')
            .merge(cells)
            .attr('x', (d, i) => (i % 10) * cellSize)
            .attr('y', (d, i) => Math.floor(i / 10) * cellSize)
            .attr('width', cellSize - 2)
            .attr('height', cellSize - 2)
            .attr('fill', d => colorScale(d.value || 0))
            .attr('opacity', 0.8);
        
        cells.exit().remove();
    }
    
    updateMarketRegime(regime) {
        // Update regime indicators
        document.querySelectorAll('.regime').forEach(el => {
            el.classList.remove('active');
        });
        
        const regimeEl = document.getElementById(`regime-${regime}`);
        if (regimeEl) {
            regimeEl.classList.add('active');
        }
        
        // Add insight
        this.addLearningInsight(`Market regime detected: ${regime.toUpperCase()} - Adjusting trading strategy`);
    }
    
    addLearningInsight(message) {
        const container = document.getElementById('learningInsights');
        if (!container) return;
        
        const insight = document.createElement('div');
        insight.className = 'insight-item';
        insight.innerHTML = `
            <span class="timestamp">${new Date().toLocaleTimeString()}</span>
            <span class="message">${message}</span>
        `;
        
        container.insertBefore(insight, container.firstChild);
        
        // Keep only last 10 insights
        while (container.children.length > 10) {
            container.removeChild(container.lastChild);
        }
        
        // Animate entry
        insight.style.opacity = '0';
        insight.style.transform = 'translateY(-10px)';
        setTimeout(() => {
            insight.style.opacity = '1';
            insight.style.transform = 'translateY(0)';
        }, 10);
    }
    
    updateModelSelection(modelType) {
        const nameEl = document.getElementById('activeModelName');
        const descEl = document.getElementById('modelDescription');
        
        const models = {
            'ANE_PPO': {
                name: 'ANE-PPO',
                description: 'Advanced Neural Evolution with Proximal Policy Optimization - Combines evolutionary strategies with policy gradients'
            },
            'DQN': {
                name: 'DQN',
                description: 'Deep Q-Network - Value-based reinforcement learning with experience replay'
            },
            'GENETIC': {
                name: 'Genetic Algorithm',
                description: 'Evolutionary optimization through natural selection and mutation'
            }
        };
        
        const model = models[modelType] || models['ANE_PPO'];
        nameEl.textContent = model.name;
        descEl.textContent = model.description;
        
        // Animate model change
        this.animateModelChange(modelType);
    }
    
    animateModelChange(modelType) {
        const canvas = document.getElementById('modelSelectionAnimation');
        if (!canvas) return;
        
        // Create particle effect for model change
        canvas.innerHTML = '<div class="model-particles"></div>';
        
        for (let i = 0; i < 20; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = `${Math.random() * 100}%`;
            particle.style.animationDelay = `${Math.random() * 2}s`;
            canvas.querySelector('.model-particles').appendChild(particle);
        }
    }
    
    setupWebSocketHandlers() {
        if (window.socket) {
            window.socket.on('learning_update', (data) => {
                if (data.type === 'ppo') {
                    this.updatePPOMetrics(data);
                } else if (data.type === 'genetic') {
                    this.updateGeneticPopulation(data);
                } else if (data.type === 'attention') {
                    this.updateAttentionHeatmap(data);
                } else if (data.type === 'regime') {
                    this.updateMarketRegime(data.regime);
                }
            });
            
            window.socket.on('model_change', (data) => {
                this.updateModelSelection(data.model);
            });
        }
    }
}

// Learning Flow Animation Class
class LearningFlowAnimation {
    constructor(container) {
        this.container = container;
        this.particles = [];
        this.connections = [];
        this.animationId = null;
        
        this.initCanvas();
    }
    
    initCanvas() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.container.appendChild(this.canvas);
        
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }
    
    resize() {
        this.canvas.width = this.container.offsetWidth;
        this.canvas.height = 250;
    }
    
    start() {
        this.createNetwork();
        this.animate();
    }
    
    createNetwork() {
        // Create neural network structure
        const layers = [3, 5, 4, 2];
        const layerSpacing = this.canvas.width / (layers.length + 1);
        
        layers.forEach((nodeCount, layerIndex) => {
            const x = layerSpacing * (layerIndex + 1);
            const nodeSpacing = this.canvas.height / (nodeCount + 1);
            
            for (let i = 0; i < nodeCount; i++) {
                const y = nodeSpacing * (i + 1);
                this.particles.push({
                    x: x,
                    y: y,
                    layer: layerIndex,
                    activation: 0,
                    pulsePhase: Math.random() * Math.PI * 2
                });
            }
        });
        
        // Create connections
        for (let i = 0; i < layers.length - 1; i++) {
            const currentLayer = this.particles.filter(p => p.layer === i);
            const nextLayer = this.particles.filter(p => p.layer === i + 1);
            
            currentLayer.forEach(node1 => {
                nextLayer.forEach(node2 => {
                    this.connections.push({
                        from: node1,
                        to: node2,
                        strength: Math.random(),
                        pulsePosition: 0
                    });
                });
            });
        }
    }
    
    animate() {
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Update and draw connections
        this.connections.forEach(conn => {
            conn.pulsePosition += 0.02;
            if (conn.pulsePosition > 1) {
                conn.pulsePosition = 0;
                conn.strength = Math.min(1, conn.strength + 0.1);
            }
            
            // Draw connection
            this.ctx.beginPath();
            this.ctx.moveTo(conn.from.x, conn.from.y);
            this.ctx.lineTo(conn.to.x, conn.to.y);
            this.ctx.strokeStyle = `rgba(0, 255, 255, ${conn.strength * 0.3})`;
            this.ctx.lineWidth = conn.strength * 2;
            this.ctx.stroke();
            
            // Draw pulse
            if (conn.pulsePosition > 0 && conn.pulsePosition < 1) {
                const pulseX = conn.from.x + (conn.to.x - conn.from.x) * conn.pulsePosition;
                const pulseY = conn.from.y + (conn.to.y - conn.from.y) * conn.pulsePosition;
                
                this.ctx.beginPath();
                this.ctx.arc(pulseX, pulseY, 3, 0, Math.PI * 2);
                this.ctx.fillStyle = `rgba(255, 255, 0, ${1 - conn.pulsePosition})`;
                this.ctx.fill();
            }
        });
        
        // Update and draw nodes
        this.particles.forEach((node, index) => {
            node.activation = Math.sin(Date.now() * 0.001 + node.pulsePhase) * 0.5 + 0.5;
            
            // Draw node
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, 8 + node.activation * 4, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(0, 255, 255, ${0.5 + node.activation * 0.5})`;
            this.ctx.fill();
            this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        });
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    stop() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.learningViz = new LearningVisualizations();
});