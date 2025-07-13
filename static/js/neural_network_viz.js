// Revolutionary Neural Network Visualization

class NeuralNetworkVisualization {
    constructor(containerId) {
        this.containerId = containerId;
        this.canvas = null;
        this.ctx = null;
        this.animationId = null;
        
        // Network structure
        this.layers = [
            { neurons: 60, type: 'input', name: 'Market Data' },
            { neurons: 128, type: 'hidden', name: 'Feature Extraction' },
            { neurons: 256, type: 'attention', name: 'Attention Layer' },
            { neurons: 128, type: 'hidden', name: 'Policy Network' },
            { neurons: 64, type: 'hidden', name: 'Value Network' },
            { neurons: 3, type: 'output', name: 'Actions' }
        ];
        
        // Visualization parameters
        this.neuronRadius = 4;
        this.layerSpacing = 120;
        this.neuronSpacing = 8;
        this.connectionOpacity = 0.3;
        this.activationThreshold = 0.5;
        
        // Animation state
        this.time = 0;
        this.weights = new Map();
        this.activations = new Map();
        this.pulses = [];
        
        // Colors
        this.colors = {
            neuron: '#00ff88',
            connection: '#ffffff',
            activation: '#0088ff',
            attention: '#ff0088',
            background: 'rgba(0, 0, 0, 0.1)'
        };
        
        console.log('🧠 Neural Network Visualization initialized');
    }
    
    init() {
        // Delay initialization to ensure canvas is rendered
        setTimeout(() => {
            this.setupCanvas();
            this.calculateLayout();
            this.initializeWeights();
            this.setupEventListeners();
            this.startAnimation();
            
            console.log('✅ Neural Network Visualization ready');
        }, 100);
    }
    
    setupCanvas() {
        const container = document.getElementById(this.containerId);
        
        this.canvas = document.createElement('canvas');
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        
        this.ctx = this.canvas.getContext('2d');
        this.ctx.imageSmoothingEnabled = true;
        
        container.appendChild(this.canvas);
    }
    
    calculateLayout() {
        // const width = this.canvas.width; // Not used currently
        const height = this.canvas.height;
        
        this.layers.forEach((layer, layerIndex) => {
            layer.x = (layerIndex * this.layerSpacing) + 60;
            layer.neurons_data = [];
            
            const totalHeight = (layer.neurons - 1) * this.neuronSpacing;
            const startY = (height - totalHeight) / 2;
            
            for (let neuronIndex = 0; neuronIndex < layer.neurons; neuronIndex++) {
                const neuron = {
                    x: layer.x,
                    y: startY + (neuronIndex * this.neuronSpacing),
                    activation: Math.random(),
                    bias: (Math.random() - 0.5) * 2,
                    layerIndex: layerIndex,
                    neuronIndex: neuronIndex
                };
                
                layer.neurons_data.push(neuron);
            }
        });
    }
    
    initializeWeights() {
        for (let i = 0; i < this.layers.length - 1; i++) {
            const currentLayer = this.layers[i];
            const nextLayer = this.layers[i + 1];
            
            for (let j = 0; j < currentLayer.neurons; j++) {
                for (let k = 0; k < nextLayer.neurons; k++) {
                    const weightKey = `${i}-${j}-${i + 1}-${k}`;
                    this.weights.set(weightKey, (Math.random() - 0.5) * 2);
                }
            }
        }
    }
    
    setupEventListeners() {
        window.addEventListener('resize', () => {
            this.handleResize();
        });
        
        this.canvas.addEventListener('mousemove', (event) => {
            this.handleMouseMove(event);
        });
        
        this.canvas.addEventListener('click', (event) => {
            this.handleClick(event);
        });
        
        // Listen for training updates
        document.addEventListener('neuralNetworkUpdate', (event) => {
            this.updateWeights(event.detail);
        });
    }
    
    handleResize() {
        const container = document.getElementById(this.containerId);
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
        this.calculateLayout();
    }
    
    handleMouseMove(event) {
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;
        
        // Check if hovering over a neuron
        this.layers.forEach(layer => {
            layer.neurons_data.forEach(neuron => {
                const distance = Math.sqrt(
                    Math.pow(mouseX - neuron.x, 2) + 
                    Math.pow(mouseY - neuron.y, 2)
                );
                
                neuron.isHovered = distance <= this.neuronRadius * 2;
            });
        });
    }
    
    handleClick(event) {
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;
        
        // Find clicked neuron
        this.layers.forEach(layer => {
            layer.neurons_data.forEach(neuron => {
                const distance = Math.sqrt(
                    Math.pow(mouseX - neuron.x, 2) + 
                    Math.pow(mouseY - neuron.y, 2)
                );
                
                if (distance <= this.neuronRadius * 2) {
                    this.highlightNeuron(neuron);
                    this.showNeuronInfo(neuron, event);
                }
            });
        });
    }
    
    highlightNeuron(neuron) {
        // Add pulse effect
        this.pulses.push({
            x: neuron.x,
            y: neuron.y,
            radius: this.neuronRadius,
            opacity: 1,
            maxRadius: this.neuronRadius * 8,
            color: this.colors.attention
        });
        
        // Highlight connections
        neuron.highlighted = true;
        setTimeout(() => {
            neuron.highlighted = false;
        }, 2000);
    }
    
    showNeuronInfo(neuron, event) {
        const info = `
            Layer: ${this.layers[neuron.layerIndex].name}
            Neuron: ${neuron.neuronIndex}
            Activation: ${neuron.activation.toFixed(3)}
            Bias: ${neuron.bias.toFixed(3)}
        `;
        
        // Create tooltip
        const tooltip = document.createElement('div');
        tooltip.className = 'neural-tooltip';
        tooltip.style.position = 'absolute';
        tooltip.style.left = event.pageX + 10 + 'px';
        tooltip.style.top = event.pageY + 10 + 'px';
        tooltip.style.background = 'rgba(0, 0, 0, 0.9)';
        tooltip.style.color = '#00ff88';
        tooltip.style.padding = '10px';
        tooltip.style.borderRadius = '5px';
        tooltip.style.fontSize = '12px';
        tooltip.style.fontFamily = 'monospace';
        tooltip.style.zIndex = '1000';
        tooltip.style.pointerEvents = 'none';
        tooltip.innerHTML = info.replace(/\n/g, '<br>');
        
        document.body.appendChild(tooltip);
        
        setTimeout(() => {
            tooltip.remove();
        }, 3000);
    }
    
    updateWeights(networkState) {
        if (networkState.weights) {
            // Update weights from training data
            Object.entries(networkState.weights).forEach(([key, value]) => {
                this.weights.set(key, value);
            });
        }
        
        if (networkState.activations) {
            // Update neuron activations
            networkState.activations.forEach((layerActivations, layerIndex) => {
                if (layerIndex < this.layers.length) {
                    layerActivations.forEach((activation, neuronIndex) => {
                        if (neuronIndex < this.layers[layerIndex].neurons_data.length) {
                            this.layers[layerIndex].neurons_data[neuronIndex].activation = activation;
                        }
                    });
                }
            });
        }
        
        // Add training pulse effect
        this.addTrainingPulse();
    }
    
    addTrainingPulse() {
        // Create pulse that travels through the network
        const startNeuron = this.layers[0].neurons_data[Math.floor(Math.random() * this.layers[0].neurons)];
        
        this.pulses.push({
            x: startNeuron.x,
            y: startNeuron.y,
            radius: 2,
            opacity: 1,
            maxRadius: 20,
            color: this.colors.activation,
            speed: 2
        });
    }
    
    startAnimation() {
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);
            
            this.time += 0.016; // ~60fps
            this.updateAnimations();
            this.draw();
        };
        
        animate();
    }
    
    updateAnimations() {
        // Update neuron activations with sine waves
        this.layers.forEach((layer, layerIndex) => {
            layer.neurons_data.forEach((neuron, neuronIndex) => {
                // Simulate neural activity
                const baseActivation = neuron.activation;
                const noise = Math.sin(this.time * 2 + neuronIndex * 0.1) * 0.1;
                neuron.currentActivation = Math.max(0, Math.min(1, baseActivation + noise));
                
                // Add sparkle effect for high activation
                if (neuron.currentActivation > 0.8 && Math.random() > 0.98) {
                    this.pulses.push({
                        x: neuron.x + (Math.random() - 0.5) * 10,
                        y: neuron.y + (Math.random() - 0.5) * 10,
                        radius: 1,
                        opacity: 1,
                        maxRadius: 5,
                        color: this.colors.neuron
                    });
                }
            });
        });
        
        // Update pulses
        this.pulses = this.pulses.filter(pulse => {
            pulse.radius += pulse.speed || 1;
            pulse.opacity = 1 - (pulse.radius / pulse.maxRadius);
            return pulse.opacity > 0;
        });
        
        // Simulate attention mechanism
        this.updateAttentionVisualization();
    }
    
    updateAttentionVisualization() {
        const attentionLayer = this.layers.find(layer => layer.type === 'attention');
        if (!attentionLayer) return;
        
        // Create attention weights visualization
        attentionLayer.neurons_data.forEach((neuron, index) => {
            const attentionWeight = Math.sin(this.time * 0.5 + index * 0.2) * 0.5 + 0.5;
            neuron.attentionWeight = attentionWeight;
            
            // Create attention beam effect
            if (attentionWeight > 0.7) {
                const inputLayer = this.layers[0];
                const targetNeuron = inputLayer.neurons_data[Math.floor(Math.random() * inputLayer.neurons)];
                
                this.createAttentionBeam(neuron, targetNeuron);
            }
        });
    }
    
    createAttentionBeam(fromNeuron, toNeuron) {
        // Create animated beam between neurons
        const beam = {
            fromX: fromNeuron.x,
            fromY: fromNeuron.y,
            toX: toNeuron.x,
            toY: toNeuron.y,
            opacity: 1,
            width: 2,
            color: this.colors.attention,
            duration: 30 // frames
        };
        
        if (!this.attentionBeams) {
            this.attentionBeams = [];
        }
        
        this.attentionBeams.push(beam);
        
        // Remove beam after duration
        setTimeout(() => {
            const index = this.attentionBeams.indexOf(beam);
            if (index > -1) {
                this.attentionBeams.splice(index, 1);
            }
        }, beam.duration * 16); // Convert frames to ms
    }
    
    draw() {
        // Clear canvas
        this.ctx.fillStyle = this.colors.background;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw connections
        this.drawConnections();
        
        // Draw attention beams
        this.drawAttentionBeams();
        
        // Draw neurons
        this.drawNeurons();
        
        // Draw layer labels
        this.drawLayerLabels();
        
        // Draw pulses
        this.drawPulses();
        
        // Draw performance metrics
        this.drawPerformanceMetrics();
    }
    
    drawConnections() {
        for (let i = 0; i < this.layers.length - 1; i++) {
            const currentLayer = this.layers[i];
            const nextLayer = this.layers[i + 1];
            
            currentLayer.neurons_data.forEach((fromNeuron, fromIndex) => {
                nextLayer.neurons_data.forEach((toNeuron, toIndex) => {
                    const weightKey = `${i}-${fromIndex}-${i + 1}-${toIndex}`;
                    const weight = this.weights.get(weightKey) || 0;
                    
                    // Skip very weak connections
                    if (Math.abs(weight) < 0.1) return;
                    
                    const opacity = Math.abs(weight) * this.connectionOpacity;
                    const width = Math.abs(weight) * 2;
                    
                    this.ctx.beginPath();
                    this.ctx.moveTo(fromNeuron.x, fromNeuron.y);
                    this.ctx.lineTo(toNeuron.x, toNeuron.y);
                    
                    this.ctx.strokeStyle = weight > 0 ? 
                        `rgba(0, 255, 136, ${opacity})` : 
                        `rgba(255, 0, 136, ${opacity})`;
                    this.ctx.lineWidth = width;
                    this.ctx.stroke();
                    
                    // Add flowing animation for strong connections
                    if (Math.abs(weight) > 0.7) {
                        this.drawFlowingConnection(fromNeuron, toNeuron, weight);
                    }
                });
            });
        }
    }
    
    drawFlowingConnection(fromNeuron, toNeuron, weight) {
        const flowSpeed = 0.05;
        const flowOffset = (this.time * flowSpeed) % 1;
        
        const dx = toNeuron.x - fromNeuron.x;
        const dy = toNeuron.y - fromNeuron.y;
        
        const flowX = fromNeuron.x + dx * flowOffset;
        const flowY = fromNeuron.y + dy * flowOffset;
        
        this.ctx.beginPath();
        this.ctx.arc(flowX, flowY, 2, 0, Math.PI * 2);
        this.ctx.fillStyle = weight > 0 ? '#00ff88' : '#ff0088';
        this.ctx.fill();
    }
    
    drawAttentionBeams() {
        if (!this.attentionBeams) return;
        
        this.attentionBeams.forEach(beam => {
            this.ctx.beginPath();
            this.ctx.moveTo(beam.fromX, beam.fromY);
            this.ctx.lineTo(beam.toX, beam.toY);
            
            this.ctx.strokeStyle = `rgba(255, 0, 136, ${beam.opacity})`;
            this.ctx.lineWidth = beam.width;
            this.ctx.setLineDash([5, 5]);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
            
            beam.opacity *= 0.95; // Fade out
        });
    }
    
    drawNeurons() {
        this.layers.forEach((layer, layerIndex) => {
            layer.neurons_data.forEach((neuron, neuronIndex) => {
                const activation = neuron.currentActivation || neuron.activation;
                const radius = this.neuronRadius + (activation * 2);
                
                // Neuron body
                this.ctx.beginPath();
                this.ctx.arc(neuron.x, neuron.y, radius, 0, Math.PI * 2);
                
                // Color based on activation and type
                let color = this.colors.neuron;
                if (layer.type === 'attention' && neuron.attentionWeight) {
                    color = this.colors.attention;
                } else if (activation > this.activationThreshold) {
                    color = this.colors.activation;
                }
                
                const alpha = 0.3 + (activation * 0.7);
                this.ctx.fillStyle = color.replace(')', `, ${alpha})`).replace('#', 'rgba(').replace(/(.{2})/g, (match, p1, offset) => {
                    if (offset === 0) return parseInt(p1, 16) + ', ';
                    if (offset === 2) return parseInt(p1, 16) + ', ';
                    if (offset === 4) return parseInt(p1, 16) + ', ';
                    return p1;
                });
                
                // Convert hex to rgba
                const hex = color.replace('#', '');
                const r = parseInt(hex.substr(0, 2), 16);
                const g = parseInt(hex.substr(2, 2), 16);
                const b = parseInt(hex.substr(4, 2), 16);
                this.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
                
                this.ctx.fill();
                
                // Neuron border
                this.ctx.strokeStyle = color;
                this.ctx.lineWidth = neuron.isHovered ? 3 : 1;
                this.ctx.stroke();
                
                // Highlight effect
                if (neuron.highlighted) {
                    this.ctx.beginPath();
                    this.ctx.arc(neuron.x, neuron.y, radius + 5, 0, Math.PI * 2);
                    this.ctx.strokeStyle = this.colors.attention;
                    this.ctx.lineWidth = 2;
                    this.ctx.stroke();
                }
                
                // Activity indicator
                if (activation > 0.9) {
                    this.ctx.beginPath();
                    this.ctx.arc(neuron.x, neuron.y, radius + 3, 0, Math.PI * 2);
                    this.ctx.strokeStyle = '#ffffff';
                    this.ctx.lineWidth = 1;
                    this.ctx.setLineDash([2, 2]);
                    this.ctx.stroke();
                    this.ctx.setLineDash([]);
                }
            });
        });
    }
    
    drawLayerLabels() {
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillStyle = '#ffffff';
        
        this.layers.forEach(layer => {
            this.ctx.fillText(layer.name, layer.x, 30);
            this.ctx.fillText(`(${layer.neurons})`, layer.x, 45);
        });
    }
    
    drawPulses() {
        this.pulses.forEach(pulse => {
            this.ctx.beginPath();
            this.ctx.arc(pulse.x, pulse.y, pulse.radius, 0, Math.PI * 2);
            
            const hex = pulse.color.replace('#', '');
            const r = parseInt(hex.substr(0, 2), 16);
            const g = parseInt(hex.substr(2, 2), 16);
            const b = parseInt(hex.substr(4, 2), 16);
            
            this.ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${pulse.opacity})`;
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        });
    }
    
    drawPerformanceMetrics() {
        // Draw mini performance chart in corner
        const chartX = this.canvas.width - 200;
        const chartY = this.canvas.height - 100;
        const chartWidth = 180;
        const chartHeight = 80;
        
        // Background
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(chartX, chartY, chartWidth, chartHeight);
        
        // Border
        this.ctx.strokeStyle = '#00ff88';
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(chartX, chartY, chartWidth, chartHeight);
        
        // Mock performance data
        this.ctx.beginPath();
        this.ctx.strokeStyle = '#00ff88';
        this.ctx.lineWidth = 2;
        
        for (let i = 0; i < chartWidth; i += 5) {
            const x = chartX + i;
            const y = chartY + chartHeight/2 + Math.sin(this.time * 0.1 + i * 0.01) * 20;
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        
        this.ctx.stroke();
        
        // Label
        this.ctx.font = '10px Arial';
        this.ctx.fillStyle = '#ffffff';
        this.ctx.textAlign = 'left';
        this.ctx.fillText('Network Performance', chartX + 5, chartY + 15);
    }
    
    simulateTraining() {
        // Simulate training by updating random weights
        const weightKeys = Array.from(this.weights.keys());
        const randomKey = weightKeys[Math.floor(Math.random() * weightKeys.length)];
        const currentWeight = this.weights.get(randomKey);
        const newWeight = currentWeight + (Math.random() - 0.5) * 0.1;
        this.weights.set(randomKey, Math.max(-1, Math.min(1, newWeight)));
        
        // Update random neuron activations
        this.layers.forEach(layer => {
            if (Math.random() > 0.9) {
                const randomNeuron = layer.neurons_data[Math.floor(Math.random() * layer.neurons_data.length)];
                randomNeuron.activation = Math.random();
            }
        });
        
        this.addTrainingPulse();
    }
    
    setLearningRate(rate) {
        // Visual indication of learning rate
        this.learningRate = rate;
        this.pulseSpeed = rate * 10;
    }
    
    setTrainingProgress(progress) {
        // Update visual elements based on training progress
        this.trainingProgress = progress;
        
        // Gradually make connections stronger as training progresses
        this.connectionOpacity = 0.1 + (progress * 0.4);
    }
    
    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    dispose() {
        this.stopAnimation();
        
        const container = document.getElementById(this.containerId);
        if (container && this.canvas) {
            container.removeChild(this.canvas);
        }
    }
}

// Export for global use
window.NeuralNetworkVisualization = NeuralNetworkVisualization;

console.log('🧠 Neural Network Visualization module loaded');
