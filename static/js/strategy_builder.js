// Strategy Builder Class
class StrategyBuilder {
    constructor() {
        this.canvas = document.getElementById('strategy-canvas');
        this.svg = document.getElementById('connection-svg');
        this.components = [];
        this.connections = [];
        this.selectedComponent = null;
        this.draggedElement = null;
        this.nextComponentId = 0;
        
        this.initializeDragAndDrop();
        this.initializeEventListeners();
    }
    
    initializeDragAndDrop() {
        // Initialize drag events for palette components
        const paletteComponents = document.querySelectorAll('.algorithm-component');
        paletteComponents.forEach(component => {
            component.addEventListener('dragstart', this.handleDragStart.bind(this));
            component.addEventListener('dragend', this.handleDragEnd.bind(this));
            component.draggable = true;
        });
        
        // Initialize drop events for canvas
        this.canvas.addEventListener('dragover', this.handleDragOver.bind(this));
        this.canvas.addEventListener('drop', this.handleDrop.bind(this));
    }
    
    initializeEventListeners() {
        // Canvas click handler for deselecting
        this.canvas.addEventListener('click', (e) => {
            if (e.target === this.canvas) {
                this.deselectAll();
            }
        });
        
        // Save configuration button
        const saveConfigBtn = document.getElementById('save-config');
        if (saveConfigBtn) {
            saveConfigBtn.addEventListener('click', this.saveComponentConfig.bind(this));
        }
    }
    
    handleDragStart(e) {
        this.draggedElement = e.target;
        e.dataTransfer.effectAllowed = 'copy';
        e.dataTransfer.setData('componentType', e.target.dataset.type);
    }
    
    handleDragEnd(e) {
        this.draggedElement = null;
    }
    
    handleDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    }
    
    handleDrop(e) {
        e.preventDefault();
        
        if (!this.draggedElement) return;
        
        const componentType = e.dataTransfer.getData('componentType');
        const canvasRect = this.canvas.getBoundingClientRect();
        const x = e.clientX - canvasRect.left;
        const y = e.clientY - canvasRect.top;
        
        this.addComponent(componentType, x, y);
        
        // Hide placeholder after first component
        const placeholder = document.getElementById('canvas-placeholder');
        if (placeholder) {
            placeholder.style.display = 'none';
        }
    }
    
    addComponent(type, x, y) {
        const componentId = this.nextComponentId++;
        
        // Create component node
        const node = document.createElement('div');
        node.className = 'component-node';
        node.id = `component-${componentId}`;
        node.style.left = `${x - 100}px`;
        node.style.top = `${y - 50}px`;
        
        // Get component info
        const componentInfo = this.getComponentInfo(type);
        
        node.innerHTML = `
            <h6 class="text-white mb-2">
                <i class="${componentInfo.icon} me-2"></i>${componentInfo.name}
            </h6>
            <p class="text-white small mb-2">${componentInfo.description}</p>
            <div class="connection-ports">
                <div class="input-port" data-component="${componentId}" data-port="input"></div>
                <div class="output-port" data-component="${componentId}" data-port="output"></div>
            </div>
        `;
        
        // Add click handler
        node.addEventListener('click', (e) => {
            e.stopPropagation();
            this.selectComponent(componentId);
        });
        
        // Add double-click handler for configuration
        node.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            this.openComponentConfig(componentId);
        });
        
        // Make node draggable
        this.makeNodeDraggable(node, componentId);
        
        // Add to canvas
        this.canvas.appendChild(node);
        
        // Store component data
        this.components.push({
            id: componentId,
            type: type,
            position: { x: x - 100, y: y - 50 },
            config: this.getDefaultConfig(type),
            node: node
        });
    }
    
    makeNodeDraggable(node, componentId) {
        let isDragging = false;
        let offsetX, offsetY;
        
        node.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('input-port') || e.target.classList.contains('output-port')) {
                return; // Don't drag when clicking on ports
            }
            
            isDragging = true;
            const rect = node.getBoundingClientRect();
            const canvasRect = this.canvas.getBoundingClientRect();
            
            offsetX = e.clientX - rect.left;
            offsetY = e.clientY - rect.top;
            
            node.style.cursor = 'grabbing';
            node.style.zIndex = '1000';
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            
            const canvasRect = this.canvas.getBoundingClientRect();
            const x = e.clientX - canvasRect.left - offsetX;
            const y = e.clientY - canvasRect.top - offsetY;
            
            node.style.left = `${x}px`;
            node.style.top = `${y}px`;
            
            // Update component position
            const component = this.components.find(c => c.id === componentId);
            if (component) {
                component.position = { x, y };
            }
            
            // Update connections
            this.updateConnections();
        });
        
        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                node.style.cursor = 'grab';
                node.style.zIndex = '';
            }
        });
    }
    
    selectComponent(componentId) {
        this.deselectAll();
        
        const component = this.components.find(c => c.id === componentId);
        if (component) {
            component.node.classList.add('selected');
            this.selectedComponent = component;
            
            // Update properties panel
            this.updatePropertiesPanel(component);
        }
    }
    
    deselectAll() {
        this.components.forEach(c => c.node.classList.remove('selected'));
        this.selectedComponent = null;
    }
    
    openComponentConfig(componentId) {
        const component = this.components.find(c => c.id === componentId);
        if (!component) return;
        
        // Generate configuration form
        const configContent = document.getElementById('modal-config-content');
        configContent.innerHTML = this.generateConfigForm(component);
        
        // Store current component ID for saving
        configContent.dataset.componentId = componentId;
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('componentConfigModal'));
        modal.show();
    }
    
    generateConfigForm(component) {
        const config = component.config;
        let html = `<h6>${this.getComponentInfo(component.type).name} Configuration</h6>`;
        
        for (const [key, value] of Object.entries(config)) {
            html += `
                <div class="mb-3">
                    <label class="form-label">${this.formatLabel(key)}</label>
            `;
            
            if (typeof value === 'number') {
                html += `
                    <input type="number" class="form-control" 
                           name="${key}" value="${value}" 
                           step="${value < 1 ? '0.0001' : '1'}">
                `;
            } else if (typeof value === 'boolean') {
                html += `
                    <select class="form-select" name="${key}">
                        <option value="true" ${value ? 'selected' : ''}>True</option>
                        <option value="false" ${!value ? 'selected' : ''}>False</option>
                    </select>
                `;
            } else if (Array.isArray(value)) {
                html += `
                    <input type="text" class="form-control" 
                           name="${key}" value="${value.join(', ')}"
                           placeholder="Comma-separated values">
                `;
            } else {
                html += `
                    <input type="text" class="form-control" 
                           name="${key}" value="${value}">
                `;
            }
            
            html += '</div>';
        }
        
        return html;
    }
    
    saveComponentConfig() {
        const configContent = document.getElementById('modal-config-content');
        const componentId = parseInt(configContent.dataset.componentId);
        const component = this.components.find(c => c.id === componentId);
        
        if (!component) return;
        
        // Get form values
        const inputs = configContent.querySelectorAll('input, select');
        inputs.forEach(input => {
            const key = input.name;
            let value = input.value;
            
            // Convert types
            if (input.type === 'number') {
                value = parseFloat(value);
            } else if (input.tagName === 'SELECT') {
                value = value === 'true';
            } else if (value.includes(',')) {
                value = value.split(',').map(v => v.trim());
            }
            
            component.config[key] = value;
        });
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('componentConfigModal'));
        modal.hide();
        
        // Show success message
        this.showAlert('Configuration saved', 'success');
    }
    
    updatePropertiesPanel(component) {
        // This would update a properties panel if implemented
        console.log('Selected component:', component);
    }
    
    updateConnections() {
        // Clear SVG
        this.svg.innerHTML = '';
        
        // Redraw all connections
        this.connections.forEach(conn => {
            this.drawConnection(conn.from, conn.to);
        });
    }
    
    drawConnection(fromId, toId) {
        const fromComponent = this.components.find(c => c.id === fromId);
        const toComponent = this.components.find(c => c.id === toId);
        
        if (!fromComponent || !toComponent) return;
        
        // Calculate connection points
        const fromX = fromComponent.position.x + 200; // Right side
        const fromY = fromComponent.position.y + 50;  // Middle
        const toX = toComponent.position.x;           // Left side
        const toY = toComponent.position.y + 50;      // Middle
        
        // Create curved path
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const d = `M ${fromX} ${fromY} C ${fromX + 50} ${fromY}, ${toX - 50} ${toY}, ${toX} ${toY}`;
        path.setAttribute('d', d);
        path.setAttribute('class', 'connection-line');
        
        this.svg.appendChild(path);
    }
    
    getComponentInfo(type) {
        const components = {
            'feature_extractor': {
                name: 'Feature Extractor',
                description: 'Technical Indicators',
                icon: 'fas fa-filter'
            },
            'transformer': {
                name: 'Transformer Attention',
                description: 'Market Regime Detection',
                icon: 'fas fa-eye'
            },
            'ppo': {
                name: 'PPO Algorithm',
                description: 'Proximal Policy Optimization',
                icon: 'fas fa-brain'
            },
            'dqn': {
                name: 'DQN Algorithm',
                description: 'Deep Q-Network',
                icon: 'fas fa-network-wired'
            },
            'genetic': {
                name: 'Genetic Optimizer',
                description: 'Hyperparameter Tuning',
                icon: 'fas fa-dna'
            },
            'risk_manager': {
                name: 'Risk Manager',
                description: 'Advanced Risk Controls',
                icon: 'fas fa-shield-alt'
            },
            'portfolio_optimizer': {
                name: 'Portfolio Optimizer',
                description: 'Position Sizing',
                icon: 'fas fa-chart-pie'
            },
            'market_detector': {
                name: 'Market Detector',
                description: 'Regime Classification',
                icon: 'fas fa-search'
            }
        };
        
        return components[type] || {
            name: type,
            description: '',
            icon: 'fas fa-cube'
        };
    }
    
    getDefaultConfig(type) {
        const configs = {
            'feature_extractor': {
                indicators: ['sma', 'ema', 'rsi', 'macd'],
                windows: [5, 10, 20, 50],
                normalize: true
            },
            'transformer': {
                heads: 8,
                layers: 4,
                dropout: 0.1,
                hidden_size: 256
            },
            'ppo': {
                learning_rate: 0.0003,
                gamma: 0.99,
                clip_range: 0.2,
                n_steps: 2048
            },
            'dqn': {
                learning_rate: 0.001,
                gamma: 0.95,
                epsilon: 0.1,
                buffer_size: 10000
            },
            'genetic': {
                population_size: 50,
                mutation_rate: 0.1,
                crossover_rate: 0.8,
                generations: 100
            },
            'risk_manager': {
                max_position_size: 10,
                max_drawdown: 0.15,
                stop_loss: 0.02,
                take_profit: 0.05
            },
            'portfolio_optimizer': {
                max_allocation: 0.25,
                min_allocation: 0.05,
                rebalance_frequency: 'daily'
            },
            'market_detector': {
                regimes: ['trending', 'ranging', 'volatile'],
                lookback_period: 100,
                confidence_threshold: 0.7
            }
        };
        
        return configs[type] || {};
    }
    
    formatLabel(key) {
        return key.replace(/_/g, ' ')
                  .replace(/\b\w/g, l => l.toUpperCase());
    }
    
    clearCanvas() {
        // Remove all component nodes
        this.components.forEach(c => c.node.remove());
        
        // Clear arrays
        this.components = [];
        this.connections = [];
        this.nextComponentId = 0;
        
        // Clear SVG
        this.svg.innerHTML = '';
        
        // Show placeholder
        const placeholder = document.getElementById('canvas-placeholder');
        if (placeholder) {
            placeholder.style.display = 'block';
        }
    }
    
    exportStrategy() {
        return {
            name: 'Custom Strategy',
            components: this.components.map(c => ({
                id: c.id,
                type: c.type,
                position: c.position,
                config: c.config
            })),
            connections: this.connections
        };
    }
    
    loadTemplate(template) {
        this.clearCanvas();
        
        // Add components
        template.components.forEach(comp => {
            this.addComponent(comp.type, comp.position.x + 100, comp.position.y + 50);
        });
        
        // Add connections
        template.connections.forEach(conn => {
            this.connections.push(conn);
        });
        
        // Update connections display
        this.updateConnections();
    }
    
    showAlert(message, type) {
        // You can implement a custom alert system here
        console.log(`${type}: ${message}`);
    }
}

// Initialize strategy builder when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.strategyBuilder = new StrategyBuilder();
});

// Global functions called from HTML
function loadTemplate(templateName) {
    const templates = {
        'ane_ppo': {
            name: 'ANE-PPO Strategy',
            components: [
                {type: 'feature_extractor', position: {x: 50, y: 50}},
                {type: 'transformer', position: {x: 200, y: 50}},
                {type: 'ppo', position: {x: 350, y: 50}},
                {type: 'genetic', position: {x: 200, y: 150}},
                {type: 'risk_manager', position: {x: 350, y: 150}}
            ],
            connections: [
                {from: 0, to: 1},
                {from: 1, to: 2},
                {from: 2, to: 3},
                {from: 3, to: 4}
            ]
        },
        'conservative': {
            name: 'Conservative Strategy',
            components: [
                {type: 'feature_extractor', position: {x: 50, y: 50}},
                {type: 'risk_manager', position: {x: 200, y: 50}},
                {type: 'ppo', position: {x: 350, y: 50}}
            ],
            connections: [
                {from: 0, to: 1},
                {from: 1, to: 2}
            ]
        },
        'aggressive': {
            name: 'Aggressive Strategy',
            components: [
                {type: 'feature_extractor', position: {x: 50, y: 50}},
                {type: 'transformer', position: {x: 200, y: 50}},
                {type: 'dqn', position: {x: 350, y: 50}},
                {type: 'genetic', position: {x: 200, y: 150}}
            ],
            connections: [
                {from: 0, to: 1},
                {from: 1, to: 2},
                {from: 2, to: 3}
            ]
        }
    };
    
    const template = templates[templateName];
    if (template) {
        const strategyBuilder = window.strategyBuilder || new StrategyBuilder();
        strategyBuilder.loadTemplate(template);
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('templatesModal'));
        modal.hide();
        
        showAlert(`Loaded ${template.name} template`, 'success');
    }
}

function saveStrategy() {
    const strategyBuilder = window.strategyBuilder || new StrategyBuilder();
    const strategy = strategyBuilder.exportStrategy();
    
    // Save to localStorage for now
    localStorage.setItem('saved_strategy', JSON.stringify(strategy));
    showAlert('Strategy saved successfully!', 'success');
}

function clearCanvas() {
    if (confirm('Are you sure you want to clear the canvas?')) {
        const strategyBuilder = window.strategyBuilder || new StrategyBuilder();
        strategyBuilder.clearCanvas();
        showAlert('Canvas cleared', 'info');
    }
}

function exportStrategy() {
    const strategyBuilder = window.strategyBuilder || new StrategyBuilder();
    const strategy = strategyBuilder.exportStrategy();
    
    // Create download link
    const blob = new Blob([JSON.stringify(strategy, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `strategy_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    showAlert('Strategy exported successfully!', 'success');
}

function showAlert(message, type) {
    // Simple alert implementation
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed top-0 end-0 m-3`;
    alertDiv.style.zIndex = '9999';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}