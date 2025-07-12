// Revolutionary AI Trading System - Main JavaScript

// Global variables
let socket = null;
let isConnected = false;
let currentTheme = 'dark';
let notifications = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    initializeSocket();
    setupTheme();
    createParticleEffect();
    startSystemMonitoring();
});

// Initialize application
function initializeApp() {
    console.log('🚀 Revolutionary AI Trading System - Initializing...');
    
    // Setup global error handler
    window.addEventListener('error', function(event) {
        console.error('Global error:', event.error);
        showAlert('An unexpected error occurred. Please refresh the page.', 'danger');
    });
    
    // Setup unhandled promise rejection handler
    window.addEventListener('unhandledrejection', function(event) {
        console.error('Unhandled promise rejection:', event.reason);
        showAlert('A system error occurred. Please check your connection.', 'warning');
    });
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize modals
    initializeModals();
    
    // Check system requirements
    checkSystemRequirements();
    
    console.log('✅ Application initialized successfully');
}

// Setup event listeners
function setupEventListeners() {
    // Navigation links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            // Add loading state
            if (this.getAttribute('href') && !this.getAttribute('href').startsWith('#')) {
                this.classList.add('loading');
            }
        });
    });
    
    // Global keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl+K for search
        if (e.ctrlKey && e.key === 'k') {
            e.preventDefault();
            openSearchModal();
        }
        
        // Ctrl+Shift+D for debug mode
        if (e.ctrlKey && e.shiftKey && e.key === 'D') {
            e.preventDefault();
            toggleDebugMode();
        }
        
        // Escape to close modals
        if (e.key === 'Escape') {
            closeAllModals();
        }
    });
    
    // Form validation
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!validateForm(this)) {
                e.preventDefault();
                showAlert('Please fill in all required fields correctly.', 'warning');
            }
        });
    });
    
    // Auto-save functionality
    document.querySelectorAll('input, textarea, select').forEach(element => {
        element.addEventListener('change', function() {
            autoSave(this);
        });
    });
}

// Initialize Socket.IO connection
function initializeSocket() {
    try {
        socket = io({
            transports: ['websocket'],
            upgrade: true,
            rememberUpgrade: true,
            reconnection: true,
            reconnectionAttempts: Infinity,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            timeout: 20000
        });
        
        socket.on('connect', function() {
            console.log('🔌 Socket.IO connected');
            isConnected = true;
            updateConnectionStatus(true);
            // Don't show notification on reconnect
            if (!socket.recovered) {
                showNotification('Connected to server', 'success');
            }
        });
        
        socket.on('disconnect', function() {
            console.log('🔌 Socket.IO disconnected');
            isConnected = false;
            updateConnectionStatus(false);
            // Don't show warning, auto-reconnect will handle it
        });
        
        socket.on('error', function(error) {
            console.error('Socket.IO error:', error);
            // Only show error for non-reconnection errors
            if (error.type !== 'TransportError') {
                showAlert('Connection error: ' + error, 'danger');
            }
        });
        
        // Training updates
        socket.on('training_update', function(data) {
            handleTrainingUpdate(data);
        });
        
        // Market data updates
        socket.on('market_data', function(data) {
            handleMarketDataUpdate(data);
        });
        
        // System notifications
        socket.on('system_notification', function(data) {
            showNotification(data.message, data.type);
        });
        
        // Performance metrics
        socket.on('performance_metrics', function(data) {
            updatePerformanceMetrics(data);
        });
        
    } catch (error) {
        console.error('Failed to initialize Socket.IO:', error);
        showAlert('Failed to connect to server', 'danger');
    }
}

// Setup theme
function setupTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);
    
    // Theme toggle button
    const themeToggle = document.createElement('button');
    themeToggle.className = 'btn btn-outline-light btn-sm position-fixed';
    themeToggle.style.top = '10px';
    themeToggle.style.right = '10px';
    themeToggle.style.zIndex = '9999';
    themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    themeToggle.addEventListener('click', toggleTheme);
    document.body.appendChild(themeToggle);
}

// Create particle effect
function createParticleEffect() {
    const particlesContainer = document.createElement('div');
    particlesContainer.className = 'particles';
    document.body.appendChild(particlesContainer);
    
    function createParticle() {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + 'vw';
        particle.style.animationDelay = Math.random() * 10 + 's';
        particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
        particlesContainer.appendChild(particle);
        
        // Remove particle after animation
        particle.addEventListener('animationend', function() {
            this.remove();
        });
    }
    
    // Create particles periodically
    setInterval(createParticle, 1000);
}

// Start system monitoring
function startSystemMonitoring() {
    setInterval(function() {
        if (isConnected) {
            socket.emit('request_system_status');
        }
    }, 5000);
}

// Utility functions
function showAlert(message, type = 'info', duration = 5000) {
    const alertContainer = document.getElementById('alert-container');
    if (!alertContainer) return;
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        <strong>${type.charAt(0).toUpperCase() + type.slice(1)}:</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    alertContainer.appendChild(alert);
    
    // Auto-dismiss alert
    setTimeout(function() {
        if (alert.parentNode) {
            alert.remove();
        }
    }, duration);
}

function showNotification(message, type = 'info') {
    const notification = {
        id: Date.now(),
        message: message,
        type: type,
        timestamp: new Date()
    };
    
    notifications.push(notification);
    
    // Create notification element
    const notificationEl = document.createElement('div');
    notificationEl.className = `notification notification-${type}`;
    notificationEl.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
    `;
    
    // Add to page
    document.body.appendChild(notificationEl);
    
    // Animate in
    setTimeout(() => {
        notificationEl.classList.add('show');
    }, 100);
    
    // Remove after delay
    setTimeout(() => {
        notificationEl.classList.remove('show');
        setTimeout(() => {
            notificationEl.remove();
        }, 300);
    }, 3000);
}

function getNotificationIcon(type) {
    const icons = {
        success: 'check-circle',
        warning: 'exclamation-triangle',
        danger: 'times-circle',
        info: 'info-circle'
    };
    return icons[type] || 'info-circle';
}

function updateConnectionStatus(connected) {
    const statusElements = document.querySelectorAll('.connection-status');
    statusElements.forEach(element => {
        element.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
        element.innerHTML = connected ? 
            '<i class="fas fa-wifi"></i> Connected' : 
            '<i class="fas fa-wifi-slash"></i> Disconnected';
    });
}

function handleTrainingUpdate(data) {
    console.log('Training update received:', data);
    
    // Update progress bars
    const progressBars = document.querySelectorAll(`[data-session-id="${data.session_id}"] .progress-bar`);
    progressBars.forEach(bar => {
        const progress = (data.episode / data.total_episodes) * 100;
        bar.style.width = progress + '%';
        bar.setAttribute('aria-valuenow', progress);
    });
    
    // Update metrics
    if (data.metrics) {
        updateTrainingMetrics(data.session_id, data.metrics);
    }
    
    // Show notification for significant events
    if (data.episode % 100 === 0) {
        showNotification(`Training session ${data.session_id} completed ${data.episode} episodes`, 'info');
    }
}

function handleMarketDataUpdate(data) {
    console.log('Market data update received:', data);
    
    // Update price displays
    const priceElements = document.querySelectorAll('.current-price');
    priceElements.forEach(element => {
        element.textContent = data.price.toFixed(2);
        element.className = `current-price ${data.change > 0 ? 'price-up' : 'price-down'}`;
    });
    
    // Update charts if available
    if (window.tradingCharts) {
        window.tradingCharts.updateRealTimeData(data);
    }
}

function updatePerformanceMetrics(data) {
    console.log('📊 Performance metrics received:', data);
    
    // Update GPU progress
    if (data.gpu_usage !== undefined) {
        const gpuProgress = document.getElementById('gpu-progress');
        const gpuText = document.getElementById('gpu-text');
        if (gpuProgress && gpuText) {
            gpuProgress.style.width = data.gpu_usage + '%';
            gpuText.textContent = data.gpu_usage.toFixed(1) + '% GPU';
        }
    }
    
    // Update Memory progress
    if (data.memory_usage !== undefined) {
        const memProgress = document.getElementById('memory-progress');
        const memText = document.getElementById('memory-text');
        if (memProgress && memText) {
            memProgress.style.width = data.memory_usage + '%';
            memText.textContent = data.memory_usage.toFixed(1) + '% Memory';
        }
    }
    
    // Update Network I/O
    if (data.network_io !== undefined) {
        const netProgress = document.getElementById('network-progress');
        const netText = document.getElementById('network-text');
        if (netProgress && netText) {
            const networkPercent = Math.min(data.network_io / 10, 100);
            netProgress.style.width = networkPercent + '%';
            netText.textContent = data.network_io.toFixed(2) + ' MB/s';
        }
    }
    
    // Update CPU usage
    if (data.cpu_usage !== undefined) {
        const cpuProgress = document.getElementById('speed-progress');
        const cpuText = document.getElementById('speed-text');
        if (cpuProgress && cpuText) {
            cpuProgress.style.width = data.cpu_usage + '%';
            cpuText.textContent = 'CPU: ' + data.cpu_usage.toFixed(1) + '%';
        }
    }
}

function updateTrainingMetrics(sessionId, metrics) {
    const metricsContainer = document.querySelector(`[data-session-id="${sessionId}"] .metrics`);
    if (!metricsContainer) return;
    
    const metricsHtml = `
        <div class="metric">
            <span class="metric-label">Reward:</span>
            <span class="metric-value">${metrics.reward.toFixed(2)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Loss:</span>
            <span class="metric-value">${metrics.loss.toFixed(4)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Epsilon:</span>
            <span class="metric-value">${metrics.epsilon.toFixed(3)}</span>
        </div>
    `;
    
    metricsContainer.innerHTML = metricsHtml;
}

function validateForm(form) {
    let isValid = true;
    const requiredFields = form.querySelectorAll('[required]');
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            field.classList.add('is-invalid');
            isValid = false;
        } else {
            field.classList.remove('is-invalid');
        }
    });
    
    return isValid;
}

function autoSave(element) {
    if (element.dataset.autoSave === 'true') {
        const key = 'autosave_' + element.id;
        localStorage.setItem(key, element.value);
        
        // Show auto-save indicator
        const indicator = document.createElement('small');
        indicator.className = 'text-muted';
        indicator.textContent = 'Auto-saved';
        element.parentNode.appendChild(indicator);
        
        setTimeout(() => {
            indicator.remove();
        }, 2000);
    }
}

function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

function initializeModals() {
    const modalElements = document.querySelectorAll('.modal');
    modalElements.forEach(modalEl => {
        modalEl.addEventListener('show.bs.modal', function() {
            this.classList.add('modal-loading');
        });
        
        modalEl.addEventListener('shown.bs.modal', function() {
            this.classList.remove('modal-loading');
        });
    });
}

function checkSystemRequirements() {
    const requirements = {
        webgl: checkWebGLSupport(),
        websocket: checkWebSocketSupport(),
        localStorage: checkLocalStorageSupport(),
        canvas: checkCanvasSupport()
    };
    
    const failed = Object.entries(requirements)
        .filter(([key, supported]) => !supported)
        .map(([key]) => key);
    
    if (failed.length > 0) {
        showAlert(`Your browser doesn't support: ${failed.join(', ')}. Some features may not work correctly.`, 'warning');
    }
}

function checkWebGLSupport() {
    try {
        const canvas = document.createElement('canvas');
        return !!(window.WebGLRenderingContext && canvas.getContext('webgl'));
    } catch (e) {
        return false;
    }
}

function checkWebSocketSupport() {
    return 'WebSocket' in window;
}

function checkLocalStorageSupport() {
    try {
        localStorage.setItem('test', 'test');
        localStorage.removeItem('test');
        return true;
    } catch (e) {
        return false;
    }
}

function checkCanvasSupport() {
    const canvas = document.createElement('canvas');
    return !!(canvas.getContext && canvas.getContext('2d'));
}

function setTheme(theme) {
    currentTheme = theme;
    document.body.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
}

function toggleTheme() {
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    
    const themeToggle = document.querySelector('.theme-toggle');
    if (themeToggle) {
        themeToggle.innerHTML = newTheme === 'dark' ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
    }
}

function openSearchModal() {
    // Create search modal if it doesn't exist
    let searchModal = document.getElementById('searchModal');
    if (!searchModal) {
        searchModal = createSearchModal();
        document.body.appendChild(searchModal);
    }
    
    const modal = new bootstrap.Modal(searchModal);
    modal.show();
}

function createSearchModal() {
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.id = 'searchModal';
    modal.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Search</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <input type="text" class="form-control" placeholder="Search..." id="searchInput">
                    <div id="searchResults" class="mt-3"></div>
                </div>
            </div>
        </div>
    `;
    
    // Add search functionality
    const searchInput = modal.querySelector('#searchInput');
    searchInput.addEventListener('input', function() {
        performSearch(this.value);
    });
    
    return modal;
}

function performSearch(query) {
    if (query.length < 2) return;
    
    const searchResults = document.getElementById('searchResults');
    searchResults.innerHTML = '<div class="spinner"></div>';
    
    // Simulate search
    setTimeout(() => {
        const results = [
            { title: 'Trading Dashboard', url: '/trading_dashboard' },
            { title: 'Strategy Builder', url: '/strategy_builder' },
            { title: 'Settings', url: '/settings' }
        ].filter(item => item.title.toLowerCase().includes(query.toLowerCase()));
        
        searchResults.innerHTML = results.map(result => `
            <div class="search-result">
                <a href="${result.url}" class="text-decoration-none">
                    <h6>${result.title}</h6>
                </a>
            </div>
        `).join('');
    }, 300);
}

function toggleDebugMode() {
    const debugMode = document.body.classList.toggle('debug-mode');
    
    if (debugMode) {
        console.log('🐛 Debug mode enabled');
        showNotification('Debug mode enabled', 'info');
        
        // Add debug panel
        const debugPanel = document.createElement('div');
        debugPanel.id = 'debug-panel';
        debugPanel.className = 'position-fixed bottom-0 end-0 p-3 bg-dark text-white';
        debugPanel.style.zIndex = '9999';
        debugPanel.innerHTML = `
            <h6>Debug Panel</h6>
            <div>Connected: ${isConnected}</div>
            <div>Theme: ${currentTheme}</div>
            <div>Notifications: ${notifications.length}</div>
            <button class="btn btn-sm btn-outline-light mt-2" onclick="console.log('Debug info:', {isConnected, currentTheme, notifications})">
                Log Info
            </button>
        `;
        document.body.appendChild(debugPanel);
    } else {
        console.log('🐛 Debug mode disabled');
        showNotification('Debug mode disabled', 'info');
        
        const debugPanel = document.getElementById('debug-panel');
        if (debugPanel) {
            debugPanel.remove();
        }
    }
}

function closeAllModals() {
    const modals = document.querySelectorAll('.modal.show');
    modals.forEach(modal => {
        const modalInstance = bootstrap.Modal.getInstance(modal);
        if (modalInstance) {
            modalInstance.hide();
        }
    });
}

// Export functions for global use
window.showAlert = showAlert;
window.showNotification = showNotification;
window.toggleTheme = toggleTheme;
window.toggleDebugMode = toggleDebugMode;

// Performance monitoring
window.addEventListener('load', function() {
    // Log performance metrics
    if (performance.navigation) {
        console.log('⚡ Page load performance:', {
            loadTime: performance.timing.loadEventEnd - performance.timing.navigationStart,
            domReady: performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart,
            renderTime: performance.timing.loadEventEnd - performance.timing.domContentLoadedEventEnd
        });
    }
});

// Service worker registration
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js')
        .then(registration => {
            console.log('Service Worker registered successfully');
        })
        .catch(error => {
            console.log('Service Worker registration failed:', error);
        });
}

console.log('🚀 Main JavaScript loaded successfully');
