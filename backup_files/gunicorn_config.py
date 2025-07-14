"""
Gunicorn configuration for the trading system
"""

import multiprocessing
import os

# Bind to 0.0.0.0:5000
bind = "0.0.0.0:5000"

# Worker configuration
workers = 1  # Single worker to avoid memory issues
worker_class = "eventlet"  # Better for WebSocket connections
worker_connections = 1000
timeout = 600  # 10 minutes for WebSocket connections
keepalive = 5
graceful_timeout = 600  # Allow graceful shutdown

# Reload on code changes
reload = True

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Preload app to save memory
preload_app = False  # Keep False for development with reload

# Worker restart settings
max_requests = 100
max_requests_jitter = 10

# Socket settings
reuse_port = True