"""
Gunicorn configuration for the trading system
"""

import multiprocessing
import os

# Bind to 0.0.0.0:5000
bind = "0.0.0.0:5000"

# Worker configuration
workers = 1  # Single worker to avoid memory issues
worker_class = "sync"
worker_connections = 1000
timeout = 300  # 5 minutes for processing large datasets
keepalive = 5

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