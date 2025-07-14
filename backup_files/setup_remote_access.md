# Remote Monitoring Setup Guide

This guide explains how to monitor your trading system from a Windows machine (or any other computer) while it's running on a remote Linux/Ubuntu server.

## Prerequisites

- Python installed on your Windows machine
- The `requests` library: `pip install requests`
- Network access to your remote machine

## Method 1: Direct Remote Monitoring (Simplest)

### On the Remote Machine (where training runs):

1. Start your trading system with network binding:
```bash
python run_local.py
```

2. Make sure the Flask app is accessible:
   - Check firewall: `sudo ufw allow 5000`
   - Or use a specific IP: Edit `run_local.py` to bind to `0.0.0.0` instead of `localhost`

### On Your Windows Machine:

1. Save the `remote_monitor.py` file
2. Install requests: `pip install requests`
3. Run the monitor:
```bash
# If on same network
python remote_monitor.py 192.168.1.100 5000

# Replace 192.168.1.100 with your remote machine's IP
```

## Method 2: SSH Port Forwarding (Most Secure)

### On Your Windows Machine:

1. Open PowerShell or Command Prompt
2. Create SSH tunnel:
```bash
ssh -L 5000:localhost:5000 username@remote-server-ip
```

3. In another terminal, run:
```bash
python remote_monitor.py localhost 5000
```

## Method 3: Web Dashboard Access

Since the system has a web interface, you can also:

1. On remote machine, edit `run_local.py`:
```python
# Change from:
app.run(debug=True, host='127.0.0.1', port=5000)

# To:
app.run(debug=True, host='0.0.0.0', port=5000)
```

2. Access from Windows browser:
```
http://remote-machine-ip:5000
```

## Method 4: Real-time WebSocket Monitor

For real-time updates, create a WebSocket client:

```python
# Save as websocket_monitor.py
import socketio

sio = socketio.Client()

@sio.on('training_update')
def on_training_update(data):
    print(f"Training Update: {data}")

@sio.on('trade_update')
def on_trade_update(data):
    print(f"New Trade: {data}")

sio.connect('http://remote-ip:5000')
sio.wait()
```

## Security Considerations

1. **For Production**: Use a reverse proxy (nginx) with SSL
2. **Firewall**: Only allow specific IPs to access port 5000
3. **Authentication**: Add API key authentication to routes.py
4. **VPN**: Consider using a VPN for secure access

## Troubleshooting

### Can't Connect?
- Check if Flask is running: `ps aux | grep python`
- Check firewall: `sudo ufw status`
- Test locally first: `curl http://localhost:5000/health`
- Check binding: Flask must bind to `0.0.0.0` not `127.0.0.1`

### Windows Firewall Issues?
- Allow Python through Windows Defender Firewall
- Run as Administrator if needed

### Network Issues?
- Ensure both machines are on same network or have internet access
- Check if port 5000 is open on remote machine
- Try ping first: `ping remote-machine-ip`