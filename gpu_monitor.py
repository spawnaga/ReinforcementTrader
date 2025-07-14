#!/usr/bin/env python3
"""
Enhanced GPU monitoring for AI Trading System
Shows detailed GPU information and usage
"""

import time
import subprocess
import requests
import socketio
from datetime import datetime
import argparse
import sys

class GPUMonitor:
    def __init__(self, api_url):
        self.api_url = api_url.rstrip('/')
        self.sio = socketio.Client()
        self.setup_handlers()
        
    def setup_handlers(self):
        """Set up WebSocket event handlers"""
        
        @self.sio.on('connect')
        def on_connect():
            print(f"✅ Connected to {self.api_url}")
            
        @self.sio.on('performance_metrics')
        def on_performance(data):
            # We'll handle this differently
            pass
            
        @self.sio.on('trade_update')
        def on_trade(data):
            print(f"\n💰 New Trade [{datetime.now().strftime('%H:%M:%S')}]")
            print(f"   Type: {data.get('position_type', 'N/A')}")
            print(f"   Entry: ${data.get('entry_price', 0):.2f}")
            if data.get('exit_price'):
                print(f"   Exit: ${data.get('exit_price', 0):.2f}")
                print(f"   P&L: ${data.get('profit_loss', 0):.2f}")
                
        @self.sio.on('session_update')
        def on_session(data):
            episode = data.get('episode', 0)
            total = data.get('total_episodes', 0)
            if episode % 10 == 0:  # Show every 10th episode
                print(f"\n🎯 Training Progress: Episode {episode}/{total}")
                
    def get_gpu_details(self):
        """Get detailed GPU information using nvidia-smi"""
        try:
            # Get detailed GPU info
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 8:
                        gpus.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_used': float(parts[2]),
                            'memory_total': float(parts[3]),
                            'gpu_util': float(parts[4]),
                            'temperature': float(parts[5]),
                            'power_draw': float(parts[6]),
                            'power_limit': float(parts[7])
                        })
                return gpus
        except Exception as e:
            print(f"Error getting GPU details: {e}")
        return []
        
    def get_process_gpu_usage(self):
        """Get GPU usage by process"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-compute-apps=pid,name,gpu_memory_usage',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                processes = []
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        processes.append({
                            'pid': int(parts[0]),
                            'name': parts[1],
                            'memory_mb': float(parts[2])
                        })
                return processes
        except:
            pass
        return []
        
    def get_system_metrics(self):
        """Get system metrics from API"""
        try:
            # Try to get CPU/Memory from psutil
            import psutil
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory().percent
            
            # Network rate calculation
            net1 = psutil.net_io_counters()
            time.sleep(1)
            net2 = psutil.net_io_counters()
            
            bytes_diff = (net2.bytes_sent + net2.bytes_recv) - (net1.bytes_sent + net1.bytes_recv)
            network_rate = bytes_diff / (1024 * 1024)  # MB/s
            
            return {
                'cpu': cpu,
                'memory': mem,
                'network': network_rate
            }
        except:
            return {'cpu': 0, 'memory': 0, 'network': 0}
            
    def monitor(self):
        """Start monitoring with detailed GPU information"""
        print("=" * 80)
        print("AI Trading System GPU Monitor")
        print("=" * 80)
        
        # Check health
        try:
            health = requests.get(f"{self.api_url}/health").json()
            print(f"✅ API Status: {health['status']}")
        except:
            print("❌ API is not responding")
            return
            
        # Connect WebSocket
        try:
            self.sio.connect(self.api_url)
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            # Continue anyway to show GPU info
            
        print("\n📡 Monitoring active... Press Ctrl+C to stop")
        print("-" * 80)
        
        try:
            while True:
                # Clear screen for refresh (optional)
                # print("\033[H\033[J", end='')
                
                print(f"\n📊 System Status [{datetime.now().strftime('%H:%M:%S')}]")
                print("=" * 80)
                
                # Get system metrics
                sys_metrics = self.get_system_metrics()
                print(f"CPU: {sys_metrics['cpu']:.1f}% | Memory: {sys_metrics['memory']:.1f}% | Network: {sys_metrics['network']:.2f} MB/s")
                
                # Get GPU details
                gpus = self.get_gpu_details()
                if gpus:
                    print(f"\n🎮 GPU Status ({len(gpus)} devices detected):")
                    print("-" * 80)
                    
                    total_util = 0
                    total_memory_used = 0
                    total_memory_total = 0
                    
                    for gpu in gpus:
                        print(f"GPU {gpu['index']}: {gpu['name']}")
                        print(f"   Utilization: {gpu['gpu_util']:.1f}%")
                        print(f"   Memory: {gpu['memory_used']:.0f}MB / {gpu['memory_total']:.0f}MB ({gpu['memory_used']/gpu['memory_total']*100:.1f}%)")
                        print(f"   Temperature: {gpu['temperature']:.0f}°C")
                        print(f"   Power: {gpu['power_draw']:.1f}W / {gpu['power_limit']:.1f}W")
                        
                        total_util += gpu['gpu_util']
                        total_memory_used += gpu['memory_used']
                        total_memory_total += gpu['memory_total']
                        
                    # Average stats
                    avg_util = total_util / len(gpus)
                    print(f"\n📈 Average GPU Utilization: {avg_util:.1f}%")
                    print(f"📈 Total GPU Memory: {total_memory_used:.0f}MB / {total_memory_total:.0f}MB ({total_memory_used/total_memory_total*100:.1f}%)")
                else:
                    print("\n❌ No GPUs detected")
                    
                # Show GPU processes
                processes = self.get_process_gpu_usage()
                if processes:
                    print(f"\n🔧 GPU Processes:")
                    for proc in processes[:5]:  # Show top 5
                        print(f"   PID {proc['pid']}: {proc['name']} - {proc['memory_mb']:.0f}MB")
                        
                # Get active sessions from API
                try:
                    response = requests.get(f"{self.api_url}/api/sessions?status=running")
                    if response.ok:
                        sessions = response.json()
                        if sessions:
                            print(f"\n🏃 Active Training Sessions: {len(sessions)}")
                            for session in sessions[:3]:
                                print(f"   - Session {session.get('id')}: Episode {session.get('current_episode', 0)}/{session.get('total_episodes', 0)}")
                except:
                    pass
                    
                print("\n" + "-" * 80)
                time.sleep(5)  # Update every 5 seconds
                
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")
            if self.sio.connected:
                self.sio.disconnect()

def main():
    parser = argparse.ArgumentParser(description='GPU Monitor for AI Trading System')
    parser.add_argument('--url', default='http://localhost:5000', 
                        help='API URL (default: http://localhost:5000)')
    args = parser.parse_args()
    
    monitor = GPUMonitor(args.url)
    monitor.monitor()

if __name__ == '__main__':
    main()