#!/usr/bin/env python3
"""
Real-time training monitor - updates every 1 minute
Simplified version with proper SSH command handling
"""

import subprocess
import time
import re
from datetime import datetime
from collections import deque

REMOTE_USER = "root"
REMOTE_HOST = "95.216.229.232"
LOG_FILE = "/root/odia_ocr/training_fixed.log"
UPDATE_INTERVAL = 60

loss_history = deque(maxlen=20)


def run_ssh(cmd):
    """Execute SSH command safely"""
    try:
        result = subprocess.run(
            ["ssh", f"{REMOTE_USER}@{REMOTE_HOST}", cmd],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {str(e)[:50]}"


def get_status():
    """Fetch all status metrics"""
    # Get last line for progress
    progress = run_ssh(f"tail -1 {LOG_FILE}")
    
    # Get GPU status
    gpu = run_ssh("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits")
    
    # Get process count
    procs = run_ssh("ps aux | grep train_fixed | grep -v grep | wc -l")
    
    # Get any loss values from recent lines
    losses = run_ssh(f"grep 'loss = ' {LOG_FILE} | tail -3")
    
    return {'progress': progress, 'gpu': gpu, 'procs': procs, 'losses': losses}


def parse_map_line(line):
    """Extract progress from Map line"""
    match = re.search(r'(\d+)%.*?(\d+)/(\d+).*?\[([\d:]+)<([\d:]+).*?([0-9.]+)\s*examples/s', line)
    if match:
        pct, curr, total, elapsed, remaining, speed = match.groups()
        return {
            'percent': int(pct),
            'current': int(curr),
            'total': int(total),
            'elapsed': elapsed,
            'remaining': remaining,
            'speed': float(speed)
        }
    return None


def parse_loss_line(line):
    """Extract loss value"""
    match = re.search(r'\[(\d+)/(\d+)\].*?loss = ([0-9.]+)', line)
    if match:
        step, total_steps, loss = match.groups()
        return {'step': int(step), 'total': int(total_steps), 'loss': float(loss)}
    return None


def display_status(data, iteration):
    """Display formatted status with 1-minute intervals"""
    os_clear = "\033[2J\033[H"
    print(f"{os_clear}", end="")
    
    print("=" * 70)
    print(f"⏱️  ODIA OCR TRAINING MONITOR - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)
    
    # Parse progress
    prog_data = parse_map_line(data['progress'])
    
    if prog_data:
        # Show progress bar
        bar_width = 25
        filled = int(bar_width * prog_data['percent'] / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        print(f"\n📊 DATA PREPROCESSING")
        print(f"   [{bar}] {prog_data['percent']:3d}%")
        print(f"   Progress: {prog_data['current']:,} / {prog_data['total']:,} examples")
        print(f"   Speed: {prog_data['speed']:.1f} examples/sec")
        print(f"   Time: {prog_data['elapsed']} | ETA: {prog_data['remaining']}")
    elif '[' in data['progress'] and '/' in data['progress']:
        print(f"\n🎯 TRAINING IN PROGRESS")
        print(f"   {data['progress'][:80]}")
    else:
        print(f"\n⏳ Loading/Initializing...")
        print(f"   {data['progress'][:80]}")
    
    # Show losses if available
    if data['losses'] and 'Error' not in data['losses']:
        print(f"\n📈 LOSS VALUES")
        for loss_line in data['losses'].split('\n')[-3:]:
            loss_data = parse_loss_line(loss_line)
            if loss_data:
                print(f"   Step [{loss_data['step']:4d}/{loss_data['total']}]: loss={loss_data['loss']:.6f}")
                loss_history.append(loss_data['loss'])
        
        if len(loss_history) > 1:
            trend = "📉" if loss_history[-1] < loss_history[0] else "📈"
            change = loss_history[-1] - loss_history[0]
            print(f"   Trend: {trend} ({change:+.4f})")
    
    # GPU status
    if data['gpu'] and 'Error' not in data['gpu']:
        try:
            parts = data['gpu'].split(',')
            mem_used = parts[0].strip()
            mem_total = parts[1].strip()
            util = parts[2].strip()
            print(f"\n💾 GPU STATUS")
            print(f"   Memory: {mem_used}MB / {mem_total}MB")
            print(f"   Utilization: {util}%")
        except:
            pass
    
    # Process status
    try:
        num_procs = int(data['procs'].strip())
        print(f"\n⚙️  SYSTEM")
        print(f"   Processes: {num_procs}")
        print(f"   Update: #{iteration}")
    except:
        pass
    
    print("\n" + "=" * 70)
    print("🔄 Updates every 1 minute (Ctrl+C to stop)")
    print("=" * 70)


def main():
    print("🚀 Starting real-time training monitor...")
    time.sleep(2)
    
    iteration = 0
    try:
        while True:
            iteration += 1
            data = get_status()
            display_status(data, iteration)
            
            # Sleep for 1 minute
            for _ in range(UPDATE_INTERVAL):
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n\n✋ Monitor stopped")
        if len(loss_history) > 1:
            print(f"\nSession Summary:")
            print(f"  Updates: {iteration}")
            print(f"  Loss samples: {len(loss_history)}")
            initial = loss_history[0]
            final = loss_history[-1]
            improvement = ((initial - final) / initial * 100) if initial > 0 else 0
            print(f"  Loss: {initial:.4f} → {final:.4f} ({improvement:+.1f}%)")


if __name__ == "__main__":
    main()
