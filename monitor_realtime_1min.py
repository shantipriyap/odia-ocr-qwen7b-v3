#!/usr/bin/env python3
"""
Real-time training monitor - updates every 1 minute
Shows: Map/Filter/training progress, loss, GPU status, ETA
"""

import subprocess
import time
import re
from datetime import datetime, timedelta
from collections import deque

# Configuration
REMOTE_USER = "root"
REMOTE_HOST = "95.216.229.232"
LOG_FILE = "/root/odia_ocr/training_fixed.log"
UPDATE_INTERVAL = 60  # seconds

# Keep history for trend analysis
loss_history = deque(maxlen=20)
start_time = None


def run_ssh_command(cmd):
    """Execute SSH command and return output"""
    full_cmd = f"ssh {REMOTE_USER}@{REMOTE_HOST} '{cmd}'"
    try:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def get_training_status():
    """Get comprehensive training status"""
    cmd = f"""
    # Get last 100 lines to analyze current state
    tail -100 {LOG_FILE} 2>/dev/null | tee /tmp/last_100.log &&
    
    # Check for latest Map/Filter/Step patterns
    grep -E "Map|Filter|^\\[" {LOG_FILE} 2>/dev/null | tail -5
    """
    
    # Get log content
    log_output = run_ssh_command(f"tail -100 {LOG_FILE} 2>/dev/null")
    
    # Get latest progress pattern
    progress_cmd = f"grep -E 'Map|Filter|^\\[' {LOG_FILE} 2>/dev/null | tail -1"
    latest_progress = run_ssh_command(progress_cmd)
    
    # Get loss values
    loss_cmd = f"grep 'loss = ' {LOG_FILE} 2>/dev/null | tail -5"
    loss_output = run_ssh_command(loss_cmd)
    
    # Get GPU status
    gpu_cmd = "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
    gpu_status = run_ssh_command(gpu_cmd)
    
    # Get process info
    process_cmd = "ps aux | grep train_fixed | grep -v grep | wc -l"
    process_count = run_ssh_command(process_cmd)
    
    return {
        'log': log_output,
        'progress': latest_progress,
        'loss': loss_output,
        'gpu': gpu_status,
        'processes': process_count
    }


def parse_map_progress(progress_line):
    """Parse Map stage progress"""
    match = re.search(r'Map.*?(\d+)%.*?(\d+)/(\d+).*?\[([\d:]+)<([\d:]+).*?([0-9.]+)\s*(examples/s|s/examples)', progress_line)
    if match:
        percent, current, total, elapsed, remaining, speed, unit = match.groups()
        return {
            'type': 'map',
            'percent': int(percent),
            'current': int(current),
            'total': int(total),
            'elapsed': elapsed,
            'remaining': remaining,
            'speed': float(speed),
            'speed_unit': unit
        }
    return None


def parse_loss(loss_line):
    """Parse loss value from training log"""
    match = re.search(r'\[(\d+)/(\d+)\].*?loss = ([0-9.]+)', loss_line)
    if match:
        step, total, loss_val = match.groups()
        return {
            'step': int(step),
            'total': int(total),
            'loss': float(loss_val)
        }
    return None


def format_progress_bar(percent, width=30):
    """Create ASCII progress bar"""
    filled = int(width * percent / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percent:3d}%"


def display_status(status):
    """Display formatted status"""
    print("\033[2J\033[H")  # Clear screen
    print("=" * 80)
    print(f"🔄 ODIA OCR TRAINING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Parse progress
    progress_data = parse_map_progress(status['progress'])
    
    if progress_data:
        print(f"\n📊 DATA PREPROCESSING ({progress_data['type'].upper()})")
        print(f"   {format_progress_bar(progress_data['percent'])}")
        print(f"   Progress: {progress_data['current']:,} / {progress_data['total']:,} examples")
        print(f"   Speed: {progress_data['speed']:.2f} {progress_data['speed_unit']}")
        print(f"   Elapsed: {progress_data['elapsed']} | Remaining: {progress_data['remaining']}")
    else:
        # Check if training has started
        if '[' in status['progress'] and '/' in status['progress']:
            print(f"\n🎯 TRAINING IN PROGRESS")
            print(f"   Latest: {status['progress'][:100]}")
        else:
            print(f"\n⏳ Waiting for training to start...")
            print(f"   Latest: {status['progress'][:100] if status['progress'] else 'No progress yet'}")
    
    # Parse and display loss
    if status['loss']:
        print(f"\n📈 LOSS METRICS")
        loss_lines = status['loss'].split('\n')
        for line in loss_lines[-5:]:
            loss_data = parse_loss(line)
            if loss_data:
                print(f"   Step [{loss_data['step']:3d}/{loss_data['total']}]: loss = {loss_data['loss']:.6f}")
                loss_history.append((loss_data['step'], loss_data['loss']))
        
        # Show trend
        if len(loss_history) > 1:
            first_loss = loss_history[0][1]
            latest_loss = loss_history[-1][1]
            change = latest_loss - first_loss
            trend = "📉 Decreasing" if change < 0 else "📈 Increasing" if change > 0 else "➡️  Stable"
            print(f"   Trend: {trend} ({change:+.6f})")
    
    # GPU Status
    if status['gpu'] and 'Error' not in status['gpu']:
        print(f"\n💾 GPU STATUS")
        try:
            gpu_parts = status['gpu'].split(',')
            mem_used = gpu_parts[0].strip()
            mem_total = gpu_parts[1].strip()
            util = gpu_parts[2].strip()
            print(f"   GPU Memory: {mem_used}MB / {mem_total}MB")
            print(f"   GPU Utilization: {util}%")
        except:
            print(f"   {status['gpu']}")
    
    # Process Status
    print(f"\n⚙️  SYSTEM STATUS")
    print(f"   Active Processes: {status['processes'].strip()}")
    
    print("\n" + "=" * 80)
    print("⏱️  Next update in 1 minute... (Press Ctrl+C to stop)")
    print("=" * 80)


def main():
    """Main monitoring loop"""
    print("🚀 Starting real-time training monitor...")
    print("Fetching initial status...")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            status = get_training_status()
            display_status(status)
            
            # Wait 1 minute before next update
            for remaining in range(UPDATE_INTERVAL, 0, -1):
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n\n✋ Monitor stopped by user")
        print("=" * 80)
        if len(loss_history) > 0:
            print(f"📊 Session Summary:")
            print(f"   Total updates: {iteration}")
            print(f"   Loss samples collected: {len(loss_history)}")
            if len(loss_history) > 1:
                initial_loss = loss_history[0][1]
                final_loss = loss_history[-1][1]
                improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
                print(f"   Initial loss: {initial_loss:.6f}")
                print(f"   Final loss: {final_loss:.6f}")
                print(f"   Improvement: {improvement:.2f}%")
        print("=" * 80)


if __name__ == "__main__":
    main()
