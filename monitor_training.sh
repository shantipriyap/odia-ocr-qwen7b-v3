#!/bin/bash
# Monitor Qwen training progress

echo "🔍 QWEN2.5-VL TRAINING MONITOR"
echo "======================================================"

while true; do
    echo ""
    echo "📊 Status ($(date +%H:%M:%S)):"
    echo "---"
    
    # Check if still running
    PROCS=$(ssh -o ConnectTimeout=5 root@95.216.229.232 "ps aux | grep qwen_full_training.py | grep -v grep | wc -l" 2>/dev/null || echo "0")
    
    if [ "$PROCS" -eq 0 ]; then
        echo "❌ Training stopped!"
        echo ""
        echo "Final results:"
        ssh -o ConnectTimeout=5 root@95.216.229.232 "tail -50 /root/odia_ocr/qwen_full_training.log 2>&1"
        echo ""
        echo "✅ Training complete!"
        break
    fi
    
    # Get last log line
    LAST_LINE=$(ssh -o ConnectTimeout=5 root@95.216.229.232 "tail -1 /root/odia_ocr/qwen_full_training.log 2>&1" 2>/dev/null)
    echo "Last: $LAST_LINE"
    
    # GPU status
    echo ""
    echo "💻 GPU Status:"
    ssh -o ConnectTimeout=5 root@95.216.229.232 "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader" 2>/dev/null || echo "GPU check failed"
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 30
done
