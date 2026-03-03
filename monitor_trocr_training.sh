#!/bin/bash
# Monitor TrOCR training progress on A100

echo "🚀 TrOCR Training Monitor"
echo "========================"
echo ""

# Check if training is running
echo "📊 Process Status:"
ssh root@95.216.229.232 "ps aux | grep python3 | grep trocr | grep -v grep || echo '   No active process yet (starting...)'"

echo ""
echo "📋 Training Log (Last 20 lines):"
ssh root@95.216.229.232 "tail -20 /root/odia_ocr/trocr_training.log" 2>/dev/null || echo "   Log file not yet created"

echo ""
echo "💾 GPU Status:"
ssh root@95.216.229.232 "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader 2>/dev/null || echo '   GPU info unavailable'"

echo ""
echo "🔄 Directory Contents:"
ssh root@95.216.229.232 "ls -lh /root/odia_ocr/trocr-* 2>/dev/null | head -10 || echo '   No checkpoints yet'"

echo ""
echo "✅ Monitor complete. Re-run this script to check progress."
