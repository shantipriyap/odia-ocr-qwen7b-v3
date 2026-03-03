#!/bin/bash

# Quick training status checker
# Usage: bash check_training_status.sh

SERVER="root@95.216.229.232"

echo "╔════════════════════════════════════════════════╗"
echo "║   Odia OCR Training Status Check              ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

echo "📊 Training Process Status:"
ssh $SERVER 'ps aux | grep train_fixed | grep -v grep | awk "{print \"   PID:\", \$2, \"| CPU:\", \$3\"%\", \"| Mem:\", int(\$6/1024)\"MB\"}" | head -1'

echo ""
echo "📈 Latest Progress:"
ssh $SERVER 'tail -20 /root/odia_ocr/training_fixed.log 2>/dev/null | grep -E "Map|Filter" | tail -1 || echo "   Loading..."'

echo ""
echo "💾 Log Statistics:"
ssh $SERVER 'du -h /root/odia_ocr/training_fixed.log 2>/dev/null | awk "{print \"   Size:\", \$1}" && wc -l /root/odia_ocr/training_fixed.log 2>/dev/null | awk "{print \"   Lines:\", \$1}"'

echo ""
echo "⏰ Time Elapsed:"
ssh $SERVER 'stat -f "%Sm" -t "%H:%M:%S" /root/odia_ocr/training_fixed.log 2>/dev/null || echo "   Calculating..." && echo "   Current time: $(date -u +'%H:%M:%S UTC')"'

echo ""
echo "🚀 Next Steps:"
echo "   • Wait for data preprocessing: Map → Filter → Training"
echo "   • Once training starts, loss metrics will appear"
echo "   • Estimated completion: ~3-3.5 hours from start"
echo ""
echo "📝 For detailed logs:"
echo "   ssh root@95.216.229.232 'tail -f /root/odia_ocr/training_fixed.log'"
echo ""
