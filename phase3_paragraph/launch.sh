#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# launch.sh — start 2-GPU DDP training inside a tmux session
# Usage: bash launch.sh [HF_TOKEN]
# ─────────────────────────────────────────────────────────────────────────────

SESSION="odia_phase3"
SCRIPT_DIR="/root/phase3_paragraph"
VENV="/root/venv/bin/activate"
LOG="$SCRIPT_DIR/train.log"

# Accept HF token as arg or env
if [[ -n "$1" ]]; then
    export HF_TOKEN="$1"
fi

if [[ -z "$HF_TOKEN" ]]; then
    echo "[ERROR] HF_TOKEN not set. Usage: bash launch.sh hf_xxxx"
    echo "        or: export HF_TOKEN=hf_xxxx && bash launch.sh"
    exit 1
fi
echo "[INFO] HF_TOKEN is set."

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "[INFO] Starting tmux session: $SESSION"
tmux new-session -d -s "$SESSION" -x 220 -y 50

# Run training inside tmux
tmux send-keys -t "$SESSION" "
source $VENV && \\
cd $SCRIPT_DIR && \\
echo '[START] '$(date) | tee $LOG && \\
HF_TOKEN=$HF_TOKEN torchrun \\
    --nproc_per_node=2 \\
    --master_port=29500 \\
    train_2gpu.py \\
    2>&1 | tee -a $LOG
" Enter

echo ""
echo "============================================================"
echo "  Training launched in tmux session: $SESSION"
echo "  Log file: $LOG"
echo ""
echo "  Attach to watch:   tmux attach -t $SESSION"
echo "  Detach:            Ctrl+B then D"
echo "  Quick log tail:    tail -f $LOG"
echo "============================================================"
