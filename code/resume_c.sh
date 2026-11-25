#!/usr/bin/env bash
# resume_c.sh "<CUTOFF>" "<HARD_STOP>"  - after moving the laptop: check CUDA, clear stopped/skipped Wave 2 jobs,
# start the queue runner again. Finished runs (final.json) are skipped; an interrupted run restarts from scratch.
set -eu
CUT="${1:?cutoff, e.g. 2026-11-27 08:00}"; HARD="${2:?hard stop, e.g. 2026-11-27 08:30}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK', torch.cuda.get_device_name(0))"
if pgrep -f queue_runner.sh >/dev/null; then echo "a queue runner is still running - stop it first (touch ~/MM_Wave1/queue/STOP)"; exit 1; fi
rm -f ~/MM_Wave1/queue/STOP
n=$(ls ~/MM_Wave1/queue/state/C_*.stopped ~/MM_Wave1/queue/state/C_*.skip 2>/dev/null | wc -l)
rm -f ~/MM_Wave1/queue/state/C_*.stopped ~/MM_Wave1/queue/state/C_*.skip
echo "cleared $n stopped/skipped Wave 2 job states"
cd ~/MM_Wave1 && SSL_BASE=$HOME/MM_Wave1 CUTOFF="$CUT" HARD_STOP="$HARD" nohup bash queue_runner.sh > logs/runner.nohup 2>&1 &
sleep 5; tail -2 ~/MM_Wave1/logs/queue.log
