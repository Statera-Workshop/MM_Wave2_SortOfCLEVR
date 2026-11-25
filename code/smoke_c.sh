#!/usr/bin/env bash
# Wave 2: generate data (if missing), gates, 1-epoch smoke of the 4 models (GPU).
set -u
cd ~/MM_Wave1/code; mkdir -p ~/MM_Wave1/logs/C_smoke
D=~/MM_Wave1/data/sortofclevr/sortofclevr.npz
[ -f $D ] || python gen_sortofclevr.py | tee ~/MM_Wave1/logs/C_smoke/gen.log
python gate_c.py --device cuda 2>&1 | tee ~/MM_Wave1/logs/C_smoke/gate.log
grep -q "GATE C: PASS" ~/MM_Wave1/logs/C_smoke/gate.log || { echo "GATE C FAILED - smoke not started"; exit 1; }
for m in cnnmlp rn film mac; do
  python mm_c.py --model $m --epochs 1 --out ~/MM_Wave1/runs/C_smoke --force > ~/MM_Wave1/logs/C_smoke/$m.log 2>&1
  tail -1 ~/MM_Wave1/logs/C_smoke/$m.log
done
nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv
echo "SMOKE C FINISHED"
