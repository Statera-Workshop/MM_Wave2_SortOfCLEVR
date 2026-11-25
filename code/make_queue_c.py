"""Append the 16 Wave 2 jobs to queue/jobs.txt from the smoke timings (20-epoch set first, then 60)."""
import json, os
S = os.path.expanduser("~/MM_Wave1/runs/C_smoke")
lines, tot = [], 0
for ep in (20, 60):
    for s in (777, 778):
        for m in ("cnnmlp", "rn", "film", "mac"):
            rows = [json.loads(l) for l in open(f"{S}/{m}_e1_s777/curve.jsonl")][1:]
            per = rows[-1]["epoch_sec"] + rows[-1]["eval_sec"]  # training + 2 evaluations per epoch (upper bound for 60 ep)
            est = max(1, round(per * ep * 1.2 / 60))
            tot += est
            lines.append(f"C_{m}_e{ep}_s{s} | {est} | cd ~/MM_Wave1/code && python mm_c.py --model {m} --epochs {ep} --seed {s}")
open(os.path.expanduser("~/MM_Wave1/queue/jobs.txt"), "a").write("\n".join(lines) + "\n")
print("\n".join(l.split(" | cd")[0] for l in lines))
print(f"{len(lines)} jobs appended, estimated {tot / 60:.1f} h")
