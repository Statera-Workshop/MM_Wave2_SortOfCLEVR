"""Wave 2 analysis: per (model, epochs) mean over seeds (range) of the final accuracy - plain and BN re-estimated -
relational / non-relational (BN re-est.), peak and its epoch, E90 (BN re-est.), dips (>= 0.1 drops, plain / BN)."""
import argparse, glob, json, os
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.expanduser("~/MM_Wave1/runs/C"))
    a = p.parse_args()
    g = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.runs, "*", "curve.jsonl"))):
        dd = os.path.dirname(f)
        if not os.path.exists(os.path.join(dd, "final.json")):
            continue
        ar = json.load(open(os.path.join(dd, "meta.json")))["args"]
        rows = [json.loads(l) for l in open(f) if l.strip()][1:]
        b = [r["bn_acc"] for r in rows]
        pl = [r["plain_acc"] for r in rows]
        pk = max(b)
        dips = lambda v: sum(1 for i in range(1, len(v)) if v[i - 1] - v[i] >= 0.1)
        g[(ar["model"], ar["epochs"])].append({
            "plain": pl[-1], "bn": b[-1], "rel": rows[-1]["bn_rel"], "nonrel": rows[-1]["bn_nonrel"], "peak": pk,
            "peak_ep": rows[b.index(pk)]["epoch"], "e90": next(r["epoch"] for r in rows if r["bn_acc"] >= 0.9 * b[-1]),
            "dips": f"{dips(pl)}/{dips(b)}"})
    m = lambda xs: sum(xs) / len(xs)
    mr = lambda xs: f"{m(xs):.3f} ({max(xs) - min(xs):.3f})" if len(xs) > 1 else f"{xs[0]:.3f}"
    print("| model | epochs | seeds | final plain | final BN re-est. | relational | non-relational | peak (epoch) | E90 | dips plain/BN |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    order = ["cnnmlp", "rn", "film", "mac"]
    for k in sorted(g, key=lambda k: (k[1], order.index(k[0]))):
        v = g[k]
        print(f"| {k[0]} | {k[1]} | {len(v)} | {mr([x['plain'] for x in v])} | {mr([x['bn'] for x in v])} | "
              f"{mr([x['rel'] for x in v])} | {mr([x['nonrel'] for x in v])} | {m([x['peak'] for x in v]):.3f} "
              f"({m([x['peak_ep'] for x in v]):.1f}) | {m([x['e90'] for x in v]):.1f} | {' '.join(x['dips'] for x in v)} |")
    print(f"\n{sum(len(v) for v in g.values())} finished runs")


if __name__ == "__main__":
    main()
