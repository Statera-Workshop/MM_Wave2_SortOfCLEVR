"""Wave 2 gates: counts, eval path 1.000, every parameter updated, 256-QA overfit. Ends 'GATE C: PASS/FAIL'."""
import argparse, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_c as M


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.expanduser("~/MM_Wave1/data/sortofclevr/sortofclevr.npz"))
    p.add_argument("--expect", default="10000,2000")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--models", default=",".join(M.MODELS))
    a = p.parse_args()
    dev = torch.device(a.device)
    ok = True

    def check(name, good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"{name} {msg} -> {'OK' if good else 'FAIL'}", flush=True)

    d = M.load_data(a.data, dev, tuple(int(x) for x in a.expect.split(",")))
    tr, te = d["train"], d["test"]
    check("[1] counts", True, f"train qa {tr['a'].numel()} (rel {int(tr['rel'].sum())}) test qa {te['a'].numel()} "
          f"answers used {sorted(torch.unique(tr['a']).tolist())}")

    class Oracle(torch.nn.Module):  # predicts the true answer: checks the evaluation path only
        def __init__(self):
            super().__init__()
            self.ans = te["a"]
            self.pos = 0

        def forward(self, x, q):
            n = x.shape[0]
            out = F.one_hot(self.ans[self.pos:self.pos + n], M.N_ANS).float()
            self.pos += n
            return out
    r = M.evaluate(Oracle(), te)
    check("[2] eval path", r["acc"] == 1.0 and r["rel"] == 1.0 and r["nonrel"] == 1.0,
          f"acc {r['acc']:.3f} rel {r['rel']:.3f} nonrel {r['nonrel']:.3f}")

    idx = torch.arange(a.n, device=dev)
    x, q, y = M.batch(tr, idx)
    for name in a.models.split(","):
        M.seed_all(777)
        m = M.build(name).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        before = {k: v.detach().clone() for k, v in m.named_parameters()}
        m.train()
        loss = F.cross_entropy(m(x[:64], q[:64]), y[:64])
        opt.zero_grad()
        loss.backward()
        opt.step()
        stale = [k for k, v in m.named_parameters() if torch.equal(before[k], v.detach())]
        check(f"[3] {name:6s}", not stale, f"params {sum(v.numel() for v in m.parameters()):,} unchanged {stale or 0}")

    for name in a.models.split(","):
        M.seed_all(777)
        m = M.build(name).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        t0, hit, acc = time.time(), None, 0.0
        for s in range(1, a.steps + 1):
            m.train()
            loss = F.cross_entropy(m(x, q), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if s % 100 == 0:
                m.eval()
                with torch.no_grad():
                    acc = (m(x, q).argmax(1) == y).sum().item() / y.numel()
                if acc == 1.0:
                    hit = s
                    break
        check(f"[4] {name:6s}", hit is not None, f"overfit {a.n}: acc {acc:.3f} at step {hit} {time.time() - t0:.0f}s")
    print("GATE C:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
