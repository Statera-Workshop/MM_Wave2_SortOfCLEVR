"""PAI-SG Multimodal Wave 2: visual question answering convergence on Sort-of-CLEVR.

Models (all from scratch, small):
  cnnmlp : CNN (4 x conv 24, stride 2) -> 5 x 5 x 24, flatten + question -> MLP 256-256-10 (RN paper baseline)
  rn     : same CNN; 25 cell objects (+ 2 coordinates) -> g(o_i, o_j, q) 4 x 256 summed -> f 256-256-10,
           dropout 0.5 in f (Relation Network, Santoro et al. 2017)
  film   : stem 2 x conv 32 stride 2 -> 19 x 19 + coordinates, 2 FiLM residual blocks (gamma, beta from the
           question), 1x1 conv 64, global max pool, MLP 64-128-10 (FiLM, Perez et al. 2018, reduced)
  mac    : same stem, knowledge base 361 x 64; 4 MAC steps with control from the question vector,
           read attention over the knowledge base, linear write; output MLP [q, m] (MAC, Hudson & Manning 2018,
           reduced: the question is an 11-d vector, so the control unit has no word attention; the two
           position-invariant biases before the read softmax are dropped because they cannot affect it) [Framework]
[Unified] Adam 1e-4 constant, batch 64, 20 / 60 epochs, seeds 777 / 778; evaluation at 2 points per epoch below
          30 epochs (>= 30 evaluations), else every epoch. Every evaluation reports the plain eval-mode accuracy
          and the accuracy after re-estimating BatchNorm statistics on 50 training batches (series practice).
A run writes final.json only when it finishes and is skipped if final.json exists (safe to stop and resume).
"""
import argparse, copy, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = ["cnnmlp", "rn", "film", "mac"]
N_ANS = 10


# ----------------------------------------------------------------------------- data
def load_data(path, device, expect=(10000, 2000)):
    z = np.load(path)
    out = {}
    for split, n in zip(("train", "test"), expect):
        img, q, a = z[f"{split}_img"], z[f"{split}_q"], z[f"{split}_a"]
        assert img.shape == (n, 75, 75, 3) and q.shape == (n, 10, 11) and a.shape == (n, 10), (split, img.shape)
        nq = q.shape[1]
        out[split] = {"img": torch.from_numpy(np.ascontiguousarray(img.transpose(0, 3, 1, 2))).to(device),
                      "qi": torch.arange(n, device=device).repeat_interleave(nq),
                      "q": torch.from_numpy(q.reshape(-1, 11)).to(device),
                      "a": torch.from_numpy(a.reshape(-1).astype(np.int64)).to(device)}
        out[split]["rel"] = out[split]["q"][:, 7] == 1
    return out


def batch(d, idx):
    return d["img"][d["qi"][idx]].float().div_(255.0), d["q"][idx], d["a"][idx]


# ----------------------------------------------------------------------------- models
def conv_bn(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.BatchNorm2d(cout), nn.ReLU())


def coords(n, h, w, device):
    ys = torch.linspace(-1, 1, h, device=device).view(1, 1, h, 1).expand(n, 1, h, w)
    xs = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, w).expand(n, 1, h, w)
    return torch.cat([xs, ys], 1)


class CNN24(nn.Module):  # RN paper encoder: 4 conv 24, stride 2 -> 5 x 5 x 24 for 75 x 75 input
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(conv_bn(3, 24, 2), conv_bn(24, 24, 2), conv_bn(24, 24, 2), conv_bn(24, 24, 2))

    def forward(self, x):
        return self.net(x)


class CNNMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN24()
        self.mlp = nn.Sequential(nn.Linear(5 * 5 * 24 + 11, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, N_ANS))

    def forward(self, x, q):
        return self.mlp(torch.cat([torch.flatten(self.cnn(x), 1), q], 1))


class RN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN24()
        d = 2 * (24 + 2) + 11
        self.g = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                               nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.f = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
                               nn.Linear(256, N_ANS))

    def forward(self, x, q):
        fm = self.cnn(x)
        n, c, h, w = fm.shape
        o = torch.cat([fm, coords(n, h, w, x.device)], 1).flatten(2).transpose(1, 2)  # (n, 25, 26)
        k = o.shape[1]
        oi = o.unsqueeze(2).expand(n, k, k, o.shape[2])
        oj = o.unsqueeze(1).expand(n, k, k, o.shape[2])
        qq = q.view(n, 1, 1, 11).expand(n, k, k, 11)
        rel = self.g(torch.cat([oi, oj, qq], 3)).sum((1, 2))
        return self.f(rel)


class Stem32(nn.Module):  # 75 -> 38 -> 19, 32 channels
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(conv_bn(3, 32, 2), conv_bn(32, 32, 2))

    def forward(self, x):
        return self.net(x)


class FiLMBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c + 2, c, 1)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1)
        self.bn = nn.BatchNorm2d(c, affine=False)

    def forward(self, x, gamma, beta):
        n, _, h, w = x.shape
        r = F.relu(self.c1(torch.cat([x, coords(n, h, w, x.device)], 1)))
        y = self.bn(self.c2(r))
        y = F.relu(gamma.view(n, -1, 1, 1) * y + beta.view(n, -1, 1, 1))
        return r + y


class FiLM(nn.Module):
    def __init__(self, c=32, blocks=2):
        super().__init__()
        self.stem = Stem32()
        self.blocks = nn.ModuleList([FiLMBlock(c) for _ in range(blocks)])
        self.gen = nn.Linear(11, blocks * 2 * c)
        self.cls = nn.Sequential(nn.Conv2d(c + 2, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, N_ANS))
        self.c, self.nb = c, blocks

    def forward(self, x, q):
        h = self.stem(x)
        gb = self.gen(q).view(-1, self.nb, 2, self.c)
        for i, b in enumerate(self.blocks):
            h = b(h, 1 + gb[:, i, 0], gb[:, i, 1])  # gamma centred at 1
        n, _, hh, ww = h.shape
        h = self.cls(torch.cat([h, coords(n, hh, ww, x.device)], 1))
        return self.mlp(h.amax((2, 3)))


class MAC(nn.Module):
    def __init__(self, d=64, steps=4):
        super().__init__()
        self.stem = Stem32()
        self.kb = nn.Conv2d(32 + 2, d, 1)
        self.qenc = nn.Linear(11, d)
        self.ctrl = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(steps)])
        self.mem_proj, self.kb_proj = nn.Linear(d, d), nn.Linear(d, d, bias=False)  # a bias here cancels in the softmax
        self.read_cat = nn.Linear(2 * d, d)
        self.read_att = nn.Linear(d, 1, bias=False)  # likewise position-invariant
        self.write = nn.Linear(2 * d, d)
        self.m0 = nn.Parameter(torch.zeros(d))
        self.c0 = nn.Parameter(torch.zeros(d))
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.ELU(), nn.Linear(d, N_ANS))
        self.d, self.steps = d, steps

    def forward(self, x, q):
        h = self.stem(x)
        n, _, hh, ww = h.shape
        kb = self.kb(torch.cat([h, coords(n, hh, ww, x.device)], 1)).flatten(2).transpose(1, 2)  # (n, 361, d)
        qv = torch.tanh(self.qenc(q))
        c = self.c0.expand(n, self.d)
        m = self.m0.expand(n, self.d)
        kbp = self.kb_proj(kb)
        for i in range(self.steps):
            c = self.ctrl[i](torch.cat([c, qv], 1))                         # control (no word attention)
            inter = self.mem_proj(m).unsqueeze(1) * kbp                      # read: memory-knowledge interaction
            inter = self.read_cat(torch.cat([inter, kb], 2))
            att = F.softmax(self.read_att(inter * c.unsqueeze(1)).squeeze(2), 1)
            r = (att.unsqueeze(2) * kb).sum(1)
            m = self.write(torch.cat([r, m], 1))                             # write
        return self.out(torch.cat([qv, m], 1))


def build(name):
    return {"cnnmlp": CNNMLP, "rn": RN, "film": FiLM, "mac": MAC}[name]()


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, d, bs=500):
    model.eval()
    preds = []
    for s in range(0, d["a"].numel(), bs):
        idx = torch.arange(s, min(s + bs, d["a"].numel()), device=d["a"].device)
        x, q, _ = batch(d, idx)
        preds.append(model(x, q).argmax(1))
    p = torch.cat(preds)
    ok = p == d["a"]
    rel = d["rel"]
    return {"acc": ok.sum().item() / ok.numel(), "rel": ok[rel].sum().item() / rel.sum().item(),
            "nonrel": ok[~rel].sum().item() / (~rel).sum().item()}


@torch.no_grad()
def bn_reestimated(model, d, n_batches, bs, seed):
    m2 = copy.deepcopy(model)
    for m in m2.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(d["a"].numel(), generator=g)[: n_batches * bs].to(d["a"].device)
    m2.train()
    for s in range(0, idx.numel(), bs):
        x, q, _ = batch(d, idx[s:s + bs])
        m2(x, q)
    m2.eval()
    return m2


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(a):
    name = f"{a.model}_e{a.epochs}_s{a.seed}"
    out = os.path.join(a.out, name)
    if os.path.exists(os.path.join(out, "final.json")) and not a.force:
        print(f"SKIP {name}: final.json exists")
        return
    os.makedirs(out, exist_ok=True)
    dev = torch.device(a.device)
    seed_all(a.seed)
    d = load_data(a.data, dev, tuple(int(x) for x in a.expect.split(",")))
    tr, te = d["train"], d["test"]
    seed_all(a.seed)
    model = build(a.model).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    n = tr["a"].numel()
    spe = n // a.batch
    per_epoch = 2 if a.epochs < 30 else 1
    eval_steps = {round(spe * (k + 1) / per_epoch) for k in range(per_epoch)}
    meta = {"name": name, "args": {k: (v.replace(os.path.expanduser("~"), "~") if isinstance(v, str) else v)
                                   for k, v in vars(a).items()},
            "n_params": sum(p.numel() for p in model.parameters()), "n_train_qa": n, "steps_per_epoch": spe,
            "evals_per_epoch": per_epoch, "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    cf = open(os.path.join(out, "curve.jsonl"), "w")
    g = torch.Generator().manual_seed(a.seed)
    print(f"START {name} params={meta['n_params']} qa={n} steps/epoch={spe}", flush=True)

    def log(ep, extra):
        t0 = time.time()
        r = {"epoch": ep, **{f"plain_{k}": v for k, v in evaluate(model, te).items()}}
        m2 = bn_reestimated(model, tr, 50, a.batch, a.seed + int(ep * 10))
        r.update({f"bn_{k}": v for k, v in evaluate(m2, te).items()})
        r.update(extra)
        r["eval_sec"] = time.time() - t0
        cf.write(json.dumps(r) + "\n")
        cf.flush()
        return r

    rec = log(0, {})
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        perm = torch.randperm(n, generator=g).to(dev)
        tot, nb = 0.0, 0
        for s in range(spe):
            model.train()
            x, q, y = batch(tr, perm[s * a.batch:(s + 1) * a.batch])
            loss = F.cross_entropy(model(x, q), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
            if not math.isfinite(loss.item()):
                json.dump({"status": "nan", "epoch": ep}, open(os.path.join(out, "final.json"), "w"))
                print(f"DONE {name} status=nan", flush=True)
                return
            if (s + 1) in eval_steps:
                frac = ep - 1 + (s + 1) / spe
                rec = log(round(frac, 3), {"train_loss": tot / nb, "epoch_sec": time.time() - t0})
        print(f"ep {ep} loss {tot / nb:.4f} acc plain {rec['plain_acc']:.4f} bn {rec['bn_acc']:.4f} "
              f"(rel {rec['bn_rel']:.3f} nonrel {rec['bn_nonrel']:.3f}) {time.time() - t0:.1f}s", flush=True)
    cf.close()
    torch.save(model.state_dict(), os.path.join(out, "last.pt"))
    json.dump({"status": "ok", **rec}, open(os.path.join(out, "final.json"), "w"), indent=1)
    print(f"DONE {name} status=ok acc_bn={rec['bn_acc']:.4f}", flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--data", default=os.path.expanduser("~/MM_Wave1/data/sortofclevr/sortofclevr.npz"))
    p.add_argument("--expect", default="10000,2000")
    p.add_argument("--out", default=os.path.expanduser("~/MM_Wave1/runs/C"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
