"""gen_sortofclevr.py - generate the Sort-of-CLEVR data of PAI-SG Multimodal Wave 2 (CPU, about a minute).

Sort-of-CLEVR (Santoro et al., "A simple neural network module for relational reasoning", 2017): 75 x 75 RGB
images with 6 objects, one per colour (red, green, blue, orange, gray, yellow), each a square or a circle (size 5).
Questions are 11-d vectors: colour one-hot (6) + type one-hot (non-relational, relational) + subtype one-hot (3).
  non-relational: 0 shape of <colour> -> rectangle/circle   1 is <colour> on the left? -> yes/no
                  2 is <colour> on the top? -> yes/no
  relational:     0 shape of the object closest to <colour>  1 shape of the object furthest from <colour>
                  2 how many objects have the shape of <colour> -> 1..6
Answers (10): yes, no, rectangle, circle, 1, 2, 3, 4, 5, 6. 5 non-relational + 5 relational questions per image.
Output: sortofclevr.npz (train 10,000 / test 2,000 images) and sortofclevr.json (counts, answer histograms, md5).
"""
import argparse, hashlib, json, os
import numpy as np
import cv2

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 156, 0), (128, 128, 128), (255, 255, 0)]  # RGB
SIZE, IMG, NQ = 5, 75, 5
ANSWERS = ["yes", "no", "rectangle", "circle", "1", "2", "3", "4", "5", "6"]


def make_image(rng):
    centers, shapes = [], []
    img = np.zeros((IMG, IMG, 3), np.uint8)
    for c in range(6):
        while True:
            ctr = rng.integers(SIZE, IMG - SIZE, size=2)
            if all(((ctr - o) ** 2).sum() > (2 * SIZE + 2) ** 2 for o in centers):
                break
        centers.append(ctr)
        sq = bool(rng.integers(2))
        shapes.append(0 if sq else 1)  # 0 rectangle, 1 circle
        col = COLORS[c]
        if sq:
            cv2.rectangle(img, (int(ctr[0] - SIZE), int(ctr[1] - SIZE)), (int(ctr[0] + SIZE), int(ctr[1] + SIZE)), col, -1)
        else:
            cv2.circle(img, (int(ctr[0]), int(ctr[1])), SIZE, col, -1)
    return img, np.array(centers), np.array(shapes)


def questions(rng, centers, shapes):
    qs, ans = [], []
    for rel in (0, 1):
        for _ in range(NQ):
            c, sub = int(rng.integers(6)), int(rng.integers(3))
            q = np.zeros(11, np.float32)
            q[c] = 1
            q[6 + rel] = 1
            q[8 + sub] = 1
            if rel == 0:
                if sub == 0:
                    a = 2 + shapes[c]
                elif sub == 1:
                    a = 0 if centers[c][0] < IMG / 2 else 1
                else:
                    a = 0 if centers[c][1] < IMG / 2 else 1
            else:
                d = ((centers - centers[c]) ** 2).sum(1).astype(float)
                if sub == 0:
                    d[c] = np.inf
                    a = 2 + shapes[int(d.argmin())]
                elif sub == 1:
                    a = 2 + shapes[int(d.argmax())]
                else:
                    a = 4 + int((shapes == shapes[c]).sum()) - 1
            qs.append(q)
            ans.append(a)
    return np.stack(qs), np.array(ans, np.int64)


def build(n, seed):
    rng = np.random.default_rng(seed)
    imgs, qs, ans = [], [], []
    for _ in range(n):
        im, ctr, sh = make_image(rng)
        q, a = questions(rng, ctr, sh)
        imgs.append(im)
        qs.append(q)
        ans.append(a)
    return np.stack(imgs), np.stack(qs), np.stack(ans)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.expanduser("~/MM_Wave1/data/sortofclevr"))
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=2000)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr = build(a.n_train, 1)
    te = build(a.n_test, 2)
    path = os.path.join(a.out, "sortofclevr.npz")
    np.savez(path, train_img=tr[0], train_q=tr[1], train_a=tr[2], test_img=te[0], test_q=te[1], test_a=te[2])
    info = {"n_train": a.n_train, "n_test": a.n_test, "questions_per_image": 2 * NQ, "answers": ANSWERS,
            "md5": hashlib.md5(open(path, "rb").read()).hexdigest()}
    for name, (q, ans) in (("train", tr[1:]), ("test", te[1:])):
        for rel in (0, 1):
            for sub in range(3):
                m = (q[..., 6 + rel] == 1) & (q[..., 8 + sub] == 1)
                info[f"{name}_{'rel' if rel else 'nonrel'}{sub}_answers"] = np.bincount(ans[m], minlength=10).tolist()
    json.dump(info, open(os.path.join(a.out, "sortofclevr.json"), "w"), indent=1)
    print(f"train {tr[0].shape} questions {tr[1].shape}  test {te[0].shape}  md5 {info['md5']}")
    print("GEN SORT-OF-CLEVR: OK")


if __name__ == "__main__":
    main()
