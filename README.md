# MM_Wave2_SortOfCLEVR

**Visual question answering convergence on Sort-of-CLEVR - PAI-SG Multimodal Wave 2**

Part of the *Physical AI* reference series of the **Statera Working Group**. Convergence of four visual question answering models trained from scratch on Sort-of-CLEVR, a small generated relational-reasoning benchmark. Model repositories: [MM_VQA_CNNMLP](https://github.com/Statera-Workshop/MM_VQA_CNNMLP), [MM_VQA_RelationNet](https://github.com/Statera-Workshop/MM_VQA_RelationNet), [MM_VQA_FiLM](https://github.com/Statera-Workshop/MM_VQA_FiLM), [MM_VQA_MAC](https://github.com/Statera-Workshop/MM_VQA_MAC).

## Protocol

**Common setup** (identical for the five Wave 2 repositories).
Sort-of-CLEVR generated with `code/gen_sortofclevr.py` (no download): 75 x 75 RGB images with 6 objects (one per
colour, square or circle), 10 questions per image as 11-d vectors (colour, relational or not, subtype), 10 answers;
5 non-relational questions (shape, left?, top?) and 5 relational ones (shape of the closest / furthest object, count of
the same shape). Train 10,000 images (100,000 questions), test 2,000 (20,000). All models from scratch: Adam 1e-4
constant, batch 64, budgets 20 and 60 epochs, seeds 777 / 778, fp32, deterministic cuDNN. Test accuracy is evaluated
twice per epoch below 30 epochs and every epoch otherwise, each time in plain eval mode and after re-estimating the
BatchNorm statistics on 50 training batches (the series' evaluation practice); the two agree within 0.004 here.
Reported value = last evaluation.

## Results (Sort-of-CLEVR test, mean of seeds; range in brackets)

| Model | Epochs | Seeds | Final accuracy, plain (range) | Final, BN re-estimated (range) | Relational | Non-relational | Peak (epoch) | E90 (epochs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN+MLP | 20 | 2 | 0.654 (0.007) | 0.654 (0.006) | 0.704 | 0.604 | 0.677 (10.2) | 2.8 |
| CNN+MLP | 60 | 2 | 0.628 (0.000) | 0.629 (0.001) | 0.678 | 0.579 | 0.676 (8.5) | 1.5 |
| Relation Network | 20 | 2 | 0.689 (0.032) | 0.690 (0.034) | 0.728 | 0.651 | 0.691 (19.8) | 7.0 |
| Relation Network | 60 | 2 | 0.914 (0.012) | 0.914 (0.013) | 0.842 | 0.986 | 0.915 (58.0) | 34.0 |
| FiLM | 20 | 2 | 0.880 (0.006) | 0.880 (0.006) | 0.777 | 0.984 | 0.884 (19.0) | 9.5 |
| FiLM | 60 | 2 | 0.883 (0.005) | 0.883 (0.005) | 0.775 | 0.990 | 0.888 (33.0) | 10.0 |
| MAC | 20 | 2 | 0.694 (0.027) | 0.690 (0.036) | 0.717 | 0.662 | 0.690 (20.0) | 7.8 |
| MAC | 60 | 2 | 0.877 (0.023) | 0.880 (0.018) | 0.770 | 0.989 | 0.883 (59.5) | 29.5 |

## Findings

- **All four models converge** well above chance (0.1); every relational model clearly beats the CNN+MLP baseline.
- **FiLM converges fastest** (its 20-epoch result is already its plateau); the Relation Network and MAC start slowly and catch up by 60 epochs, where the Relation Network is best - the ranking depends on the budget.
- **Relational questions separate the models**: the Relation Network is best on them; non-relational questions are near-solved by every relational model at 60 epochs.
- **CNN+MLP peaks early and overfits.**
- The Relation Network and MAC are still rising at 60 epochs (this study checks convergence behaviour, not the best reachable accuracy). Plain and BN re-estimated evaluation agree within 0.004 (no BatchNorm lag at lr 1e-4).

Per-run curves: `results/runs/<run>/curve.jsonl`. Weights are in the model repositories.

## Run

```bash
cd code && bash smoke_c.sh && python make_queue_c.py
```
