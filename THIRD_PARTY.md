# Third-party code and data

No third-party code is included. The models are re-implemented from the papers:

| Component | Paper |
| --- | --- |
| Sort-of-CLEVR task and generator design, CNN+MLP baseline, Relation Network | Santoro et al., *A simple neural network module for relational reasoning*, NeurIPS 2017 |
| FiLM | Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018 |
| MAC | Hudson & Manning, *Compositional Attention Networks for Machine Reasoning*, ICLR 2018 |

**Data.** Sort-of-CLEVR is generated locally by `code/gen_sortofclevr.py` (seeds 1 and 2); nothing is downloaded.
