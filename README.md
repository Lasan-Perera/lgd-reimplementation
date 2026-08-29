# LGD Reimplementation

From-scratch reimplementation of "Language-driven Grasp Detection"
(Vuong et al., CVPR 2024) — arXiv:2406.09489.

Goal: understand every component from first principles, not to re-run released code.

## Layout
- `derivations/` — math worked by hand (Modules 0-6)
- `notes/`       — findings, paper-vs-code discrepancies
- `src/`         — the reimplementation
- `experiments/` — runs and results

## Status
- [x] Environment (Python 3.9, torch 1.12.1+cpu)
- [x] Dataset downloaded (Grasp-Anything + ++, ~73 GB)
- [ ] Module 0-6 derivations
- [ ] Data pipeline
- [ ] Diffusion core
- [ ] Network
- [ ] Losses
- [ ] Eval harness
- [ ] Training (needs GPU)

## Key references
- Paper: arXiv:2406.09489
- Released code: github.com/Fsoft-AIC/LGD
- Dataset: huggingface.co/datasets/airvlab/Grasp-Anything{,-pp}
