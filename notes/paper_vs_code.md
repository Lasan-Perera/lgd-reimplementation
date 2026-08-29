# LGD: paper vs released code

Running log of discrepancies between the CVPR 2024 paper and github.com/Fsoft-AIC/LGD.

---

## 1. Contrastive loss is not Equation 4
Paper Eq 4: hinge on ||(sqrt(abar_T) x0_tilde - x_T)/sqrt(1-abar_T)||^2 - M
Code (`diffusion/gaussian_diffusion.py`, `NCELoss` ~line 101): margin loss on
Euclidean distances between x_t, guiding point, model output. Weight 1e-3.
=> Proposition 1 does not apply to what is implemented.
Status: confirmed by reading code. Not yet run.

## 2. L_total is computed then discarded
`train_network_diffusion.py`, `train()`: `losses = compute_losses()` produces
mse + 1e-3*contr, then `loss` is overwritten by `net.compute_loss(...)` (plain MSE)
two lines later. Contrastive term never reaches the gradient.
Status: confirmed by reading code.

## 3. ALBEF is randomly initialized
`_init_albef()` builds ALBEF from Grounding.yaml; no checkpoint is loaded anywhere
in the repo. Only BERT is pretrained. Attention maps driving x0_tilde start as noise.
Fix: load ALBEF RefCOCO+ grounding checkpoint.
Status: confirmed by grep for load_state_dict across repo -> no hits.

## 4. Train split is the test split
`utils/data/grasp_anywhere_data.py` filters on split/grasp-anything++/test/seen.obj
for both train and val. split/grasp-anything++/train/seen.obj (14,516 samples) unused.
Status: confirmed by reading code.

## 5. Text-visual fusion line commented out
`inference/models/lgdm/network.py`: `img = img + y` is commented. Language reaches
the network only via the ALBEF attention mask multiplied into RGB.
Status: confirmed by reading code.

## 6. Table 9 parameter count excludes ALBEF
See notes/param_count_finding.md. Claimed 5.18M vs actual 573.63M total /
213.08M trainable. ALBEF alone is 418.54M and trainable.
Status: measured from released checkpoint. Baseline counts need verification.

## 7. Architecture differs from supplementary Table 7/8
Table 7/8 describes MLPs predicting a 5-parameter grasp vector (M=5).
Released LGDM is a GG-CNN-style conv encoder-decoder predicting pixel-wise
pos/cos/sin/width heatmaps at 224x224.
Status: confirmed by reading code.

## Minor
- README says `--network lgd`; registry only knows `lgdm`.
- Optimizer rebuilt every epoch inside `train()`, discarding Adam state.
- `_process_attention_mask` upsamples 14x14 -> 224x224 with a triple nested
  Python loop.
