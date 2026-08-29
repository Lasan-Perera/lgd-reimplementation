# LGD: paper vs released code

Running log of discrepancies between the CVPR 2024 paper and github.com/Fsoft-AIC/LGD.

**Status:** Track A (corrected reference run) is functional as of this session.
Qualitative results obtained from the released pretrained checkpoint on real
Grasp-Anything++ test images (apple/basket, spoon/cup, scissors on shelf) —
see experiments/qualitative_pretrained/. Quantitative IoU evaluation not yet run.

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
**Patch applied** to `_init_albef()` in this session — loads
`checkpoints/refcoco.pth` via `interpolate_pos_embed` + `load_state_dict(strict=False)`.
Affects future training runs only; does not retroactively change the released
`lgd_pretrained.pth` checkpoint, which was trained without it.

## 4. Train split is the test split

`utils/data/grasp_anywhere_data.py` filters on split/grasp-anything++/test/seen.obj
for both train and val. split/grasp-anything++/train/seen.obj (14,516 samples) unused.
Status: confirmed by reading code. Not yet fixed in the reference run.

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

## Detail: split key format

`grasp_anywhere_data.py:33` filters via `x.split('/')[-1][:-5]`, stripping 5 chars
from e.g. `<hash>_0_1.pt` -> `<hash>_0`. So split .obj lists are keyed by
image-hash + object index, not per-grasp-file. Confirm against .obj contents.

## Detail: ruamel.yaml import breaks on current pip versions

network.py:6 does `import ruamel.yaml as yaml` then calls the old
`yaml.load(file, Loader=yaml.Loader)` API, which ruamel.yaml removed in a
later release. requirements.txt pins no upper bound. Fix: `import yaml`
(plain PyYAML, already a dependency, same API still supported).
**Applied.**

---

## Bugs found getting Track A running (this session, CPU)

None of these are paper-vs-code discrepancies in the theoretical sense —
they're the code failing to run on current library versions, or genuine
logic bugs unrelated to the paper's claims. Listed because a reimplementation
should not repeat them, and because they're evidence of how unmaintained
this codebase is.

### 8. np.float / np.int / np.bool deprecated aliases

Removed in NumPy 1.20+; requirements.txt pins no numpy upper bound. Present in:
utils/visualisation/gridshow.py, utils/dataset_processing/grasp.py,
models/grasp_det_seg/data_OCID/misc.py, models/grasp_det_seg/data_OCID/transform.py,
diffusion/resample.py. Fix: plain float/int/bool. **Applied to all five files.**

### 9. mask is loaded but never read (language-driven path)

`grasp_anywhere_data.py` globs and sorts self.mask_files and computes
mask_file per-item, but the load (`mask.Mask.from_file`) and use
(`mask_out_image`) are both commented out. mask.zip expands to 324GB /
1.87M files for data that is never opened. Note: `grasp_anything_data.py`
(the non-language loader, different class) does read masks.
**Extraction skipped entirely for the language-driven path.**

### 10. validate() passes batched tensors where scalars are expected

`train_network_diffusion.py` validate() iterates val_data (batch_size=1) and
passes didx/rot/zoom_factor -- shape-(1,) tensors from default collate --
straight into get_gtbb() -> GraspRectangles.rotate(). np.cos(-angle) on a
(1,)-shaped input makes each rotation-matrix entry a length-1 array, giving R
shape (2,2,1) instead of (2,2); np.dot then fails against the (2,4) points
array. Fix: .item() on all three at the call site. **Applied.**

Same bug, three call sites, in evaluate_diffusion.py (get_gtbb, get_rgb,
get_depth). **Applied.**

### 11. torch.load defaults to CUDA regardless of --cpu

evaluate_diffusion.py: `torch.load(network)` with no map_location. The
released checkpoint was saved from a GPU machine, so torch.load tries to
restore tensors onto CUDA even when --cpu is set. Fix:
`torch.load(network, map_location=device)`. **Applied.**

### 12. --vis unconditionally requests depth, ignoring --use-depth

evaluate_diffusion.py's --vis block calls get_depth() regardless of
args.use_depth. Grasp-Anything++ has no depth channel, so with --use-depth 0
this crashes (depth_files was never populated, since include_depth=False in
the dataset constructor). Fix: pass depth_img=None at the call site.
**Applied.**

### 13. save_results crashes on depth_img=None despite it being the default

utils/visualisation/plot.py save_results() signature defaults depth_img=None,
but the body unconditionally calls `depth_img.any()` with no None check --
so the declared default is not actually usable as shipped. Fix:
`if depth_img is not None and depth_img.any():`. **Applied.**

### 14. results/ directory not created automatically

save_results() calls fig.savefig('results/rgb.png') without ensuring the
directory exists. Fix: `mkdir -p results/` before running --vis. Trivial,
noted because it's the kind of thing that silently varies by whoever's cwd
happened to already have the directory when this was last run by the authors.

---

## Open / needs verification

### `gt` (ground truth) appears unused inside p_sample_loop

`p_sample_loop` -> `p_sample_loop_progressive` -> `p_sample` all accept a `gt`
parameter, but `p_sample`'s call into `p_mean_variance` does not pass it.
Read as far as line ~480 of `diffusion/gaussian_diffusion.py`; no reference
to `gt` found in the visible body beyond accepting it. If confirmed: (a)
evaluation is not leaking ground truth into generation -- good; (b) the
parameter is vestigial. Needs a full trace of p_mean_variance before stating
as fact.

---

## Theoretical concerns (separate from code)

**"Contrastive" without negatives.** Eq. 4 has no negative pairs. Standard
InfoNCE contrasts a positive pair against sampled negatives; nothing
analogous appears here. Whether the term is contrastive in any meaningful
sense is arguable -- Appendix B explicitly distinguishes this approach from
prior work using "intermediate layer features for calculating contrastive
loss in negative sample pairs," so the absence is deliberate, but the naming
remains a stretch.

**Is Proposition 1's bound tight?** Conclusion is `E[||x_tilde0 - x0||^2] < C*delta`
with `C = beta^-2`, `beta = sqrt(abar_T / (1-abar_T))`. Under standard
schedules abar_T -> 0 as T grows, so beta -> 0 and C -> infinity. Needs
numerical evaluation against the actual cosine schedule (1000 steps). If C
is enormous the bound is vacuous. Not yet computed.

**Dimensional ambiguity.** ALBEF's attention map is W x H; the grasp vector
in the supplementary is 5-dimensional. Eq. 4 treats x0_tilde and x_T as
living in the same space. How the attention map becomes x0_tilde in a
compatible space is not spelled out, and the two architectural descriptions
(Table 7 vs released code) resolve it differently.

**Real-robot sample size.** Table 4 reports 30 trials per method per
scenario. Differences of 0.03 between methods are well inside the noise at
that n.

## First quantitative result (released checkpoint, corrected pipeline)

n=28 (via --split 0.98, seen-test), IoU threshold 0.25, angle offset <30deg:
7/28 = 0.250 success rate. Paper reports 0.48 on seen-test (n unclear, larger
internal split per README). At n=28, one sample flip = 3.6pp -- not
statistically comparable to the paper's figure yet. Needs a larger overnight
run (e.g. --split 0.85, ~200 samples) for a meaningful comparison.

## Detail: y_flatten's output is dead code, not just "commented out"

network.py forward(): `y = self.y_flatten(y)` is computed every forward pass
(3.75M params), but the only consumer -- `img = torch.clone(img).detach() + y`
-- is commented out. Since y contributes to no loss term, y_flatten receives
zero gradient during training; its weights are at random initialization in
the released checkpoint regardless of how long training ran.

Consequence: uncommenting this line now would inject an untrained/random
signal into lgd_pretrained.pth's already-calibrated features -- not a fix,
likely a regression. Properly restoring this requires retraining from
scratch with the line active (Phase 8), not a checkpoint-time patch.

Also note: the commented line uses `.detach()` on img, which would block
gradient flow from pos/cos/sin/width losses back into conv1-3 even if
restored as-is. Unclear if intentional (protect visual trunk from noisy
early-training text signal) or another oversight. Worth deciding explicitly
when writing the corrected version for Track B.

Empirical support: single_image_demo.py on the same real image with two
different prompts ("Pick up apple by its skin." vs "grasp the duck")
produced near-identical grasp boxes, both on the duck. Consistent with
language having no effective path to the prediction via this checkpoint.
See experiments/language_grounding_test/.
