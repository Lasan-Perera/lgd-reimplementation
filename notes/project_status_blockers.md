# Project Status — Blockers and Open Problems

Last updated: end of session 2 (30 Aug 2026)

---

## Where the project actually stands

**Done:** the codebase now implements the paper's method. Six substantive fixes applied and verified working together on CPU. Dataset downloaded, extracted, and validated. Baseline measured. Two behavioural experiments run.

**Not done:** the model has never been trained. Every result so far comes from the *released* checkpoint (`lgd_pretrained.pth`), which was produced by the buggy original code. Nothing has been trained with the fixes active.

**This is the single blocker.** Everything else is downstream of it.

---

## THE blocker: no GPU

### Why it blocks everything

The model chains ResNet-50 + ViT + BERT + CLIP + a diffusion process. Measured on this machine: **~55 seconds per single inference**, and forward+backward during training is heavier still.

Arithmetic: 28,621 training samples at batch size 8 ≈ 3,578 batches per epoch. At tens of seconds per batch on CPU that is roughly **a day per epoch**, and convergence needs dozens. Not slow — infeasible.

### Why the hardware can't be fixed locally

- Lenovo IdeaPad 5 15ABA7, AMD Ryzen with integrated Vega graphics (Barcelo)
- AMD dropped ROCm support for Vega/GCN5 after ROCm 5.6; APU support was never reliable even then
- Current ROCm (7.x) does not support this chip at all
- No CUDA path exists — this is not an NVIDIA GPU

There is no software fix. The only route is renting compute.

### What is blocked until this is resolved

| Blocked item | Why |
|---|---|
| Any performance claim about the fixed implementation | Nothing has been trained |
| Eq. 4 vs NCE comparison (the planned extension) | Requires two trained models |
| Testing whether restoring language fusion fixes grounding | Requires a trained model |
| Reproducing the paper's 0.48 figure | Requires training on the real split |
| Testing whether Proposition 1 holds empirically | Requires observing a real training run |

### Resolution

Rent a GPU: Vast.ai / RunPod / Lambda, roughly $0.20–0.80/hr for an RTX 3090/4090-class card. Estimated a few hours, likely under $10 total.

Workflow: push patched repo → clone on instance → re-download dataset there (faster than uploading 73 GB) → train in `tmux` → `rsync` checkpoint back → destroy instance.

Sizing note: the original defaults (1000 epochs × 1000 batches) were tuned for a much larger internal dataset the authors did not release. Size the run from measured per-batch time on the rented GPU, not from those defaults.

---

## Secondary problems (real, not blocking)

### 1. Eq. 4 required an interpretive decision, not a literal transcription

The paper writes Eq. 4 in terms of `x_T` specifically — it falls out of the boundary term of the Eq. 2 telescoping sum. Training samples one random `t` per batch, never literally `t=T`.

The implementation generalizes Eq. 4 to the sampled `t` using the same closed-form relation (Eq. 11) that holds for any `t`. Mathematically sound, but it is a design choice, and it should be stated explicitly in the write-up rather than presented as "what the paper says".

Consequence found in practice: dividing by `sqrt(1-abar_t)` blows up as `t → 0` (observed `norm_sq` of 1.7M on a `t=0` sample). Mitigated with an epsilon in the denominator. Worth revisiting whether excluding or downweighting very low `t` is more principled — cannot be decided without observing a real training run.

### 2. `contr_eq4` reads 0.0000 on an untrained network

Diagnosed, not a bug. An untrained residual network outputs `pos_output = x_t + (≈0)`, so `x0_tilde ≈ x_t` and the hinge sits below the margin `M`. Verified the arithmetic by hand against printed intermediates.

But this means **the Eq. 4 term has never been observed doing anything during actual learning.** Whether it contributes meaningfully once the network starts predicting is untested and untestable without training.

### 3. Table 9 parameter accounting — one open question remains

Measured: 573.63M total / 213.08M trainable, against the paper's claimed 5.18M. ALBEF alone is 418.54M and trainable.

Unresolved: supplementary E.1 states the linguistic baselines "also utilize ALBEF architecture to do the fusion". If their reported counts also exclude ALBEF, the *relative* comparison partially survives even though every absolute number is wrong. Settling this needs the baseline configs, which have not been checked.

### 4. `gt` in the sampling loop — traced but not fully confirmed

`p_sample_loop` → `p_sample_loop_progressive` → `p_sample` all accept a `gt` parameter; `p_sample`'s call into `p_mean_variance` does not appear to pass it. If confirmed, evaluation is not leaking ground truth into generation (good) and the parameter is vestigial.

Read as far as ~line 480 of `gaussian_diffusion.py`. A full trace of `p_mean_variance` has not been done. Should not be stated as fact in the write-up until it is.

### 5. The `.detach()` decision was made without evidence

The original commented-out fusion line read `img = torch.clone(img).detach() + y`. The `.detach()` would block gradients from the grasp losses reaching `conv1-3` through this path.

It was removed on the reasoning that blocking those gradients looks unintentional. But it is possible the authors did this deliberately — e.g. to protect the visual trunk from a noisy text signal early in training. **This is an untested architectural choice**, and if training is unstable it is the first thing to try reverting.

---

## Data problems (resolved, noted for reproducibility)

- **`mask.zip` expands to 324 GB / 1.87M files** and is never read by the language-driven loader (`grasp_anywhere_data.py` computes the path but the load and use are both commented out). Skipped entirely. Note: if the GA++-Spatial extension idea is ever pursued, this data becomes necessary.
- **Two mandatory renames** — `scene_description` → `prompt`, `grasp_label_positive` → `positive_grasp`. Silent empty dataset otherwise.
- **Archives nest** — `image.zip` → `image/image/`, `mask.zip` → `mask/mask/`. Flatten by renaming the directory, never `mv dir/*` (argument list overflow at ~1M files).
- **`hf download` drops files** when several are passed to `--include` at once. One file per invocation.
- **SSD is NTFS via `ntfs-3g`** (FUSE). `ntfs3` refuses to mount after unclean disconnect. Slower on many small files; extraction took hours. Always `sudo umount` before unplugging.

---

## Known limitations that will not be fixed

These are properties of the method and dataset, not defects to engineer around. They belong in the write-up as findings.

### Synthetic-to-real domain gap

Every training and test image in Grasp-Anything++ is Stable-Diffusion-generated. The paper documents its own generation failures (Fig. 3: lens-less sunglasses, malformed scissors). Real camera photographs have different pixel statistics — specular highlights, sensor noise, real depth of field.

Observed: two real photographs failed; a freshly-generated AI image (unseen, but stylistically in-distribution) succeeded on the knife-handle prompt. This is consistent with a domain-shift explanation, though n is far too small to be conclusive.

The paper's own real-robot validation (Section 5.1) used 20 curated physical objects with a fixed camera, not arbitrary photographs. Generalizing to any internet image was never the claim.

### Measured performance is below the paper's figure

187/505 = **37.0%** on seen-test with the released checkpoint (95% CI approx [32.8%, 41.2%]). Paper reports 0.48, which falls outside that interval.

Candidate explanations, not distinguished: (a) the README's own caveat that reported numbers used a larger internal dataset; (b) `--split` sample selection differing from the authors' protocol; (c) a genuine reproduction gap. Honest framing for the write-up is "measurably below the paper's claim, under conditions not guaranteed identical" — not "the paper is wrong".

### CPU inference is slow even for demos

~55 s/image with DDIM50 respacing (1000 steps → 50). Full 1,363-sample evaluation is a multi-day job locally. Any large evaluation should be run on the rented GPU while it is available.

---

## Immediate next actions

1. **Rent a GPU** — this unblocks everything else
2. Train with `contrastive_loss_type = 'eq4'`, checkpoint regularly
3. Train a second run with `'nce'` for the comparison
4. Pull checkpoints back; re-run `experiments/language_grounding_test/` (knife/fork/cup, duck/apple) against the trained model — does the box move when the prompt changes?
5. Full quantitative eval on both, seen and unseen splits
6. Write up: fixes applied, before/after grounding behaviour, Eq. 4 vs NCE

---

## Deferred / optional

- Modules 3–6 of the math derivation (contrastive term, denoising score, Proposition 1, total loss). Not blocking implementation; needed to justify the Eq. 4 choices rigorously in writing.
- Numerically evaluating whether Proposition 1's bound is vacuous (`C = beta^-2` under the cosine schedule at T=1000).
- Baseline configs for the Table 9 parameter-count question.
- Full trace of `p_mean_variance` for the `gt` question.
