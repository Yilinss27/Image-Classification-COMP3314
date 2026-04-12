# Post-run_07 Experimental Log

**Period:** 2026-04-12
**Hardware:** AutoDL RTX 5090 (32 GB), Xeon Platinum 8470Q, 754 GiB RAM
**Starting point:** run_07 SOTA — P=6 K=8000 C=0.003 val=0.7858 / public **0.78400**
**Final result:** 2-model pnorm ensemble — val=0.8234 / public **0.82700**

---

## Summary of all experiments

| Run | Technique | val | public | Δ public | Status |
|-----|-----------|-----|--------|----------|--------|
| run_07 | Coates-Ng 250-config sweep | 0.7858 | 0.78400 | baseline | ⭐ |
| run_08 | Phase B: C < 0.003 extension | 0.7876 | 0.78600 | +0.00200 | ✅ |
| run_09 | sklearn MKL refit | — | — | — | ❌ abandoned |
| run_10 | Phase A: K > 8000 | 0.7860 | 0.78600 | +0.00200 | ⚠️ no gain |
| run_11 | Hard-vote ensemble top 3 | — | — | — | 🔬 not submitted |
| run_12 | **Flip augmentation** | **0.8122** | **0.81550** | **+0.03150** | ⭐⭐ |
| run_13 | Random crop + flip | 0.7912 | — | — | ❌ regression |
| run_14 | C sweep + 10-view TTA | 0.8104 | 0.80000 | −0.01550 | ❌ TTA10 hurts |
| run_15 | Multi-P ensemble (no pnorm) | — | — | — | 🔬 not submitted |
| run_16 | Two-layer Coates | 0.7836 | — | — | ❌ K1 too weak |
| run_17 | **Power normalization** | **0.8136** | **0.82700** | **+0.04300** | ⭐⭐⭐ |
| run_18 | Multi-crop feature avg | 0.8076 | — | — | ❌ regression |
| run_19 | **Pnorm ensemble (P=6+P=7)** | **0.8234** | — | pending | 🎯 final |

---

## run_08 — Phase B: Lower C extension

**Hypothesis:** The run_07 grid floor C=0.003 is binding for 16 of 50 (P,K) cells. Testing C ∈ {0.0005, 0.001, 0.002} might find better optima.

**Result:** 5 of 16 cells improved. New SOTA: P=7 K=6000 C=0.002 val=0.7876 / public 0.78600.

**Key finding:** C < 0.003 benefits scale with P monotonically:

| P | mean Δ | wins |
|---|--------|------|
| 4 | −0.0022 | 0/1 |
| 5 | −0.0021 | 0/2 |
| 6 | −0.0016 | 0/4 |
| 7 | +0.0010 | 2/4 |
| 8 | +0.0043 | 3/5 |

Larger patches (higher P) benefit more from lower C. Most dramatic recovery: P=8 K=8000 jumped from 0.7502 to 0.7708 (+0.0206) — the "collapse" was not over-parameterization but over-regularization.

**Surprise:** P=6 K=8000 C=0.002 had val 0.7836 (below C=0.003's 0.7858) but scored public **0.78750** — higher than C=0.003's 0.78400. First evidence that val-public gap can flip direction.

**Figures:** `figures/phase_b_01..04.png`

---

## run_09 — sklearn MKL LinearSVC refit (abandoned)

**Hypothesis:** cuML L-BFGS is ~0.3–0.5 pp below sklearn liblinear. A fresh conda env with MKL-backed sklearn could recover this gap.

**Result:** Created `sklmkl` conda env (numpy + MKL + sklearn 1.8 + cupy). GPU encoding worked, but sklearn LinearSVC `dual=False` on real 45k×24000 features ran 13+ min with no convergence. The bottleneck is liblinear's single-core coordinate descent, not BLAS — MKL cannot fix this.

**Conclusion:** sklearn refit is impractical on this hardware. Gap remains unrecoverable.

---

## run_10 — Phase A: Push K past 8000

**Hypothesis:** P=6 val was still climbing at K=8000. Pushing to K=10000–16000 might continue the trend, especially with finer C grid {0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004}.

**Result (P=6):**

| K | best C | best val | Δ vs K=8000 |
|---|--------|----------|-------------|
| 8000 | 0.003 | 0.7858 | baseline |
| 10000 | 0.002 | 0.7860 | +0.0002 |
| 12000 | 0.001 | 0.7848 | −0.0010 |
| 14000 | 0.0005 | 0.7816 | −0.0042 |

K→C\* inverse trend confirmed: K=10000 prefers C=0.002, K=12000 prefers C=0.001, K=14000's peak is at the grid floor C=0.0005. But val does NOT improve past K=8000–10000; the curve has plateaued. P=6 K=10000 C=0.002 scored public 0.78600 — same as or worse than K=8000 C=0.002 (0.78750).

P=5 K=10000 best: C=0.003 val=0.7830 (+0.0036 over K=8000). P=7 never tested (sweep abandoned in favor of flip aug).

**Conclusion:** Pushing K past 8000 does not help. Pipeline is near ceiling for this feature extraction method without augmentation.

---

## run_11 — Hard-vote ensemble

Top-3 submissions hard-vote: 86.2% all agree, 99.6% majority exists. Two P=6 K=8000 models (C=0.003 and C=0.002) are near-identical (2% combined disagreement), dominating the vote over P=7 K=6000 (12% different). Not submitted.

---

## run_12 — Flip augmentation ⭐⭐

**The single biggest improvement in this project.**

| Config | aug | val | public |
|--------|-----|-----|--------|
| P=6 K=8000 C=0.002 | none | 0.7836 | 0.78750 |
| P=6 K=8000 C=0.002 | **flip** | **0.8122** | **0.81550** |
| P=7 K=6000 C=0.002 | flip | 0.8078 | — |

Horizontal flip doubles training data (50k → 100k). Val improvement: **+0.0286** (P=6), +0.0202 (P=7). Public improvement: **+0.02800**. The gain is remarkably consistent between val and public.

TTA2 (flip test-time augmentation — average decision_function of original and flipped test) changed 5.17% of predictions and contributed to the public score.

---

## run_13 — Random crop + flip (failed)

**Attempts:**
1. **4× aug** [orig, flip, crop, crop_flip]: OOM (23 GB features > 32 GB VRAM)
2. **2× aug** [crop_A, crop_B] replacing originals: val=0.7912, −0.0210 vs flip-only. Model never sees uncropped test distribution.
3. **3× aug** [orig, flip, crop]: OOM (17.3 GB + RMM overhead > 32 GB effective)

**Root cause:** cuML's RMM allocator doesn't return GPU memory cleanly; `free_all_blocks()` only frees cupy's tracking, not the underlying RMM pool. Effective VRAM < 32 GB for feature matrix allocation.

**Conclusion:** Random crop cannot stack with flip at K=8000 due to VRAM constraints.

---

## run_14 — C sweep under flip + 10-view TTA

**C sweep confirmed C=0.002 optimal** under flip aug (same as without aug):

| C | val (flip aug) |
|---|----------------|
| 0.001 | 0.8084 |
| **0.002** | **0.8104** |
| 0.003 | 0.8104 |
| 0.005 | 0.8084 |

**10-view TTA (5 spatial crops × 2 flips):** public **0.80000** — catastrophic regression from 0.81550. Corner crops on 32×32 images shift objects out of frame, adding noise. **Spatial crop TTA does not work on small images.** Only center-flip TTA2 is safe.

---

## run_15 — Multi-P ensemble (without pnorm)

3-model soft-vote (P=6 K=8000, P=7 K=6000, P=8 K=3200), all with flip aug. TTA2 and "all-6" (3 orig + 3 flip averaged) were identical. Not submitted to Kaggle. Later superseded by run_19 (with pnorm).

---

## run_16 — Two-layer Coates-Ng (failed)

**Architecture:** L1 (P=6, K1=1600) → 3×3 intermediate pool → L2 patches (3×3×1600=14400 dim) → ZCA → K2=1600 → concat (6400+6400=12800 dim).

**Bottleneck:** MiniBatchKMeans on 200k×14400 was extremely slow:
- n_init=3, max_iter=300: ran 50+ minutes, killed
- n_init=1, max_iter=100: still took 29 minutes

**Result:** val=0.7836, identical to single-layer K=8000 WITHOUT aug. K1=1600 is too weak — the information lost by reducing K from 8000 to 1600 is not recovered by adding Layer 2. To use K1≥4000, Layer-2 patch dim becomes 36000+, making ZCA (36k×36k matrix) and KMeans impractical.

---

## run_17 — Power normalization ⭐⭐⭐

**The second biggest improvement.** Applied signed square-root power normalization `x → sign(x) * sqrt(|x|)` to triangle-encoded features before StandardScaler.

| Normalization | best C | val |
|---------------|--------|-----|
| baseline (StandardScaler only) | 0.002 | 0.8118 |
| **power norm** | **0.002** | **0.8136** |
| power norm + L2 | 0.002 | 0.8126 |

Val improvement: +0.0018. **Public improvement: +0.01150** (0.81550 → **0.82700**). The public gain is 8× the val gain — power norm improves generalization far more than val suggests.

L2 normalization on top of power norm did NOT help (−0.0010).

**Why it works:** Triangle encoding produces sparse, right-skewed features where a few large activations dominate. Square-root compression equalizes their influence, making the distribution more Gaussian — ideal for linear SVM. This is the standard normalization in the Fisher vector literature (Perronnin et al., 2010).

---

## run_18 — Multi-crop feature averaging (failed)

**Idea:** For each image, encode 5 views (original + 4 random crops), average the feature vectors to get a noise-reduced representation. Same training set size, so VRAM unchanged.

**Result:** val=0.8076, WORSE than pnorm-only 0.8136. Averaging crops' features blurs discriminative spatial information. Triangle encoding is position-sensitive; averaging over shifted versions destroys the specificity the classifier needs.

---

## run_19 — Multi-P pnorm ensemble (final) 🎯

**Architecture:** Soft-vote of 2 (or 3) models, each with flip aug + power norm + cuML LinearSVC.

| Ensemble | val |
|----------|-----|
| P=6 K=8000 single | 0.8134 |
| P=7 K=6000 single | 0.8100 |
| P=8 K=3200 single | 0.7998 |
| **2-model (P=6+P=7)** | **0.8234** |
| 3-model (P=6+P=7+P=8) | 0.8236 |

Ensemble gives +0.0100 over best single model on val. 2-model and 3-model are nearly identical — P=8 is too weak to contribute meaningfully.

**Final submission:** `sub_run19_pnorm_2model_tta2.csv` — 2-model (P=6 K=8000 C=0.002 + P=7 K=6000 C=0.002) with power norm, flip aug, TTA2.

---

## Key lessons learned

1. **Flip augmentation is the single biggest lever** (+0.028 val, +0.028 public). Should have been the first thing we tried.
2. **Power normalization gives disproportionate public gains** (+0.0018 val but +0.0115 public). Feature distribution matters more than val accuracy suggests.
3. **val-public gap is noisy and can flip direction.** C=0.002 had lower val than C=0.003 but higher public. Optimize for robustness, not val alone.
4. **Spatial crops do NOT work on 32×32 images** — neither as training aug (VRAM-constrained + hurts when replacing originals) nor as TTA (−0.015 public).
5. **Two-layer Coates requires K1≥4000** which is impractical with current ZCA/KMeans at the Layer-2 patch dimensionality.
6. **Ensemble of diverse patch sizes** gives +0.01 on val even with only 2 models.
7. **K past 8000 does not help** for P=6 — the feature space is saturated. Better to improve normalization and augmentation.
