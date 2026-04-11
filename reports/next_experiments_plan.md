# Next-Batch Experiment Plan (run_08)

**Goal:** drive the Coates-Ng cuML pipeline to its global optimum. Current SOTA: `P=6 K=8000 C=0.003` val=0.7858 / public 0.78400.

---

## 1. Quantitative motivation from run_07

### 1.1 Slopes at the K=8000 ceiling (best-C per (P, K))

| P | K=6000 | K=8000 | Slope (per 1000 K) | Linear extrapolation K=10000 | Status |
|---|--------|--------|---------------------|------------------------------|--------|
| 4 | 0.7666 | 0.7686 | **+0.0010** | 0.7706 | slow, near plateau |
| 5 | 0.7788 | 0.7794 | **+0.0003** | 0.7800 | essentially flat, done |
| 6 | 0.7780 | **0.7858** | **+0.0039** | **0.7936** | 🚀 still climbing fast |
| 7 | 0.7810 | 0.7752 | −0.0029 | 0.7694 | turned over, don't extend |
| 8 | 0.7750 | 0.7502 | −0.0124 | 0.7254 | collapsed, definitely don't extend |

**Lesson:** **P=6 is the only curve that's still climbing fast** at the right edge of the current grid. Everyone else either turned over or nearly flat.

### 1.2 The C=0.003 floor is binding

16 of 50 (P, K) cells have C=0.003 (the grid floor) as their winner. These are almost entirely the top performers:

```
P=6 K=8000 C=0.003 → 0.7858 ⭐    P=7 K=4000 C=0.003 → 0.7824
P=8 K=3200 C=0.003 → 0.7816       P=7 K=6000 C=0.003 → 0.7810
P=5 K=8000 C=0.003 → 0.7794       P=6 K=4000 C=0.003 → 0.7786
P=6 K=6000 C=0.003 → 0.7780       ... (16 total)
```

All of these are **at the grid edge**, so the true optimum C\* for these cells may be below 0.003.

### 1.3 C sensitivity near the top (K=8000)

```
               C=0.003  C=0.005  C=0.01
P=6 K=8000:    0.7858   0.7842   0.7780
P=5 K=8000:    0.7794   0.7768   0.7714
P=4 K=8000:    0.7686   0.7656   0.7612
```

Val drops smoothly as C grows. Extrapolating the curve toward C=0.001 is speculative, but any gain here is essentially "free" because lower C costs nothing in run time.

### 1.4 What we never varied

| Knob | Run_07 value | Sensitivity |
|------|--------------|-------------|
| Pool | 2×2 (fixed)  | **Coates & Ng 2011 reports pool variation gives +2 pp on CIFAR-10.** Never tried 3×3 or 4×4. |
| ZCA ε | 0.1 (fixed)  | Paper reports sensitivity; we never checked. |
| Stride | 1 (fixed)    | Paper says stride=1 is best. Unlikely to help. Skip. |
| n_patches (K-means) | 1M (fixed) | For K=16000 we'd want 2M (∼125 patches/cluster). |
| Classifier | cuML L-BFGS | Known ~0.3–0.5 pp below sklearn liblinear. |

---

## 2. Proposed experiments, ranked by expected-gain / cost

### Phase A — Push K past 8000 for P=6 (and P=5 sanity) [HIGHEST PRIORITY]

**Rationale:** P=6's slope is +0.0039 per 1000 K, still climbing. No turnover in the grid.

**Grid:**
```
P = 6,              K ∈ {10000, 12000, 14000, 16000}, C ∈ {0.001, 0.003, 0.005}
P = 5 (sanity),     K ∈ {10000, 12000},                C ∈ {0.003, 0.005}
```
= 4×3 + 2×2 = **16 configs**

**Settings:** `n_patches = 2_000_000` (2× current, needed for K≥12000 to keep ≥125 patches/cluster), `pool=2×2`, `batch_size=64` (for K=16000 to avoid the 8 GB single-allocation crash that killed run_07 at P=4 K=6000 with batch_size=400).

**Risk:** cuML might not converge cleanly for K=16000. Also disk: K=16000 P=6 features = 64000 × 50000 × 4 = 12.8 GB train + 2.56 GB test. Need to delete run_07 features first (we already cleaned those during run_07 anyway).

**Expected outcome:**
- P=6 K=10000 ≈ 0.788–0.790 (linear projection 0.7936 is an upper bound; curvature will likely reduce it)
- P=6 K=12000 ≈ 0.789–0.792
- P=6 K=16000 uncertain — could plateau or very slowly climb

**Best case:** +0.5–1.0 pp on val (hitting ~0.79+), +0.3–0.6 pp on public.

**Budget:** Encode time is ~3 min per K (GPU is fast); one LinearSVC fit at K=16000 is ~3–5 min. Total ≈ **90 min** wall clock.

---

### Phase B — Extend C downward for proven-good cells [HIGH PRIORITY]

**Rationale:** C=0.003 is the grid floor and wins 16 of 50 cells — the true optimum may lie below.

**Grid:** run only on cells where the run_07 winner was C=0.003, i.e. the cells listed in §1.2 above. Plus the new Phase A big-K cells.

```
(P, K) ∈ {  (4, 8000),
            (5, 4000), (5, 8000),
            (6, 2400), (6, 4000), (6, 6000), (6, 8000),
            (7, 1600), (7, 4000), (7, 6000), (7, 8000),
            (8, 2400), (8, 3200), (8, 4000), (8, 6000), (8, 8000)  }
C ∈ {0.0005, 0.001, 0.002}
```
= 16 (P,K) × 3 C = **48 configs**. Features are all cached from run_07 — **no encoding cost**, only LinearSVC fits.

**Budget:** LinearSVC fits take 30–150 s each depending on K. 48 × 90 s ≈ **75 min**.

**Expected outcome:** If C* is exactly 0.003, no gain — confirms the current optimum. If C* lies below (more likely), +0.1–0.3 pp.

---

### Phase C — 3×3 pool on top P=6 cells [POTENTIALLY HUGE PAYOFF]

**Rationale:** The biggest lever we never pulled. Coates & Ng 2011 explicitly tested `pool ∈ {2, 3, 4}` and saw pool=3×3 gave +1–2 pp over pool=2×2 on CIFAR-10, at the cost of 9K-dim features instead of 4K.

**Grid:**
```
P = 6, K ∈ {4000, 6000, 8000}, pool = 3, C ∈ {0.001, 0.003, 0.005}
```
= 3 × 3 = **9 configs**

**Implementation cost:** `run_07_cuml_sweep.py` already supports `--pool`; the encoder supports arbitrary pool. Just need to verify pool=3 splits work (they should — the encoder divides `nh` by `pool` cleanly).

**Compute cost:**
- Feature dim = 9K = 72000 at K=8000 → 45000 × 72000 × 4 = **12.3 GB per scaled copy**. Fits in the 754 GB RAM box trivially.
- GPU encode memory: batch_size=64 → 64 × 29 × 29 × 8000 × 4 = 1.7 GB per batch, safe.
- Time per fit: probably 2× slower than pool=2 (2.25× features) → 5–10 min each.
- Disk: K=8000 P=6 pool=3 features = 72000 × 50000 × 4 = 14.4 GB. Delete K=8000 P=4/5/6 pool=2 features from previous runs first.

**Budget:** 9 × 8 min ≈ **~75 min** wall clock.

**Expected outcome:** If the Coates-Ng +1–2 pp holds, val could hit **0.795–0.805**. This is probably the single biggest remaining gain.

---

### Phase D — 4×4 pool on the top-2 P=6 cells [SPECULATIVE]

**Grid:** `P=6`, `K ∈ {6000, 8000}`, `pool=4`, `C ∈ {0.001, 0.003}` = 4 configs.

Only run if Phase C shows 3×3 > 2×2. 4×4 features = 16K × K_dict = 128000 dim @ K=8000. Still feasible but slow (~15 min per fit).

**Budget:** 4 × 15 min ≈ **60 min**.

---

### Phase E — ZCA ε sensitivity sanity check [LOW PRIORITY]

**Grid:** `P=6 K=8000 C=0.003 pool=2`, ε_ZCA ∈ {0.01, 0.03, 0.1, 0.3, 1.0}. = 5 configs.

**Budget:** ~15 min. Worth running as a single quick script to rule out easy wins here.

---

### Phase F — sklearn liblinear refit of top 5 cuML winners [CHEAPEST POSSIBLE WIN, BLOCKED]

**Rationale:** We have hard evidence that cuML L-BFGS lands ~0.3–0.5 pp below sklearn liblinear at the same (P, K, C). This is a **deterministic free gain** — same features, stronger classifier.

**Blocker:** the AutoDL Xeon + OpenBLAS "Haswell" build makes sklearn LinearSVC hang indefinitely. We'd need either:
1. A different cloud box where sklearn works
2. Rebuild OpenBLAS with Sapphire Rapids ISA (risky, could break cupy)
3. Use the local WSL2 box (slower but sklearn works there) — transfer the 6 top feature caches (~24 GB)

**Grid:** top 5 (P, K, C) from run_07 Phase A if it produces new winners, else from run_07 top 5:
1. P=6 K=8000 C=0.003 (current SOTA)
2. P=6 K=8000 C=0.005
3. P=7 K=4000 C=0.003
4. P=8 K=3200 C=0.003
5. P=7 K=6000 C=0.003

Plus any top Phase A/C winners.

**Budget:** 5 fits × 3 min (with working sklearn) = ~15 min *once the blocker is resolved*.

**Expected outcome:** **+0.3–0.5 pp uniformly** across the top. This alone would push val from 0.7858 → ~0.789–0.791 on P=6 K=8000 — roughly matching what Phase A/C might deliver, but with zero stochasticity.

---

## 3. Recommended execution order

1. **Phase B** first (cheapest, no encoding, 75 min) — confirms whether C<0.003 helps at all. Sets C-grid for the rest.
2. **Phase A** (90 min) — the big K push for P=6. Uses Phase B's best C.
3. **Phase C** (75 min) — 3×3 pool on top P=6 cells. Biggest speculative payoff.
4. **Phase D** (60 min) — only if Phase C shows >0.1 pp gain over 2×2.
5. **Phase E** (15 min) — quick ε sensitivity sanity check, run in parallel with anything.
6. **Phase F** — fix the sklearn block first (maybe on local WSL2), then run the top-5 refit.

Total expected compute: **~4–5 hours** on the 5090 (Phases A/B/C/D), + separate box for F.

---

## 4. Implementation notes

### Script changes needed
- `run_07_cuml_sweep.py` already supports `--patch-list`, `--k-list`, `--c-list`, `--pool`, `--n-patches`, `--batch-size`, `--tag`. For Phase C, add `--pool-list` or just run separate invocations per pool value.
- Cache filenames include `K` and `P` but **not** `pool`, `n_patches`, `ZCA eps`. Phase C (pool=3) and Phase E (ε≠0.1) would clobber existing caches unless we extend the cache naming. **Add `pool` and `eps` to cache filename to be safe.**
- Phase F needs a refit-only script that loads existing features and runs sklearn LinearSVC.

### Disk management
Run_07 features for P=4..8 were mostly deleted already. Current cache on remote is ~12 GB. Phase A K=16000 P=6 adds 12.8+2.56 = ~15 GB. Phase C K=8000 P=6 pool=3 adds ~14.4+2.88 = ~17 GB. We should periodically purge old caches.

### Risk of cuML breaking on K ≥ 12000
cuML's L-BFGS has been noisy at high K (we saw "line search failed" warnings often). K=16000 might fail to converge or give bad numbers. Mitigation: widen `max_iter` to 5000, try `tol=1e-5`.

---

## 5. Success criteria

- **Phase A success**: find any (P=6, K, C) with val ≥ 0.790 (conservatively, beats current 0.7858 by +0.4 pp).
- **Phase C success**: find any pool=3 config with val ≥ 0.790.
- **Overall target**: val ≥ 0.795 → public ≥ 0.79 (0.6 pp over current 0.78400).

If we hit val ≥ 0.80 we should stop and submit everything — that would be a dramatic gain and probably near the ceiling of this pipeline on this dataset.
