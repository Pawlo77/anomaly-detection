# Evaluating Outlier Detection Algorithms

---

## Conventions and Prerequisites

Throughout this project, I'll use `RobustScaler` (median/IQR) for all preprocessing unless noted otherwise. No label information enters any fitting step at any point. I'll refer to algorithms by their PyOD class names for reproducibility: `OCSVM`, `IForest`, `LOF`, `DBSCAN` (via scikit-learn), `ECOD`, `HBOS`.

All experiments are seeded at `random_state=42`, with five additional seed repeats (seeds 0, 1, 2, 3, 42) to estimate variance.

**Key references:**
- Rayana (2016) — ODDS benchmark corpus
- Gagolewski et al. (2022, DOI:10.1016/j.softx.2022.101270) — 2D clustering suite
- Li et al. (2022, arXiv:2104.01422) — model selection properties

---

## 1. Datasets

### 1.1 ODDS Multidimensional Point Datasets

I'll use the ODDS-aligned ADBench Classical mirror of **26** multidimensional point datasets (subset of Rayana, 2016). The table below summarizes each dataset's properties and the analytical role it plays in my experiments; physical dimensions mirror the shipped npz files (some differ slightly from textbook ODDS excerpts).

| Dataset | n | d | Contamination (%) | Notes |
|---|---|---|---|---|
| `annthyroid` | 7,200 | 6 | 7.4 | Low-d, mid-contamination, large n |
| `arrhythmia` | 452 | 274 | 14.6 | HDLSS; requires dedicated preprocessing (§1.3) |
| `breastw` | 683 | 9 | 35.0 | Extreme contamination — stresses prior-dependent methods |
| `cardio` | 1,831 | 21 | 9.6 | Medium-d, clean structure |
| `cover` | 286,048 | 10 | 0.96 | Extreme n; algorithm-specific subsampling applies (§1.4) |
| `glass` | 214 | 7 | 4.2 | Small n, low contamination; expect high bootstrap variance |
| `http` | 567,498 | 3 | 0.39 | Extreme n, near-zero contamination; subsampling applies |
| `ionosphere` | 351 | 33 | 35.9 | High-d, extreme contamination |
| `letter` | 1,600 | 32 | 6.25 | Medium-d, balanced |
| `lympho` | 148 | 18 | 4.1 | Very small n — report ±1 std across 10 bootstrap resamples |
| `mammography` | 11,183 | 6 | 2.3 | Large n, very sparse outliers |
| `mnist` | 7,603 | 100 | 9.2 | High-d, image-derived; good for reconstruction-error methods |
| `musk` | 3,062 | 166 | 3.2 | Very high-d; expect distance degradation |
| `optdigits` | 5,216 | 64 | 2.9 | High-d, structured latent space |
| `pendigits` | 6,870 | 16 | 2.3 | Medium-d, multiclass-derived |
| `pima` | 768 | 8 | 34.9 | Extreme contamination, medical |
| `satellite` | 6,435 | 36 | 31.6 | Extreme contamination, remote sensing |
| `satimage-2` | 5,803 | 36 | 1.2 | Very sparse outliers |
| `shuttle` | 49,097 | 9 | 7.2 | Large n, low-d; subsampling applies |
| `smtp` | 95,156 | 3 | 0.03 | Near-zero contamination — accuracy metric is especially misleading here (§3.1) |
| `speech` | 3,686 | 400 | 1.65 | Ultra-high-d (d=400) — distance metrics will collapse (§4.1) |
| `thyroid` | 3,772 | 6 | 2.5 | Low-d, clean reference |
| `vertebral` | 240 | 6 | 12.5 | Small n, low-d |
| `vowels` | 1,456 | 12 | 3.4 | Low contamination, structured |
| `wbc` | 223 | 9 | 4.48 | ADBench manifest; leukocyte screening |
| `wine` | 129 | 13 | 7.8 | Smallest n — evaluation metrics unstable, use bootstrap |

**Cross-dataset analytical strata (used in §4):**

- **Curse-of-dimensionality probes:** `arrhythmia` (d=274), `musk` (d=166), `speech` (d=400), `mnist` (d=100)
- **Prior-violation probes (contamination ≥ 30%):** `breastw`, `ionosphere`, `pima`, `satellite`
- **Sparse-anomaly probes (contamination < 2%):** `smtp`, `http`, `satimage-2`, `mammography`
- **Small-n probes (n < 250):** `wine`, `lympho`, `glass`, `vertebral`

---

### 1.2 Gagolewski 2D Datasets

I'll use the following 2D datasets from Gagolewski et al. (2022, v1.1.0), referenced as `battery/dataset`. All are in ℝ².

| Dataset | n | k | Why I'm using it |
|---|---|---|---|
| `wut/smile` | 1,000 | 6 | Non-convex arcs with isolated points between arcs — local outliers invisible to global methods |
| `wut/circles` | 4,000 | 4 | Concentric rings; points inside rings are structural anomalies that DBSCAN should catch but OCSVM will miss |
| `wut/isolation` | 9,000 | 3 | Three clusters with a large void — sanity check; all six methods should perform well here |
| `wut/windows` | 2,977 | 5 | Frame-shaped clusters; outliers appear inside the frames — proximity-based methods will misjudge them |
| `wut/x2` | 120 | 3/4 | Contains **10 explicitly labeled noise points** — the only Gagolewski dataset where I can compute quantitative metrics directly |
| `sipu/compound` | 399 | 6 | Mixed density (tight blobs next to elongated clusters) — classic LOF failure case when k is set globally |
| `sipu/jain` | 373 | 2 | Two interleaved crescents; points in the concave gap are locally dense but globally anomalous |
| `sipu/flame` | 240 | 2 | Contains **12 labeled noise points**; fractal-like boundary breaks smooth decision-surface methods |
| `sipu/spiral` | 312 | 3 | Three interleaved spirals; only DBSCAN with the right `eps` detects interstice anomalies |
| `fcps/lsun` | 400 | 3 | L-shaped clusters — tests non-spherical geometry, which will break linear OCSVM |
| `graves/fuzzyx` | ~600 | var | Fuzzy overlapping clusters; tests anomaly score calibration under boundary ambiguity |

**Hypotheses I'm testing on specific 2D datasets:**

- **`wut/circles`:** DBSCAN (correct `eps`) detects intra-ring anomalies. OCSVM-RBF fits a smooth ellipse around all rings combined and misses interior points. IForest also misses them — intra-ring points aren't easily isolated by random partitions.
- **`sipu/compound`:** LOF with globally tuned k assigns low anomaly scores to points in the low-density elongated cluster because their neighborhood is consistent with that cluster's own density. IForest correctly isolates truly sparse points regardless.
- **`sipu/spiral`:** DBSCAN with optimal `eps` labels interstice points as noise. LOF fails when k is large enough to span multiple spiral arms. OCSVM-RBF fits a global convex hull.
- **`wut/isolation`:** All six methods should perform well — this is the sanity check.
- **`fcps/lsun`:** Algorithms assuming spherical geometry (OCSVM-RBF, LOF with Euclidean metric) will incorrectly score normal points in the L-shaped arms as outliers.

**Visual protocol for all 2D datasets** — I'll produce a 4-panel figure per algorithm per dataset:
1. Raw scatter colored by ground truth cluster label
2. Anomaly score as a continuous contour overlay (diverging colormap, centered at the natural contamination threshold)
3. Binary predictions at the dataset's natural contamination rate
4. PR curve with AP annotated

---

### 1.3 Preprocessing Sub-Pipeline for `arrhythmia`

`arrhythmia` requires dedicated preprocessing because it contains approximately 0.33% missing values and includes nominal (categorical) attributes alongside continuous ECG features. Passing NaN values to any algorithm will cause runtime errors.

**Mandatory sequence (applied only to `arrhythmia`):**

1. **Missing value imputation:** Use `IterativeImputer` with `BayesianRidge` as the estimator, max 10 iterations. I'm not using simple median imputation because ECG features are strongly correlated, and marginal imputation would distort joint relationships. I'll document the fraction of values imputed per feature.

2. **Categorical encoding:** One-hot encode all nominal features. This expands d from 274 to approximately 280+. I'll document the exact post-encoding dimensionality.

3. **Post-encoding PCA:** Apply PCA retaining 95% of variance before running LOF and DBSCAN. For OCSVM, IForest, ECOD, and HBOS, I'll run on both raw and PCA-reduced data and report both results.

---

### 1.4 Algorithm-Specific Subsampling Protocol

A uniform n=50,000 cap applied to all algorithms on large datasets is not defensible, because algorithms have fundamentally different computational complexity. I'll instead apply the following:

| Algorithm | Time Complexity | Subsampling? | Affected Datasets |
|---|---|---|---|
| IForest | O(n log n) | **No** — full data | All |
| ECOD | O(nd log n) | **No** — full data | All |
| HBOS | O(nd log n) | **No** — full data | All |
| OCSVM | O(n² to n³) | **Yes**, cap at n=20,000 | `cover`, `http`, `smtp`, `shuttle` |
| LOF | O(n²) / O(n log n) with BallTree | **Yes**, cap at n=20,000 | `cover`, `http`, `smtp`, `shuttle` |
| DBSCAN | O(n log n) / O(n²) without index | **Yes**, cap at n=20,000 | `cover`, `http`, `smtp`, `shuttle` |

Running IForest, ECOD, and HBOS on the full dataset and reporting runtime is itself a contribution — it demonstrates their industrial scalability. For the others, I'll apply stratified random subsampling (preserving contamination rate) and include a runtime comparison table.

**Runtime table format (included in the notebook):**

| Algorithm | Dataset | n_used | Runtime (s) | Notes |
|---|---|---|---|---|
| IForest | http | 567,498 | — | Full data |
| LOF | http | 20,000 | — | Subsampled |
| ... | ... | ... | ... | ... |

---

## 2. The Six Algorithms

### 2.1 One-Class SVM (OCSVM)

**How it works:** Maximum-margin boundary in kernel feature space (Schölkopf et al., 2001). The dual formulation finds the hyperplane with maximum distance from the origin that separates most training data from the origin in kernel space. The parameter `nu` is simultaneously an upper bound on the outlier fraction and a lower bound on the fraction of margin errors — setting `nu` significantly below the true contamination rate structurally caps recall.

**Known failure mode:** In high-d spaces, the RBF kernel K(x,z) = exp(−γ‖x−z‖²) collapses — too small a γ makes all kernel values approach 1; too large a γ makes them all approach 0. To show this on `speech` (d=400), I'll plot the distribution of pairwise kernel values K(xᵢ, xⱼ) at each γ value. When the distribution concentrates near 0 or 1, the kernel has effectively collapsed.

**Hyperparameter grid:**

```python
ocsvm_grid = {
    "kernel":       ["rbf", "poly", "sigmoid", "linear"],
    "nu":           [0.01, 0.05, 0.10, 0.20, 0.35, 0.50],
    "gamma":        ["scale", "auto", 0.001, 0.01, 0.1, 1.0, 10.0],  # rbf/poly/sigmoid only
    "degree":       [2, 3, 4],   # poly only
    "coef0":        [0.0, 1.0],  # poly/sigmoid only
}
# Primary sweep: kernel=rbf, nu × gamma = 6 × 7 = 42 configurations
# Full combinatorial ~350 configs → sample 80 via Latin Hypercube Sampling
```

I'll use SALib to compute Sobol first-order sensitivity indices and verify that `nu` dominates PR-AUC variance on extreme-contamination datasets.

---

### 2.2 Isolation Forest (IForest)

**How it works:** Random recursive partitioning (Liu, Ting & Zhou, 2008). Anomalies require fewer splits to isolate because they are "few and different." The normalized score in (0,1) is based on expected path length relative to a balanced BST of n samples. Random splits on uninformative features contribute a constant expected path length — they add noise but not systematic bias, making IForest theoretically robust to irrelevant dimensions.

**Known failure mode:** Local outliers embedded within moderately dense regions won't be isolated faster than nearby normal points. I expect to demonstrate this on `sipu/compound` and `wut/windows`.

**Hyperparameter grid:**

```python
iforest_grid = {
    "n_estimators":   [50, 100, 200, 500],
    "max_samples":    ["auto", 32, 64, 128, 256, 512],  # ψ — key parameter per Liu et al.
    "contamination":  [0.01, 0.05, 0.10, 0.20, 0.35, 0.50],
    "max_features":   [0.5, 0.75, 1.0],
    "bootstrap":      [True, False],
    "random_state":   [42],
}
```

**Convergence check:** I'll plot PR-AUC vs. `n_estimators` (50→500) for three representative datasets (one low-d, one high-d, one extreme-contamination). IForest typically stabilizes by 200 trees — I want to show this empirically and use it to justify the final `n_estimators` choice rather than just defaulting to 100.

**On `max_samples`:** Liu et al. argued ψ=256 is often sufficient. I'll verify this on `cover` and `http` by comparing PR-AUC at ψ=256 vs. ψ=2048 vs. ψ="auto". I expect near-identical performance, which is practically important when dealing with very large datasets.

---

### 2.3 Local Outlier Factor (LOF)

**How it works:** Local density ratio (Breunig et al., 2000). LOF compares the local reachability density of a point to the average local reachability density of its k-nearest neighbors. LOF ≈ 1 means normal; LOF >> 1 means locally anomalous. It is explicitly designed for local outliers.

**Known failure mode:** As dimensionality increases, the concentration-of-measure phenomenon causes all pairwise distances to converge — (d_max − d_min) / d_mean → 0. When all reachability distances are equal, all LOF scores converge to 1 and the method becomes blind. I'll quantify this directly in §4.1.

**Hyperparameter grid:**

```python
lof_grid = {
    "n_neighbors":    [5, 10, 15, 20, 30, 50, 75, 100],
    "metric":         ["euclidean", "manhattan", "minkowski", "cosine"],
    "p":              [1, 2, 3],    # Minkowski p only
    "contamination":  [0.01, 0.05, 0.10, 0.20, 0.35],
    "novelty":        [False],      # transductive use throughout
}
# Primary sweep: n_neighbors × metric = 8 × 4 = 32 core configurations
```

---

### 2.4 DBSCAN

**How it works:** Density-based spatial clustering (Ester et al., 1996). Anomalies are any points that are not core points and are not density-reachable from any core point. Unlike all other methods here, DBSCAN produces a hard binary output — no continuous score.

**Known failure mode:** DBSCAN assumes globally uniform density. On `sipu/compound` (mixed-density clusters), no single `eps` can simultaneously resolve tight and diffuse clusters. I'll use this as the primary illustration of the method's limitations.

**`eps` selection — data-driven, not a fixed grid:**

For each dataset:
1. Compute the sorted k-NN distance curve for k = (target `min_samples` − 1)
2. Identify the "knee" point (maximum curvature) as the data-driven `eps` anchor
3. Sweep ±50% around this anchor

```python
dbscan_eps_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # × eps_knee

dbscan_grid = {
    "eps":          "6 data-driven values per dataset (see above)",
    "min_samples":  [3, 5, 10, 15, 20, 30],
    "metric":       ["euclidean", "manhattan"],
    "algorithm":    ["auto"],
}
# 6 eps × 6 min_samples × 2 metrics = 72 configurations per dataset
```

**Continuous score for AUC computation:** Since DBSCAN is binary, I need to convert it to a continuous score to compute PR-AUC and ROC-AUC alongside the other methods. I'll define:

```python
def dbscan_score(X, labels, core_sample_indices_, eps):
    """
    Core points:   score = 0
    Border points: score = distance_to_nearest_core / eps  → in [0, 1]
    Noise points:  score = 1 + distance_to_nearest_core / eps  → always > 1
    """
```

This is monotonic, respects the DBSCAN decision structure, and enables consistent AUC computation. I'll justify this construction explicitly in the notebook.

---

### 2.5 ECOD

**How it works:** Empirical Cumulative Distribution-based Outlier Detection (Li et al., 2022, arXiv:2201.00382). ECOD estimates the empirical CDF of each feature independently, computes tail probabilities per dimension per point, and aggregates via log-sum. It is the only purely distributional method in my suite — no distances, no kernels, no random partitions.

**Known failure mode:** The independence assumption is the explicit testable weakness. On datasets where anomalies are outlying only in joint space (not individual marginal tails), ECOD assigns near-zero anomaly scores.

**Hyperparameter grid:**

```python
ecod_grid = {
    "contamination": [0.01, 0.05, 0.10, 0.20, 0.35, 0.50],
    # ECOD has no other tunable parameters — this is by design (Li et al., 2022).
}
```

Sweeping `contamination` only shifts the binary threshold, not the underlying score ranking. I'll report rank correlation of binary predictions across contamination values to confirm this, and use ECOD's bootstrap variance as the reference stability floor for all methods.

**Independence assumption test:** For each ODDS dataset, I'll compute mean absolute pairwise Pearson correlation ρ̄ and regress |PR-AUC(ECOD) − PR-AUC(LOF)| against ρ̄ across all mirrored ODDS-classical datasets. A positive slope is evidence that ECOD's independence assumption causes measurable degradation on correlated data.

---

### 2.6 HBOS

**How it works:** Histogram-Based Outlier Score (Goldstein & Dengel, 2012). HBOS(x) = Σₖ log(1/p_k(xₖ)) where p_k is the bin frequency of feature k. Like ECOD, it assumes feature independence. Unlike ECOD, it uses static bin widths rather than the exact ECDF — which means it can detect **density valleys** (interior low-frequency regions) rather than only marginal tails.

**Known failure mode:** When `n_bins` is too high and `alpha=0`, unseen bin values produce score = ∞, collapsing the ranking. The `alpha` regularization parameter smooths zero-count bins. I'll plot PR-AUC as a 2D heatmap of (`n_bins`, `alpha`) to locate the stable operating region.

**Scientific contrast with ECOD:** On datasets where the anomaly class occupies an interior low-density region of each marginal (not a tail), HBOS should outperform ECOD. `satellite` is a candidate — I'll test this hypothesis explicitly.

**Hyperparameter grid:**

```python
hbos_grid = {
    "n_bins":         [5, 10, 20, 30, 50, "auto"],  # "auto" uses sqrt(n) heuristic
    "alpha":          [0.0, 0.1, 0.2, 0.5],
    "tol":            [0.1, 0.5],
    "contamination":  [0.01, 0.05, 0.10, 0.20, 0.35],
}
# Primary sensitivity sweep: n_bins × alpha = 6 × 4 = 24 configurations
```

---

## 3. Evaluation Framework

### 3.1 Metrics

I'll report the following metrics for all datasets. All metrics are computed at a threshold that labels exactly ⌈contamination × n⌉ points as outliers, to eliminate threshold selection as a confound.

**Accuracy** = (TP + TN) / (TP + TN + FP + FN). I'll report this for completeness, but I'll also include an explicit cautionary demonstration: for `smtp` (0.03% contamination), a naive "all inlier" classifier achieves Accuracy ≈ **99.997%**. If my algorithm reaches 99.5%, it's strictly worse than predicting everything as normal. This calculation must appear in the notebook — not to dismiss accuracy, but to show exactly why it's meaningless on extreme imbalance.

**Precision** = TP / (TP + FP). Proportion of predicted outliers that are true outliers.

**Recall** = TP / (TP + FN). Proportion of true outliers that were detected.

**ROC-AUC** = P(score(outlier) > score(inlier)) for a randomly chosen pair. Invariant to class distribution — measures ranking capability independently of contamination rate. A classifier can have high ROC-AUC and near-zero PR-AUC on extreme imbalance; this isn't a paradox, it's information. I'll use ROC-AUC for cross-dataset method ranking (it's distribution-invariant) and PR-AUC for within-dataset operational assessment.

**PR-AUC (Average Precision):** Highly sensitive to contamination rate. The random baseline equals the contamination rate itself — achieving PR-AUC = 0.10 on `smtp` (0.03% contamination) is a massive success; on `ionosphere` (35.9% contamination) it's a failure.

**MCC (Matthews Correlation Coefficient):** Balanced treatment of all confusion matrix cells under extreme imbalance.

**F1:** At the natural contamination threshold.

---

### 3.2 Sensitivity Analysis

**Two-track evaluation protocol (addresses data leakage):**

- **Track A — Unsupervised execution (primary benchmark):** I'll run each algorithm using domain-standard default hyperparameters with no ground truth involved: IForest (n_estimators=200, max_samples=256), LOF (n_neighbors=20), DBSCAN (eps=eps_knee, min_samples=5), OCSVM (kernel="rbf", nu=0.1), ECOD and HBOS (defaults). These are the numbers reported as primary comparison results.

- **Track B — Oracle sensitivity analysis (secondary):** After revealing ground truth, I'll analyze the full hyperparameter grid retrospectively. Results are clearly labeled "oracle-optimal" and are not used to claim superiority over Track A. The question they answer: "How much performance do we leave on the table with default parameters?"

**Method 1 — Sobol Sensitivity Indices (SALib):**

First-order index Sᵢ = Var(E[Y|θᵢ]) / Var(Y) quantifies the fraction of PR-AUC variance attributable to parameter θᵢ alone. Total-order index Tᵢ captures interactions. For IForest (4 hyperparameters), N=200 gives 1,200 evaluations — tractable. I'll report results as a bar chart showing fraction of PR-AUC variance per hyperparameter.

**Method 2 — Bootstrap stability pass (Isolation Forest candidate set):**

A full oracle grid yields thousands of θ values; estimating bootstrap variance **for each θ globally** would dominate compute without changing the qualitative story I need (whether defaults sit in a brittle peak). Implementation therefore uses a **finite IForest-only candidate lattice**: several dozen deterministic tuples sampled from `(n_estimators × max_samples × contamination × max_features)`, each scored on **`thyroid`** with **10 stratified bootstrap resamples (90% strata)** — same statistics as originally intended, scoped to tractable cardinality. Define `stability(θ) = std(PR-AUC across resamples)` and impose `σ ≤ 0.03`; the chosen θ maximizes bootstrap mean PR-AUC among surviving candidates (otherwise fallback to unconditional best mean). **Sobol indices** on IForest/OCSVM and **LHS-drawn nu–gamma** probes cover complementary global sensitivity axes without layering bootstrap dispersion on every grid cell — those surfaces are plotted from point estimates unless I explicitly annotate bootstrap bands for the IForest stabilization run.

**Visualization protocol:**

- **1D sweep:** Line plot of PR-AUC ± 1 bootstrap std vs. parameter value (log x-axis for scale parameters: `eps`, `gamma`, `n_neighbors`)
- **2D sweep:** PR-AUC heatmap with contour lines every 0.05. Bold contour at max(AUC) − 0.05 defines the "acceptable operating region" — its width is a direct measure of hyperparameter robustness
- **2D decision boundaries:** Continuous anomaly score as a filled contour map per algorithm per Gagolewski dataset, animated across parameter values using `matplotlib.animation.FuncAnimation`
- **High-d hyperparameter space:** Parallel coordinate plots, lines colored by PR-AUC percentile (top 10% = dark, bottom 10% = light)

---

## 4. Robustness and Failure Cases

### 4.1 Dimensionality Stress Test

I'll run this protocol on `arrhythmia`, `musk`, `speech`, and `mnist`:

1. Apply PCA; retain components explaining [1%, 5%, 10%, 25%, 50%, 75%, 90%, 99%] of variance.
2. At each truncation level d', run all 6 algorithms with Track A defaults.
3. Plot PR-AUC vs. d' for all 6 methods on one figure per dataset.
4. On the same x-axis, plot the **distance concentration ratio** CR(d') = (d_max − d_min) / d_mean, computed on a random subsample of 500 pairwise distances.

When CR → 0, distance-based methods must fail. I expect LOF's PR-AUC to degrade precisely when CR drops below a dataset-specific threshold — that's the quantitative causal argument, not just an anecdote.

**Expected results:**
- IForest and ECOD: relatively flat across d'
- LOF: degrades monotonically with d'
- HBOS: flat (histogram-based, dimensionality-independent)
- OCSVM-RBF: degrades when γ="scale" because automatic scaling collapses with distance concentration

Importantly, I'll first run LOF and DBSCAN on raw high-dimensional data to demonstrate the failure, then run with PCA preprocessing to show whether performance recovers.

---

### 4.2 Local vs. Global Outlier Taxonomy

**Definitions:**

A **global outlier** is a point xₒ where d(xₒ, μ) > c × σ, with μ the global dataset centroid and σ the global spread. It is anomalous with respect to the entire data distribution.

A **local outlier** is a point xₒ where d(xₒ, μₗ) is normal (μₗ being the nearest cluster centroid), but its local reachability density is significantly lower than that of its neighbors. It is anomalous only relative to its immediate neighborhood.

**Algorithm-to-topology mapping (stated as testable hypotheses):**

- **IForest:** Detects global outliers (easy to isolate), systematically under-detects local outliers hidden in moderate-density regions.
- **LOF:** Designed for local outliers; will under-detect global outliers when global density is low but uniform.
- **DBSCAN:** Detects points not belonging to any density-connected cluster — structurally equivalent to global or severe local anomalies; misses subtle local deviations within clusters.
- **OCSVM:** Primarily a global method; performance depends on whether the outlier lies outside the RBF kernel's effective decision envelope.
- **ECOD/HBOS:** Detect marginal tail anomalies — a specific form of global outlier; completely blind to anomalies that are outlying only in joint or local space.

**Empirical classification protocol:**

For `wut/x2` (10 labeled noise points) and `sipu/flame` (12 labeled noise points), I'll classify each noise point as Type G (global) or Type L (local): if distance to nearest cluster centroid > 2 × within-cluster std → Type G; else → Type L.

I'll compute per-type recall for each algorithm and report as a 2×6 table (Type G/L × Algorithm).

For ODDS datasets (no geometric labels), I'll use a proxy: for each true positive, compute its normalized isolation score (distance to nearest inlier, normalized by dataset diameter). Binning true outliers by isolation decile and plotting recall per decile per algorithm will reveal which methods only detect the most globally isolated anomalies.

---

## 5. Verification Task: Blind Ensemble on `test_data.csv`

### 5.1 Preprocessing

1. Load `test_data.csv` — inspect dtypes, missing values, feature ranges, and near-zero-variance columns (drop any column where std < 0.01 after scaling).
2. Apply `RobustScaler`. I'm using this rather than `StandardScaler` because the data may contain the very outliers I'm trying to find — StandardScaler's mean and variance would be skewed by those outliers, compressing true outlier scores toward the center.
3. Check pairwise feature correlations. Drop one feature from any pair where |ρ| > 0.95 — collinear features double-count anomaly signal in distance-based methods.
4. Compute the distance concentration ratio CR at full dimensionality. If CR < 0.1, note that distance-based methods are unreliable and adjust ensemble weights accordingly.

### 5.2 Rank Aggregation Ensemble

Min-max normalization of raw anomaly scores is flawed: a single extreme score compresses 99.9% of scores into a near-zero band, destroying ranking information. Instead, I'll convert directly to fractional ranks:

```python
from scipy.stats import rankdata

def normalize_to_rank(scores):
    """Convert anomaly scores to fractional ranks in [0, 1].
    Higher rank = more anomalous. Invariant to scale and extreme values."""
    return rankdata(scores) / len(scores)

# For each algorithm i:
r_i = normalize_to_rank(scores_i)   # shape: (n,)
```

**Weighted Borda aggregation:**

```python
if d > 50 or CR < 0.1:
    # High-d or concentrated distance space: down-weight distance-based methods
    weights = {"OCSVM": 0.5, "LOF": 0.5, "DBSCAN": 0.5,
               "IForest": 1.0, "ECOD": 1.5, "HBOS": 1.5}
else:
    weights = {k: 1.0 for k in algorithms}

ensemble_score = sum(weights[alg] * r_i[alg] for alg in algorithms)
ensemble_score /= sum(weights.values())
```

### 5.3 Threshold Selection and Export

**Contamination estimate:** Find the 3 ODDS datasets with the most similar (d, n, feature type) profile to `test_data.csv` and use their mean contamination rate as ĉ.

**Fallback — elbow method:** If no ODDS dataset is structurally similar, plot the sorted ensemble score curve and use the maximum gradient point as the natural threshold.

**Consensus confidence filter:**

```python
high_confidence_threshold = 2 * c_hat

agreement = {
    x: sum(1 for alg in algorithms
           if r_i[alg][x] > (1 - high_confidence_threshold))
    for x in range(n)
}

# Label: 1 if ≥ 4 algorithms agree (majority); use ensemble threshold for borderline cases
final_labels = np.where(agreement >= 4, 1,
               np.where(agreement <= 1, 0,
               (ensemble_score > np.percentile(ensemble_score, 100*(1-c_hat))).astype(int)))
```

**CSV export — exact required format:**

```python
import pandas as pd

output = pd.DataFrame({"class": final_labels.astype(int)})
output.to_csv("test_labels.csv", index=False)

# Required assertions before submission:
assert output.columns.tolist() == ["class"]
assert set(output["class"].unique()).issubset({0, 1})
assert len(output) == len(test_data)
print(f"Outliers labeled: {output['class'].sum()} ({output['class'].mean():.3%})")
```

Column must be named exactly `class`. Values must be integers (not floats or booleans). No index column. All three assertions must pass before submission.

---

## 6. Cross-Cutting Scientific Questions

These are the broader questions I'm using this experimental setup to answer:

1. **Does contamination rate destabilize prior-dependent methods more than prior-free ones?**
   I'll compare PR-AUC variance across the contamination sweep for OCSVM and IForest (prior-dependent) vs. ECOD and HBOS (prior-free — contamination only shifts the threshold post-scoring). I expect OCSVM and IForest to show much higher sensitivity on `breastw`, `ionosphere`, `pima`, and `satellite`.

2. **Is IForest's dimensionality robustness empirically real?**
   From the PCA ablation in §4.1: if IForest's PR-AUC is flat while LOF's degrades monotonically, Liu et al.'s theoretical claim is empirically confirmed on my corpus.

3. **Does ECOD's independence assumption cause measurable degradation on correlated data?**
   From the regression of |PR-AUC(ECOD) − PR-AUC(LOF)| against mean pairwise ρ̄ across the ODDS mirrored corpus: a positive slope confirms the failure mode.

4. **Are the six algorithms measuring complementary or redundant information?**
   I'll compute Kendall rank correlation τ among anomaly score vectors per dataset. If any pair consistently achieves τ > 0.8, they're providing redundant information and the more expensive one's ensemble weight should be reduced.

5. **Does the elbow method produce contamination estimates consistent with ODDS ground truth?**
   For 5 ODDS datasets with known contamination, I'll apply the elbow method and compare the estimated ĉ to the true contamination. This validates the threshold selection strategy I'm using on `test_data.csv`.
