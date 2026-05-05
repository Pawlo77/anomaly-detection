# Model Intuition + Hyperparameter Effects

Scope: six detectors from `docs/plan.md` (`OCSVM`, `IForest`, `LOF`, `DBSCAN`, `ECOD`, `HBOS`). This file only explains model behavior and planned hyperparameters; no extra methods added. [P]

## 1) One-Class SVM (`OCSVM`)

### Intuition
- Learns decision boundary that separates data from origin in feature space; points outside boundary treated as anomalies. [R1][R2]
- With RBF kernel, similarity is `exp(-gamma * ||x-z||^2)`, so `gamma` controls locality of boundary. [R1]
- `nu` has strict role: upper bound on training errors, lower bound on support vectors. So too-small `nu` caps achievable recall when true contamination is higher. [R1][R2]

### Hyperparameters used in plan, and what each changes
- `kernel` (`rbf`, `poly`, `sigmoid`, `linear`): chooses feature map / decision surface family; directly changes geometry model can represent. [R1]
- `nu`: increases fraction allowed outside boundary and usually increases outlier predictions; also changes margin/support-vector regime by definition above. [R1][R2]
- `gamma` (`scale`, `auto`, numeric): larger `gamma` -> tighter, more local boundary; smaller `gamma` -> smoother/global boundary. (`scale` and `auto` are sklearn-defined formulas.) [R1]
- `degree` (poly only): raises polynomial order; increases boundary complexity for polynomial kernel. [R1]
- `coef0` (poly/sigmoid only): shifts kernel function (`coef0` term in formula), affecting influence of higher-order terms / activation offset. [R1]

## 2) Isolation Forest (`IForest`)

### Intuition
- Builds random trees; anomalies isolate in fewer splits (short path length). Score derived from expected path length normalization. [R3][R4]
- Works by partitioning, not density estimation; robust when many features are irrelevant because splits are random subsamples of features/samples. [R4][P]

### Hyperparameters used in plan, and what each changes
- `n_estimators`: number of trees. More trees reduce variance of score estimate but cost more runtime. [R3]
- `max_samples` (`auto`, int): samples per tree. Sklearn `auto` = `min(256, n_samples)`; directly sets subsample size each tree sees. [R3]
- `contamination`: used to set decision threshold after scoring; higher value marks larger top-score fraction as outliers. [R3]
- `max_features`: feature subsampling ratio per tree; lower values increase randomness/diversity, can help high-d/noisy settings. [R3]
- `bootstrap`: sample with replacement vs without replacement for tree training subsets. [R3]

## 3) Local Outlier Factor (`LOF`)

### Intuition
- Compares local reachability density of point to densities of its neighbors; score near 1 means similar local density, much larger than 1 means local anomaly. [R5][R6]
- Built for local (not purely global) anomalies because scoring is neighborhood-relative. [R6]

### Hyperparameters used in plan, and what each changes
- `n_neighbors`: neighborhood size `k`; small `k` = very local/sensitive, large `k` = smoother/more global behavior. [R5]
- `metric` (`euclidean`, `manhattan`, `minkowski`, `cosine`): changes neighbor graph and reachability distances, so can fully change ranking. [R5]
- `p` (Minkowski only): distance order for Minkowski metric (`p=1` Manhattan, `p=2` Euclidean). [R5]
- `contamination`: threshold parameter for binary labeling (`predict`), not core local-density formula itself. [R5][R6]
- `novelty=False` in plan: transductive outlier detection on training set (fit+predict same data), not novelty scoring on unseen data. [R5][R6][P]

## 4) DBSCAN

### Intuition
- Density clustering: core points have at least `min_samples` neighbors within `eps`; anomalies are points not density-reachable from any core cluster (noise label). [R7][R8]
- Assumes one global density scale (`eps`) per run; mixed-density data can break single-`eps` fit. [R7][P]

### Hyperparameters used in plan, and what each changes
- `eps`: neighborhood radius; most sensitive knob. Larger `eps` merges regions and reduces noise labels; smaller `eps` splits clusters and increases noise labels. [R7]
- `min_samples`: density requirement for core status; higher values require denser regions, typically producing fewer core points and more noise. [R7]
- `metric`: distance function used for neighborhoods; changes which points are considered neighbors. [R7]
- `algorithm="auto"`: lets sklearn choose neighbor-search backend based on data/metric. [R7]

## 5) ECOD

### Intuition
- Uses empirical CDF tails per feature, then aggregates tail evidence into anomaly score; no kNN graph, no kernel, no tree ensemble. [R9][R10]
- Independence assumption is structural: marginal-tail aggregation can miss anomalies visible only in joint feature interactions. [R9][P]

### Hyperparameters used in plan, and what each changes
- `contamination` (only tuned param): does not redefine ECOD score construction; used in PyOD detector interface to set threshold / labels from decision scores. [R9][R11]

## 6) HBOS

### Intuition
- Histogram per feature; low-frequency bins imply high anomaly contribution; total score aggregates per-feature histogram evidence. [R12][R13]
- Like ECOD, effectively marginal and independence-based; unlike ECOD, discretization via bins makes behavior sensitive to binning choices. [R12][R13]

### Hyperparameters used in plan, and what each changes
- `n_bins` (`int` or `auto`): histogram granularity. Too coarse can hide local valleys; too fine can create sparse/empty bins. [R12]
- `alpha`: regularizer to avoid numerical issues from zero/near-zero bin probability (stabilizes extreme scores). [R12]
- `tol`: controls flexibility for values near/outside learned bin support in PyOD implementation. [R12]
- `contamination`: PyOD thresholding parameter for binary labels from decision scores. [R11][R12]

## References

- [P] Project plan: [`docs/plan.md`](./plan.md)
- [R1] scikit-learn `OneClassSVM` API: <https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html>
- [R2] Schölkopf et al. (2001), *Estimating the Support of a High-Dimensional Distribution*: <https://link.springer.com/article/10.1023/A:1007612920974>
- [R3] scikit-learn `IsolationForest` API: <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html>
- [R4] Liu, Ting, Zhou (2008), *Isolation Forest*: <https://ieeexplore.ieee.org/document/4781136>
- [R5] scikit-learn `LocalOutlierFactor` API: <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html>
- [R6] Breunig et al. (2000), *LOF: Identifying Density-Based Local Outliers*: <https://sigmodrecord.org/?smd_process_download=1&download_id=6631>
- [R7] scikit-learn `DBSCAN` API: <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>
- [R8] Ester et al. (1996), *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise*: <https://cdn.aaai.org/KDD/1996/KDD96-037.pdf>
- [R9] PyOD `ECOD` source docs: <https://pyod.readthedocs.io/en/latest/_modules/pyod/models/ecod.html>
- [R10] Li et al. (2022), *ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions*: <https://arxiv.org/abs/2201.00382>
- [R11] PyOD `BaseDetector` docs (`contamination` threshold semantics): <https://pyod.readthedocs.io/en/latest/_modules/pyod/models/base.html>
- [R12] PyOD `HBOS` source docs: <https://pyod.readthedocs.io/en/latest/_modules/pyod/models/hbos.html>
- [R13] Goldstein, Dengel (2012), *Histogram-based Outlier Score (HBOS)*: <https://www.dfki.de/fileadmin/user_upload/import/6431_HBOS-KI-2012.pdf>
