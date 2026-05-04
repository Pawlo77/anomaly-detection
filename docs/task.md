**Data Exploration and Visualization 2025/2026**

## Project Assignment No. 2
Evaluate the effectiveness of various methods designed to detect outliers (anomalies).

### Learning the Methods
Consider the Outlier Detection DataSets (ODDS) repository: [https://shebuti.com/outlier-detection-datasets-odds/](https://shebuti.com/outlier-detection-datasets-odds/)

Each dataset includes a class variable `y`, where `1` denotes outliers and `0` denotes inliers. Please review the dataset descriptions available on the ODDS webpage. The provided anomaly labels will be treated as ground truth. This will allow you to evaluate the performance of outlier detection algorithms using metrics such as AUC, Accuracy, Precision, Recall, etc.

Additionally, use selected 2-dimensional datasets from the Benchmark Suite for Clustering Algorithms: [https://clustering-benchmarks.gagolewski.com/weave/suite-v1.html](https://clustering-benchmarks.gagolewski.com/weave/suite-v1.html)

Choose datasets that, in your opinion, are well suited for evaluating outlier/anomaly detection. This will allow you to include a visual assessment of the performance of the investigated algorithms (in addition to quantitative metrics).

Assume that anomaly labels are not available during the training stage.

## Algorithms
Use the following methods:
*   One-Class SVM
*   Isolation Forest
*   Local Outlier Factor
*   DBSCAN
*   One additional method of your choice (see, e.g., [https://deadwood.gagolewski.com/](https://deadwood.gagolewski.com/))

---

## Evaluation and Analysis

### 1. Performance Evaluation
Evaluate and compare the methods using selected performance metrics (e.g., AUC, Accuracy, Precision, Recall) based on comparison with the ground truth. Compare results across all methods.

### 2. Hyperparameter Analysis
For each method, perform a systematic analysis of key hyperparameters. At minimum:
*   Identify the most important hyperparameters for each algorithm.
*   Vary their values over a meaningful range.
*   Analyze how sensitive the method is to these parameters.
*   Present results using plots or tables.

Examples include:
*   **One-Class SVM:** `nu`, `kernel type`, `gamma`
*   **Isolation Forest:** `contamination`, `number of estimators`
*   **DBSCAN:** `eps`, `min_samples`
*   **Local Outlier Factor:** `number of neighbors`

**Discuss:**
*   Stability of results
*   Sensitivity to parameter changes

> *Copyright © 2026 Anna Cena. Last update: April 30, 2026 r.*
> *Data Exploration and Visualization 2025/2026 (2)*

*   Your thoughts on practical guidelines for parameter selection.

### 3. Robustness and Failure Case Analysis
Analyze when and why the methods succeed or fail. Consider dataset characteristics:
*   number of features (dimensionality)
*   number of observations
*   inability to detect local vs global outliers

### 4. Discussion
Discuss the strengths and weaknesses of each method as well as relationship between methods. Summarize your conclusions clearly in the final report (Jupyter Notebook).

---

## Verification Task
The dataset `test_data.csv` contains only features, with no information about outliers.

Your task is to identify potential outliers/anomalies using:
*   a single method, or
*   a combination (ensemble) of methods

In your report describe your approach, justify your choice of method as well as hyperparameters you used and any preprocessing steps that you performed.

In your final submission, include:
`test_labels.csv` containing a single binary column `class`, where:
*   `1` denotes an outlier
*   `0` denotes an inlier

---

## Submission Requirements

**Submission content**
As your project submission, you must upload the following to the MS Teams Assignment before the deadline:
1.  The Jupyter Notebook (including all analysis, experiments, and discussion).
2.  The `test_labels.csv` file.

> *Copyright © 2026 Anna Cena. Last update: April 30, 2026 r.*
