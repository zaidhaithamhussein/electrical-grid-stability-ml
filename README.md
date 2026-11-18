# Electrical Grid Stability Prediction — Machine Learning

University Artificial Intelligence project using Python and scikit-learn to study electrical grid stability.

## Overview

Uses the Electrical Grid Stability dataset (10,000 rows) for two supervised learning tasks:
- **Regression:** Predict the continuous `stab` stability value with Linear Regression.
- **Classification:** Predict `stabf` (stable/unstable) with K-Nearest Neighbors (KNN).

## Techniques

- Exploratory data analysis and missing-value checks
- Train/test split (80/20) and stratified classification split
- StandardScaler for distance-based KNN classification
- KNN neighbor-count comparison (k=1–20)
- Regression metrics: MSE, RMSE, R²
- Classification metrics: accuracy, precision, recall, F1 score, confusion matrix
- MLflow tracking and model logging

## Results recorded in the notebook

| Metric | Value |
| --- | ---: |
| Linear Regression R² | 0.6445 |
| Baseline KNN accuracy (k=5) | 86.25% |
| Selected KNN accuracy (k=10) | 87.95% |
| Selected KNN F1 (unstable class) | 0.9086 |

**Evaluation caveat:** The notebook selected the best k using the same test set on which it reports accuracy. This may make the selected score optimistic. A better future version would tune using validation data or cross-validation, then evaluate once on a held-out test set.

## Files

- `FinalNoteBookAB.ipynb`: original university notebook (unchanged)
- `Electrical Grid Stability.csv`: input dataset
- `.gitignore`: excludes notebook checkpoints and local MLflow artifacts

## How to run

1. Install Python and Jupyter Notebook/JupyterLab.
2. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib scikit-learn mlflow notebook
   ```

3. Open `FinalNoteBookAB.ipynb` in Jupyter.
4. Ensure `Electrical Grid Stability.csv` is in the same directory as the notebook.
5. Run the notebook from top to bottom. MLflow may create local `mlruns/` and `mlflow.db` files; those are ignored by Git.

## Notes

This is an academic ML experiment, not a deployed production prediction service. The original notebook and dataset were preserved as supplied.
