# Assignment 5 Summary

## Problem Statement

- Target variable: `Bankrupt?`
- Positive class: `1` means the company went bankrupt
- Business goal: identify risky companies early enough for a risk team to investigate
- Primary discrimination metric: `PR-AUC`
- Primary calibration metric: `Brier score`

## Dataset Summary

- Rows:
- Columns:
- Class balance:
- Missing values:
- Duplicate rows:

## Metric Choice

- Why PR-AUC fits this imbalanced problem:
- Why Brier score matters for probability quality:
- Why accuracy is misleading:
- Why thresholded metrics are still secondary:

## Feature Sets

- Feature Set A:
- Feature Set B:
- Selection method for B:
- Number of features kept:
- Tradeoff:

## Experiment Results

| exp_id | model | feature_set | main_settings | train_pr_auc | val_pr_auc | overfit_gap | val_roc_auc | val_brier | threshold | val_precision | val_recall | val_f1_or_f2 | selected_finalist | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Winning Model

- Selected model:
- Why it won:
- Validation threshold:
- Final settings:

## Final Test Results

- Test `PR-AUC`:
- Test `ROC-AUC`:
- Test `Brier score`:
- Test precision:
- Test recall:
- Test F1-score or F2-score:

## Confusion Matrix

- True negatives:
- False positives:
- False negatives:
- True positives:

## Calibration Result

- Reliability / calibration curve summary:
- Does higher predicted risk line up with higher observed risk?:
- Does the model look overconfident or underconfident?:

## Interpretability

- Top feature:
- Why it might make sense:
- Limitation of this interpretation:

## Overfitting Discussion

- Final model train PR-AUC:
- Final model validation PR-AUC:
- Overfit gap:
- Interpretation:

## AI Usage

- AI tool used:
- What it helped with:
- What I had to verify or fix myself:
- One mistake or weak suggestion I caught:
