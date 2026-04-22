# Assignment 5 Summary

## Problem Statement

- Target variable: `Bankrupt?`
- Positive class: `1` means the company went bankrupt
- Business goal: identify risky companies early enough for a risk team to investigate
- Primary discrimination metric: `PR-AUC`
- Primary calibration metric: `Brier score`
- Business rule: false negatives matter more than false positives, so recall matters at the chosen operating point

## Dataset Summary

- Rows: `6,819`
- Columns: `96` total, including the target
- Target balance: `220` bankrupt and `6,599` non-bankrupt companies, about `3.2%` positive class
- Missing values: `0`
- Duplicate rows: `0`
- Split sizes: train `4,773`, validation `1,023`, test `1,023`
- Positive cases by split: train `154`, validation `33`, test `33`

## Metric Choice

- The positive class is rare, so a metric that focuses on ranking bankrupt firms above healthy firms is more useful than plain accuracy.
- Accuracy is misleading here because a model can predict almost every company as non-bankrupt and still look strong numerically.
- `PR-AUC` is the main discrimination metric because it pays attention to performance on the minority class instead of being dominated by the many negative cases.
- `Brier score` is the main calibration metric because it checks whether predicted probabilities are close to the observed outcomes, with lower values being better.
- Thresholded metrics like precision, recall, and `F2-score` are still useful for describing one operating point, but they do not fully describe ranking quality or probability calibration.
- A default threshold of `0.50` was used during the baseline experiment stage because it is the standard starting point for binary classification before any threshold tuning is done on the validation set.

## Feature Sets

- Feature Set A: all numeric predictors after basic cleaning
- Feature Set B: reduced set built from XGBoost feature importance on the training split only
- Selection method for B: kept the top `25` importance-ranked features from a training-only selector model
- Number of features kept: `25` out of `95` usable predictors
- Tradeoff: Feature Set B was easier to explain, but it lost noticeable validation `PR-AUC` compared with the stronger all-feature XGBoost runs

## Experiment Results

| exp_id | model | feature_set | main_settings | train_pr_auc | val_pr_auc | overfit_gap | val_roc_auc | val_brier | threshold | val_precision | val_recall | val_f1_or_f2 | selected_finalist | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Logistic Regression | A | median impute + scale + default logistic regression | 0.5308 | 0.2587 | 0.2721 | 0.8766 | 0.0316 | 0.50 | 0.3333 | 0.1818 | 0.2000 | No | Simple baseline using the default `0.50` threshold before tuning |
| 2 | XGBoost Baseline | A | n_estimators=300, learning_rate=0.05, max_depth=4 | 1.0000 | 0.5525 | 0.4475 | 0.9645 | 0.0207 | 0.50 | 0.7692 | 0.3030 | 0.3448 | No | First serious model |
| 3 | XGBoost Imbalance-Aware | A | scale_pos_weight=29.99 | 1.0000 | 0.5377 | 0.4623 | 0.9620 | 0.0242 | 0.50 | 0.5263 | 0.6061 | 0.5882 | No | Better recall, weaker calibration |
| 4 | XGBoost Tuned | A | n_estimators=500, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0 | 0.9973 | 0.5831 | 0.4142 | 0.9665 | 0.0199 | 0.50 | 0.6667 | 0.3030 | 0.3401 | Yes | Best validation PR-AUC and Brier |
| 5 | XGBoost Selected Features | B | top 25 XGBoost importance features | 0.9988 | 0.4430 | 0.5558 | 0.9522 | 0.0233 | 0.50 | 0.6250 | 0.3030 | 0.3378 | No | Simpler, but performance dropped |

## Winning Model

- Selected model: `XGBoost Tuned` on Feature Set A
- Why it won: highest validation `PR-AUC` at `0.5831`
- Tie-break support: best validation `Brier score` at `0.0199` among the top models
- Extra support: strong validation `ROC-AUC` at `0.9665`
- Simpler alternative check: the selected-features model was easier to explain but not close enough in validation `PR-AUC`
- Validation threshold: `0.15`, chosen on the validation set to maximize `F2-score`
- Final settings: `n_estimators=500`, `learning_rate=0.03`, `max_depth=3`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=2.0`

## Final Test Results

- Test `PR-AUC`: `0.5161`
- Test `ROC-AUC`: `0.9549`
- Test `Brier score`: `0.0225`
- Test precision: `0.3953`
- Test recall: `0.5152`
- Test `F2-score`: `0.4857`
- Test `F1-score`: `0.4474`

## Confusion Matrix

- True negatives: `964`
- False positives: `26`
- False negatives: `16`
- True positives: `17`

## Calibration Result

- Test-set `Brier score`: `0.0225`
- Reliability / calibration curve summary: predicted risk generally rose with observed bankruptcy rate, so the model retained useful ranking information
- Higher predicted risk did usually line up with higher observed bankruptcy rates, especially in the higher-probability bins
- Calibration was not perfect because a few upper bins were unstable and noisy, which is expected with only `33` positive cases in the test set
- Practical takeaway: the probabilities look usable for prioritization, but I would still treat the exact risk numbers cautiously

## Interpretability

- Top feature: `Persistent EPS in the Last Four Seasons`
- Why it might make sense: sustained earnings behavior can reflect a company's financial health and stability
- Other high-importance features included `Net Value Growth Rate`, `Borrowing dependency`, `Net worth/Assets`, and `Continuous interest rate (after tax)`
- Limitation of this interpretation: feature importance shows influence on the model, not the direction of effect or a causal relationship

## Overfitting Discussion

- Final model train `PR-AUC`: `0.9973`
- Final model validation `PR-AUC`: `0.5831`
- Overfit gap: `0.4142`
- Interpretation: this gap is well above the assignment's caution range, so the final XGBoost model is clearly overfit even though it still had the best validation performance in the experiment set
- Why I still kept it: model selection followed the rubric, and no other submitted experiment matched its validation `PR-AUC` plus `Brier score`

## AI Usage

- AI tool used: Codex in VS Code
- What it helped with: notebook organization, reusable evaluation helpers, plotting code, and cleaning up repeated experiment code
- What I had to verify or fix myself: that the test set stayed untouched until the final section and that the final model was picked from validation metrics rather than test results
- One mistake or weak suggestion I caught: an earlier repo note said to choose the winner mainly with `F2-score`, which did not match the assignment rubric, so I corrected the selection logic to use validation `PR-AUC`, `Brier score`, and overfitting evidence
