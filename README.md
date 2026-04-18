# Data 1204 Assignment 5

Binary classification project for predicting whether a company is likely to go bankrupt.

## Repository Structure

- `lastname_firstname_assignment5.ipynb` - main notebook submission
- `src/model_utils.py` - reusable helpers for evaluation and feature selection
- `reports/assignment5_summary.md` - short summary template you can fill in after results are ready
- `data/` - place `data.csv` here before running the notebook

## Setup

1. Create and activate a Python environment.
2. Install the required packages:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

3. Download the workshop dataset and save it as `data/data.csv`.
4. Open `lastname_firstname_assignment5.ipynb` and run it from top to bottom.

## Workflow

- Use a stratified `70/15/15` train/validation/test split with random seed `42`.
- Keep the test set untouched until the final evaluation section.
- Compare exactly 5 experiments using the same evaluation function.
- Select the winning model using validation `F2-score`, then check overfitting and simplicity.

## Deliverables

- GitHub repository link
- Notebook
- Short summary
- 10-minute walkthrough video

