# Marquette Basketball Predictor (2023-24)

Logistic regression model to predict Marquette men's basketball game outcomes from box-score statistics.

## What it does

1. **Random Forest feature importance** — ranks which game stats (points, shooting, rebounds, turnovers, etc.) are most associated with wins and losses.
2. **Logistic Regression** — trains a classifier on Marquette and opponent stats to predict win (`0`) or loss (`1`).

The notebook explored using only the top Random Forest features, but kept the full feature set for the final model because the reduced set overfit on this small dataset.

## Data

- `marquette.csv` — 37 games from Marquette's 2023-24 season (Sports Reference format)
- Target: `W/L` encoded as win = 0, loss = 1

## Results

On an 80/20 train-test split (`random_state=42`), the logistic regression model reaches **87.5% accuracy** on the 8-game test set. With only 37 games total, treat this as exploratory — the holdout set is very small and class balance can shift results.

## Requirements

- Python 3
- pandas, numpy, scikit-learn, matplotlib

## Run

Open and run all cells in `logisticReg.ipynb`. The notebook reads `marquette.csv` from the project root.
