# Linear Regression from Scratch

Linear, Ridge, Lasso and ElasticNet regression implemented **by hand with NumPy** and benchmarked against `scikit-learn` on a real-world task: predicting apartment rental prices.

## Goal

Understand linear models by building them instead of calling `.fit()`. Every model is implemented from the math up, then verified against `scikit-learn` to make sure the results match.

## Dataset

[Two Sigma Connect: Rental Listing Inquiries](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/data) (Kaggle, listings from renthop.com).

- **Target:** `price` (monthly rent, $)
- **Cleaning:** prices outside the 1st–99th percentile are dropped
- **Split:** 80 / 20 train/test, `random_state=21`
- **Features (22):** `bathrooms`, `bedrooms` + 20 binary amenity flags built from the messy `features` column (the 20 most frequent amenities)

## What's implemented

| Area | Details |
|------|---------|
| **Linear regression** | Gradient descent, stochastic gradient descent (deterministic), closed-form solution (normal equation) |
| **Regularization** | Ridge (L2), Lasso (L1), ElasticNet, all written from scratch |
| **Preprocessing** | Custom `MinMaxScaler` and `StandardScaler`, checked against `sklearn` |
| **Metrics** | MAE, RMSE and a custom R² |
| **Overfitting study** | Degree-10 polynomial features, effect of regularization, tuning of `alpha` |
| **Baselines** | Naive mean and median predictors |
| **Extras** | Log-transformed target, removing outliers from train only, mini-batch vs full-batch GD |

The notebook also contains written derivations and explanations:

- the analytical solution of linear regression in vector form
- what changes when L1 and L2 penalties are added
- why L1 produces exactly-zero weights (feature selection)
- how linear models capture non-linear dependencies (linear in parameters, not in features)

## Results

| | MAE (test) | RMSE (test) | R² (test) |
|---|---|---|---|
| Best model (`sklearn` Lasso) | **≈ 708 $** | ≈ 1 026 $ | 0.586 |

- The best model beats the naive median predictor by **34.5%** (MAE).
- The top ~10 models, including my own analytical solution, Ridge and Lasso, differ by **less than $1 of MAE**. On 22 features and ~38k objects, the choice between linear models barely matters.
- **Features mattered more than the model:** 2 base features gave 754 $ MAE, while adding 20 amenity flags gave 708 $. Inflating the same 2 features to 66 polynomial ones gave only 757 $.
- Regularization did not help where nothing was broken. It only paid off on degree-10 polynomials, where overfitting was real.

## Project structure

```
.
├── notebooks/
    └── ml_linear_regression.ipynb
├── data/
    └── train.json # not tracked by git
├── requirements.txt
└── README.md
```

## How to run

1. **Clone the repository and install dependencies**

   ```bash
   git clone https://github.com/tv-papikian/linear-regression-from-scratch.git
   cd linear-regression-from-scratch
   pip install -r requirements.txt
   ```

2. **Download the data.** The dataset is not included in the repo. Accept the competition rules on Kaggle, download `train.json` and place it at `data/train.json`.

3. **Run the notebook**

   ```bash
   jupyter notebook ml_linear_regression.ipynb
   ```

   The first cells preprocess `train.json` and save `data/train_part.json` and `data/test_part.json`; the rest of the notebook works with those files.

## Tech stack

Python · NumPy · pandas · matplotlib · seaborn · scikit-learn (as a reference only) · Jupyter
