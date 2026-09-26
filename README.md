# DSN Bootcamp Qualification Hackathon 2026 — ML Track

Predicting `total_sales` for a product at a given DSN Mart store (regression, scored by RMSE).

## Problem

DSN Mart operates stores across Nigeria, from corner shops to flagship hypermarkets. Given
product attributes (weight, price, category, shelf visibility) and store attributes (size,
format, location tier, age), the task is to predict total sales for that product at that store.

## Data quality issues found

| Issue | Fix applied |
|---|---|
| `product_category` typed in mixed case (48 raw values, 16 real categories) | lowercase + strip |
| `product_weight_kg` missing (~18%) | imputed from the mean weight of the same `product_code` (weight is a product attribute, not a store one), falling back to category mean |
| `store_size` missing (~28%), entirely missing for 3 specific stores | kept as its own `"Unknown"` category rather than guessed |
| `shelf_visibility == 0` | not physically meaningful — treated as missing, imputed from category mean |

## Feature engineering

- `price_per_kg`, `price_rank_in_cat` — price positioning within a category
- `visibility_ratio_in_cat` — relative shelf space vs. category average
- `category_broad` — Food / Drinks / Non-Consumable grouping
- `product_freq`, `category_freq` — frequency encodings
- `store_age_bucket`, `store_format_tier` — coarser bucket / interaction features
- Leak-safe K-fold target encoding of historical average `total_sales` for `product_code`,
  `product_category`, and `store_code` (computed out-of-fold on train, so no row ever sees its
  own target)

## Model

Three gradient-boosting models — **LightGBM**, **XGBoost**, **CatBoost** — each trained with
5-fold cross-validation repeated over 3 random seeds (bagging), then blended using weights
chosen to minimize out-of-fold RMSE.

## Results (out-of-fold RMSE)

| Model | RMSE |
|---|---|
| Linear Regression (baseline) | 1123.82 |
| LightGBM | 1078.73 |
| XGBoost | 1079.89 |
| CatBoost | 1071.62 |
| **Blended ensemble** | **1071.07** |

## Files

- `dsn_mart_sales_solution.ipynb` — full solution: EDA, cleaning, feature engineering, modeling, submission
- `submission.csv` — final predictions for the competition test set

## How to run

1. Place `train.csv`, `test.csv`, `sample_submission.csv` in the same folder as the notebook (or update `DIR` in the first code cell).
2. `pip install lightgbm xgboost catboost scikit-learn pandas numpy matplotlib seaborn`
3. Run all cells top to bottom. This produces `submission.csv`.
