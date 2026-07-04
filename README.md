# 🏥 Medical Insurance Cost Predictor

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange) ![Status](https://img.shields.io/badge/Status-Complete-green)

## Problem Definition
Task: Predict `charges` (medical insurance cost) from `age`, `sex`, `bmi`, `children`, `smoker`, `region`.

Unlike the House Price Predictor (single model, capped by missing features), this project compares **5 regression models** with hyperparameter tuning to find which algorithm actually fits the data best.

---

## 📊 Results

| Model | Train R² | Test R² | CV R² (mean) | CV R² (std) | Test MAE | Test RMSE |
|---|---|---|---|---|---|---|
| LinearRegression | 0.7399 | 0.7801 | 0.7449 | 0.0238 | $4222.91 | $5843.15 |
| Ridge | 0.7399 | 0.7798 | 0.7449 | 0.0235 | $4235.49 | $5847.09 |
| Lasso | 0.7398 | 0.7796 | 0.7450 | 0.0235 | $4231.60 | $5848.97 |
| DecisionTree | 0.8537 | 0.8531 | 0.8445 | 0.0368 | $2865.64 | $4776.26 |
| **RandomForest** | **0.8810** | **0.8653** | **0.8520** | 0.0335 | **$2673.61** | **$4572.65** |

**Best model: RandomForest** — lowest test error, and Train/Test/CV scores are close enough to say it's not badly overfit (a ~1.6-point gap between train and test R² is normal for a tuned tree ensemble, not a red flag).

> **Note:** The linear models (LinearRegression, Ridge, Lasso) plateau around R² ≈ 0.78 no matter how much you tune them. That's not a tuning problem — it's a model-capacity problem. `charges` has a highly non-linear relationship with `smoker` × `bmi` (smokers with high BMI have a cost cliff), and a linear model can't represent an interaction like that without you engineering it explicitly. Tree-based models pick it up automatically via splits, which is why DecisionTree and RandomForest jump ~8-10 points in R² without any feature engineering.

---

## 📓 View Notebook
https://nbviewer.org/github/Siyalkamble/Insuarance-Cost-Predictor/blob/main/Medical_Insurance_Cost.ipynb

---

## 🛠️ Tech Stack

- Python 3.12
- Pandas — data loading and preprocessing
- NumPy — numerical operations
- Scikit-learn — `ColumnTransformer`, `Pipeline`, `GridSearchCV`, 5 regression algorithms
- Matplotlib / Seaborn — EDA and visualization

---

## ⚙️ ML Pipeline

```
Load Data → EDA (skew, scatter, boxplot, correlation heatmap) → Train/Test Split (80/20)
→ ColumnTransformer (StandardScaler + OneHotEncoder) → Pipeline
→ GridSearchCV over 5 models → Evaluate (Train/Test/CV R², MAE, RMSE) → Compare
```

---

## 🧠 Concepts Learned

### 1. ColumnTransformer + Pipeline (vs. manual preprocessing)
Instead of manually scaling numeric columns and one-hot encoding categoricals separately (like in the House Price Predictor), this project uses `ColumnTransformer` to apply different preprocessing to different column subsets in one object, then wraps it in a `Pipeline` with the model.

```python
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['age', 'bmi']),
    ('cat', OneHotEncoder(drop='first'), ['smoker', 'sex', 'region'])
])
pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
```

**Key insight:** This isn't just cleaner code — it's a correctness guarantee. `Pipeline.fit()` only ever fits the preprocessor on training data, and `.predict()` reuses those fitted statistics. It's structurally impossible to leak test-set statistics the way the House Price Predictor's Bug 3 did, because there's no separate manual `.fit_transform()` call on test data to get wrong.

### 2. GridSearchCV over multiple model families
```python
models_and_params = {
    'Ridge': (Ridge(), {'regressor__alpha': [0.1, 1, 10, 100]}),
    'RandomForest': (RandomForestRegressor(random_state=42),
                      {'regressor__n_estimators': [100, 200],
                       'regressor__max_depth': [5, 10, None]}),
    ...
}
```
Note the `regressor__` prefix — that's how you address a hyperparameter that belongs to a step *inside* a `Pipeline` when passing a `param_grid` to `GridSearchCV`. Each candidate pipeline is cloned and cross-validated independently; `grid.best_estimator_` gives you the pipeline with the winning hyperparameters already refit on the full training set.

### 3. Train R² vs Test R² vs CV R² — three different questions
- **Train R²**: can the model fit data it has already seen?
- **Test R²**: does it generalize to one specific held-out split?
- **CV R² (mean/std)**: how *consistent* is that generalization across 5 different splits?

A model can look good on a single test split by luck. `cross_val_score` with `cv=5` protects against that — the low CV std (0.02–0.04) here says the RandomForest's performance isn't a fluke of `random_state=42`.

### 4. Linear vs. non-linear model ceiling
Linear models assume additive, straight-line relationships between each feature and the target. When the real relationship is an interaction (smoker × bmi), a linear model literally cannot represent it unless you manually create an interaction feature. This is a modeling-capacity limitation, not a bug — different from the House Price Predictor, where the ceiling was a **missing feature**, not a **wrong model type**.

---

## ❌ Bugs I Made and Fixed

### Bug 1 — Stale kernel state from out-of-order execution
At one point, the fitted models in `best_estimators` were trained on a version of `X_train` with 7 columns (an earlier, unfixed version of `X = df` that hadn't dropped `charges` yet). After fixing the feature-selection cell and re-running it, `X_train` in memory had 6 columns — but the pipelines in `best_estimators` were never re-fit, so their internal `ColumnTransformer` still expected 7.

```
ValueError: X has 6 features, but ColumnTransformer is expecting 7 features as input.
```

**Fixed by:** Kernel → Restart & Run All, top to bottom, instead of trusting state built by clicking cells out of order.

**Lesson learned:** A notebook that "looks right" cell-by-cell can still hold two contradictory realities in memory at once. Before trusting any result, do a clean restart-and-run-all — this is a discipline issue, not a one-time typo.

### Bug 2 — Appending to the wrong variable
```python
# Wrong — 'results' is a dict from an earlier cell, dicts have no .append()
eval_results = []
for name, model in best_estimators.items():
    ...
    results.append({...})

# Fixed
eval_results.append({...})
```
**Impact:** Would have crashed with `AttributeError: 'dict' object has no attribute 'append'` the moment the stale-kernel bug above was fixed — two bugs stacked on the same cell.

---

## ⚠️ Dataset Limitations

This is the classic Kaggle `insurance.csv` (1338 rows, 6 features). It's clean — no missing values, no messy categoricals — which is exactly why it's a weak portfolio signal on its own. There's no real-world data engineering problem to solve here (no nulls, no outlier contamination in the raw sense, no leakage traps beyond the ones I built myself).

**What's actually missing:** no medical history, no pre-existing conditions, no insurance plan tier — the features that would matter most in a real underwriting model. R² ≈ 0.86 is a ceiling set by the dataset, not by model choice past this point.

---

## 🔄 What I Would Do Differently

- Engineer an explicit `smoker × bmi` interaction feature and re-test the linear models — if their R² jumps close to the tree models', that confirms the ceiling was purely about missing the interaction, not linear models being "worse" in general
- Log-transform `charges` (it's right-skewed) and compare R²/RMSE before vs. after
- Add a residual plot per model to see *where* each one fails (e.g., does RandomForest still underpredict high-cost smokers?)
- Try `XGBoost`/`LightGBM` since they usually beat vanilla RandomForest on tabular data like this, and are more standard in production ML systems than sklearn's RandomForest
- Report feature importances from the RandomForest/DecisionTree to sanity-check that `smoker` dominates (it should) — if it doesn't, that's a preprocessing bug, not a modeling finding
- Wrap the whole thing in a `Pipeline`-level `assert` on expected feature count before every `.fit()` call, to catch the stale-kernel-state bug automatically instead of by surprise

---

## 📁 Project Structure

```
medical-insurance-cost-predictor/
├── insurance.csv
├── Medical_Insurance_Cost.ipynb
├── model_comparison.png
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/Siyalkamble/medical-insurance-cost-predictor

# Install dependencies
uv pip install -r requirements.txt

# Open notebook
jupyter lab Medical_Insurance_Cost.ipynb
```
