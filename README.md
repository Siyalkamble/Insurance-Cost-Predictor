# 🏠 House Price Predictor

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange) ![Status](https://img.shields.io/badge/Status-Complete-green)

A Linear Regression model trained on a simplified Ames Housing dataset to predict house sale prices. Built as part of a structured ML learning roadmap — focused on understanding the full ML workflow, not just getting a high score.

---

## 📊 Results

| Metric | Score |
|---|---|
| Training R² | 0.61 |
| Testing R² | 0.62 |
| Model Status | Underfitting (both scores low and similar) |

> **Note:** The low R² is expected. The dataset used here is a simplified 13-column version of the full Ames Housing dataset (79 columns). The most important feature — `OverallQual` — is missing. See Dataset Limitations below.

---

## 📓 View Notebook
[Click here to view notebook](https://nbviewer.org/github/Siyalkamble/house-price-predictor/blob/main/House%20Price%20Predictor.ipynb)

---

## 🛠️ Tech Stack

- Python 3.12
- Pandas — data loading and preprocessing
- NumPy — numerical operations
- Scikit-learn — model training, scaling, evaluation
- Matplotlib — visualization

---

## ⚙️ ML Pipeline

```
Load Data → Drop ID Column → Handle Missing Values → Encode Categorical Features
→ Train/Test Split (80/20) → Feature Scaling → Train Linear Regression → Evaluate
```

---

## 🧠 Concepts Learned

### 1. One-Hot Encoding
Converts nominal categorical features into binary vectors.
Used `pd.get_dummies()` for columns: `MSZoning`, `LotConfig`, `BldgType`, `Exterior1st`.

```
MSZoning: RL → [1, 0, 0, 0]
MSZoning: RM → [0, 1, 0, 0]
```

**Key insight:** Nominal categories (colour, zone type) have no order. Ordinal categories (ratings, rankings) do. Never use label encoding on nominal features — it implies a false ranking.

### 2. Dummy Variable Trap
When using One-Hot Encoding, drop one column per feature (`drop_first=True`). Without this, features are perfectly correlated and the model breaks.

### 3. Train/Test Split
- 80% training — model learns from this
- 20% testing — model is evaluated on unseen data
- `random_state=42` ensures reproducibility

### 4. Feature Scaling (StandardScaler)
Converts all features to the same scale: mean = 0, std = 1.

```python
# Learn statistics ONLY from training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply same statistics to test data — never refit
X_test_scaled = scaler.transform(X_test)
```

**Why this matters:** `LotArea` ranges 1,300–215,245. `OverallCond` ranges 1–9. Without scaling, the model treats `LotArea` as ~23,000x more important before learning anything.

### 5. Overfitting vs Underfitting

| Situation | Train R² | Test R² | Meaning |
|---|---|---|---|
| Overfitting | High (0.95) | Low (0.50) | Memorised training data |
| Underfitting | Low (0.61) | Low (0.62) | Didn't learn enough |
| Good fit | High (0.85) | Similar (0.82) | Generalises well |

### 6. Correlation Analysis
```python
df.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)
```
**Finding:** `TotalBsmtSF` (0.61) had the strongest correlation with `SalePrice`. `OverallCond` was slightly negative — because condition alone doesn't predict price; material quality matters more.

---

## ❌ Bugs I Made and Fixed

### Bug 1 — `drop()` without assignment
```python
# Wrong — returns new DataFrame but never saves it
df.drop(['Id'], axis=1)

# Fixed
df = df.drop(['Id'], axis=1)
```

### Bug 2 — train_size vs test_size backwards
```python
# Wrong — 20% training, 80% testing
train_test_split(X, y, train_size=0.2)

# Fixed — 80% training, 20% testing
train_test_split(X, y, test_size=0.2)
```
**Impact:** Model was learning from only 292 rows and testing on 1168. This directly caused overfitting.

### Bug 3 — Scaler applied to test data incorrectly
```python
# Wrong — leaks test statistics into the model
X_test_scaled = scaler.fit_transform(X_test)

# Fixed — use training statistics on test data
X_test_scaled = scaler.transform(X_test)
```
**Why this matters:** This is called data leakage. You cannot use test data statistics before the model has seen it — that is cheating.

---

## ⚠️ Dataset Limitations

This project uses a simplified 13-column version of the Ames Housing dataset.

**Missing critical feature:** `OverallQual` — overall material and finish quality — is the single strongest predictor of house price in the full dataset. Without it, R² is capped around 0.62 regardless of model complexity.

**Lesson learned:** No amount of feature engineering or model tuning fixes a dataset with missing important columns. Always ask — do I have the right data? — before writing any model code. **Garbage in, garbage out.**

---

## 🔄 What I Would Do Differently

- Use the full Ames Housing dataset (79 features) from Kaggle
- Add EDA before any modelling — histograms, scatter plots, correlation heatmap
- Log-transform `SalePrice` to fix right skew
- Create engineered features: `HouseAge`, `YearsSinceRemodel`, `TotalSF`
- Add cross-validation instead of a single train/test split
- Add residual plot to diagnose where the model fails
- Try Ridge/Lasso regression to handle multicollinearity

---

## 📁 Project Structure

```
house-price-predictor/
├── data.csv
├── House_Price_Predictor.ipynb
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/Siyalkamble/house-price-predictor

# Install dependencies
pip install pandas numpy scikit-learn matplotlib

# Open notebook
jupyter lab House_Price_Predictor.ipynb
```