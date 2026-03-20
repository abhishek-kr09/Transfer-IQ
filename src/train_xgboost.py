# ============================================================
# FILE: src/train_xgboost.py
#
# WHAT CHANGED (and why the old version was broken):
#
# OLD APPROACH (broken):
#   - XGBoost predicted the RESIDUAL (actual - lstm_pred)
#   - Used 80/20 row split → train=seasons 3-4, test=last 20% of season 5
#   - Residuals were wildly inconsistent at test time → RMSE blew up -927%
#
# NEW APPROACH (correct — matches mentor instruction):
#   "add lstm_pred as a column, then retrain with XGBoost"
#   - XGBoost predicts market_value_eur DIRECTLY
#   - lstm_pred is just one feature among many (but an important one)
#   - XGBoost naturally learns to use and correct lstm_pred
#   - SPLIT: train on seasons 3-4, test on season 5 (proper time-series)
#     This is cleaner than a random 80/20 row split because season 5
#     is genuinely unseen future data.
# ============================================================

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# ------------------------------------------------
# Load enriched dataset
# ------------------------------------------------
df = pd.read_csv("data/processed/lstm_enriched.csv")
df = df.sort_values(["season_encoded", "player_name"]).reset_index(drop=True)

print(f"Loaded: {df.shape}")
print(f"Seasons: {sorted(df['season_encoded'].unique())}")

# ------------------------------------------------
# Feature set
# lstm_pred is included as a feature — XGBoost will learn
# to weight it alongside the other columns.
# ------------------------------------------------
xgb_features = [
    # LSTM signal — the single most important feature
    "lstm_pred",

    # Player context
    "current_age",
    "age_decay_factor",
    "position_encoded",
    "season_encoded",

    # Performance
    "attacking_output_index",
    "injury_burden_index",
    "availability_rate",
    "goals_per90",
    "assists_per90",
    "goal_contributions_per90",
    "minutes_played",
    "pass_accuracy_pct",

    # Sentiment & social
    "vader_compound_score",
    "social_buzz_score",
]

# Keep only columns that exist
xgb_features = [f for f in xgb_features if f in df.columns]
print(f"\nUsing {len(xgb_features)} features: {xgb_features}")

TARGET = "market_value_eur"

X = df[xgb_features]
y = df[TARGET]

# ------------------------------------------------
# Time-series split: train on seasons 3-4, test on season 5
#
# WHY THIS SPLIT:
#   - Season 5 is the most recent data → true hold-out test
#   - Prevents any future leakage into training
#   - Gives a fair comparison: both LSTM and XGBoost are
#     evaluated on season 5 which neither saw during training
# ------------------------------------------------
train_mask = df["season_encoded"] <= 4
test_mask  = df["season_encoded"] == 5

X_train = X[train_mask]
X_test  = X[test_mask]
y_train = y[train_mask]
y_test  = y[test_mask]

lstm_test  = df.loc[test_mask, "lstm_pred"].values
actual_test = y_test.values

print(f"\nTrain rows: {len(X_train)}  (seasons 3-4)")
print(f"Test rows:  {len(X_test)}   (season 5)")

# ------------------------------------------------
# XGBoost — conservative settings to avoid overfitting
# min_child_weight=20 prevents learning from tiny player subgroups
# ------------------------------------------------
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=20,
    reg_alpha=0.1,       # L1 regularisation
    reg_lambda=1.0,      # L2 regularisation
    random_state=42,
    verbosity=0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# ------------------------------------------------
# Predictions (direct market value, not residual)
# ------------------------------------------------
xgb_preds = model.predict(X_test)
xgb_preds = np.maximum(xgb_preds, 0)   # no negative values

# ------------------------------------------------
# Evaluation — LSTM baseline vs XGBoost Ensemble
# Both evaluated on the identical held-out season 5 rows
# ------------------------------------------------
lstm_rmse = np.sqrt(mean_squared_error(actual_test, lstm_test))
xgb_rmse  = np.sqrt(mean_squared_error(actual_test, xgb_preds))

lstm_mae  = mean_absolute_error(actual_test, lstm_test)
xgb_mae   = mean_absolute_error(actual_test, xgb_preds)

rmse_pct = (lstm_rmse - xgb_rmse) / lstm_rmse * 100
mae_pct  = (lstm_mae  - xgb_mae)  / lstm_mae  * 100

print("\n" + "=" * 52)
print("  MODEL COMPARISON  (test = season 5, unseen)")
print("=" * 52)
print(f"  LSTM RMSE      : €{lstm_rmse:>12,.0f}")
print(f"  XGBoost RMSE   : €{xgb_rmse:>12,.0f}  {'✅ improved' if xgb_rmse < lstm_rmse else '❌ worse'}")
print(f"  LSTM MAE       : €{lstm_mae:>12,.0f}")
print(f"  XGBoost MAE    : €{xgb_mae:>12,.0f}  {'✅ improved' if xgb_mae < lstm_mae else '❌ worse'}")
print(f"\n  RMSE reduction : {rmse_pct:+.1f}%")
print(f"  MAE  reduction : {mae_pct:+.1f}%")
print("=" * 52)

# Feature importance
fi = pd.Series(model.feature_importances_, index=xgb_features).sort_values(ascending=False)
print("\nTop feature importances:")
for feat, imp in fi.head(8).items():
    bar = "█" * int(imp * 40)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

# ------------------------------------------------
# Save model
# ------------------------------------------------
joblib.dump(model, "dashboard/xgb_model.pkl")
print("\n✅ Model saved → dashboard/xgb_model.pkl")
print("   Run: streamlit run dashboard/app.py")