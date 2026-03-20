import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------
# Load & sort
# ------------------------------------------------
df = pd.read_csv("data/processed/player_transfer_value_with_sentiment.csv")
df = df.sort_values(["player_name", "season_encoded"]).reset_index(drop=True)

print(f"Loaded dataset: {df.shape}  ({df['player_name'].nunique()} players)")

# ------------------------------------------------
# LSTM feature columns (must match training config)
# ------------------------------------------------
lstm_features = [
    "market_value_eur",
    "attacking_output_index",
    "injury_burden_index",
    "availability_rate",
    "vader_compound_score",
    "social_buzz_score",
]

SEQUENCE_LENGTH = 3

# ------------------------------------------------
# Scaler — fit on whole dataset (same as LSTM training)
# ------------------------------------------------
scaler = MinMaxScaler()
scaler.fit(df[lstm_features])

# ------------------------------------------------
# Load LSTM
# ------------------------------------------------
model = load_model("dashboard/lstm_model.h5", compile=False)
print("LSTM model loaded.")

# ------------------------------------------------
# Build sequences for ALL valid rows (season >= 3)
# Each sequence uses the player's own prior 3 seasons only.
# ------------------------------------------------
seq_list = []   # will become (N, 3, n_features)
idx_list = []   # original df index for each sequence

for player_name, player_df in df.groupby("player_name"):
    player_df = player_df.sort_values("season_encoded")
    scaled    = scaler.transform(player_df[lstm_features])
    orig_idx  = player_df.index.tolist()

    for i in range(SEQUENCE_LENGTH, len(player_df)):
        # seasons i-3, i-2, i-1  ──► predict season i
        seq_list.append(scaled[i - SEQUENCE_LENGTH : i])
        idx_list.append(orig_idx[i])

X_all = np.array(seq_list)                           # (N, 3, 6)
print(f"\nRunning batch LSTM prediction on {len(X_all)} sequences...")

# ------------------------------------------------
# Single batch prediction (fast — one model.predict call)
# ------------------------------------------------
preds_scaled = model.predict(X_all, batch_size=256, verbose=1)  # (N, 1)

# Inverse-transform: only market_value_eur column is meaningful
pad        = np.zeros((len(preds_scaled), len(lstm_features) - 1))
preds_full = np.concatenate([preds_scaled, pad], axis=1)
preds_eur  = scaler.inverse_transform(preds_full)[:, 0]
preds_eur  = np.maximum(preds_eur, 0)  # no negatives; no ±40% cap here

# ------------------------------------------------
# Write lstm_pred back into the original dataframe
# Rows without a prediction stay NaN (seasons 1 & 2)
# ------------------------------------------------
df["lstm_pred"] = np.nan
for df_idx, pred in zip(idx_list, preds_eur):
    df.at[df_idx, "lstm_pred"] = float(pred)

# ------------------------------------------------
# Drop rows where lstm_pred could not be generated
# ------------------------------------------------
df_enriched = df.dropna(subset=["lstm_pred"]).copy().reset_index(drop=True)

print(f"\n✅ Enriched dataset: {df_enriched.shape}")
print(f"   Seasons present:  {sorted(df_enriched['season_encoded'].unique())}")
print(f"   Players covered:  {df_enriched['player_name'].nunique()}")

residuals = df_enriched["market_value_eur"] - df_enriched["lstm_pred"]
print(f"\n   Residual mean :  €{residuals.mean():>12,.0f}")
print(f"   Residual std  :  €{residuals.std():>12,.0f}")
print(f"   Residual range:  €{residuals.min():>12,.0f}  →  €{residuals.max():,.0f}")

# ------------------------------------------------
# Save
# ------------------------------------------------
df_enriched.to_csv("data/processed/lstm_enriched.csv", index=False)
print("\n✅ Saved → data/processed/lstm_enriched.csv")
print("   Next step: python src/train_xgboost.py")