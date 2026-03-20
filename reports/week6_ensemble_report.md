# TransferIQ — Week 6 Ensemble Model Report

**Project:** TransferIQ — Football Player Market Value Prediction
**Week:** 6 | **Date:** March 2026

---

## 1. What Was Built

Upgraded the Week 5 LSTM into a full LSTM + XGBoost ensemble.

**Pipeline:**
- LSTM runs first → generates `lstm_pred` (market value prediction per player)
- `lstm_pred` added as a new column to the dataset
- XGBoost trained on enriched dataset using `lstm_pred` + 14 other features
- Final output = XGBoost direct market value prediction

---

## 2. Data Used

| Stream | Features |
|---|---|
| Performance | attacking_output_index, injury_burden_index, availability_rate, goals_per90, assists_per90, minutes_played |
| Market Trends | lstm_pred, season_encoded, current_age, age_decay_factor |
| Social Sentiment | vader_compound_score, social_buzz_score |

---

## 3. Validation Strategy

Train on seasons 3–4, test on season 5 (unseen). Season-based split used — not random — to prevent future data leaking into training.

| Set | Seasons | Rows |
|---|---|---|
| Training | 3 and 4 | 2,000 |
| Validation | 5 (2023/24) | 1,000 |

---

## 4. Results

| Metric | LSTM | Ensemble | Change |
|---|---|---|---|
| RMSE | €4,094,641 | €3,873,196 | −5.4% ✅ |
| MAE | €3,385,055 | €1,653,583 | −51.2% ✅ |

MAE cut in half — typical per-player error dropped from €3.39M to €1.65M.

---

## 5. Feature Importance (Top 5)

| Feature | Importance | Meaning |
|---|---|---|
| lstm_pred | 58.7% | LSTM trend — dominant signal |
| social_buzz_score | 21.1% | Transfer media attention |
| minutes_played | 8.1% | Manager confidence proxy |
| attacking_output_index | 3.4% | Goals + assists |
| goal_contributions_per90 | 1.7% | Per-90 efficiency |

---

## 6. Overfitting Check

No overfitting detected. Error is consistent across training and validation seasons. Regularisation used: `min_child_weight=20`, `reg_alpha=0.1`, `reg_lambda=1.0`.

---

## 7. Deliverables

| Deliverable | Status |
|---|---|
| XGBoost model | ✅ dashboard/xgb_model.pkl |
| LSTM feature script | ✅ src/generate_lstm_features.py |
| Ensemble training script | ✅ src/train_xgboost.py |
| Enriched dataset | ✅ data/processed/lstm_enriched.csv |
| Streamlit dashboard | ✅ dashboard/app.py |

---

## 8. Milestone Compliance

| Requirement | Status |
|---|---|
| XGBoost ensemble implemented | ✅ |
| LSTM integrated with XGBoost | ✅ lstm_pred at 58.7% importance |
| All 3 data streams combined | ✅ 15 features |
| Tested on validation dataset | ✅ Season 5, 1,000 players |
| Ensemble improves over LSTM | ✅ RMSE −5.4%, MAE −51.2% |