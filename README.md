# 🚀 Transfer-IQ: Player Market Value Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://transfer-iq.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)

> **Dynamic Player Transfer Value Prediction using AI and Multi-source Data.**

## 🌐 Live Application  
👉 **[Try Transfer-IQ on Streamlit](https://transfer-iq.streamlit.app/)**

---

## 📌 Project Overview
**Transfer-IQ** is a data-driven analytics project designed to predict and analyze football player market values. By integrating multi-source data—including on-pitch performance metrics, injury histories, and social media sentiment—the project aims to build a robust, machine-learning-ready dataset that uncovers the hidden factors influencing a player's valuation in the modern transfer market.

## 🗂️ Data Sources
To build a comprehensive profile for each player, data was aggregated from multiple domains:
- **StatsBomb:** Player performance and match data.
- **Transfermarkt:** Historical and current market value valuations.
- **Social Media:** Public sentiment signals processed via NLP (VADER / TextBlob).
- **Injury Records:** Player availability, historical absences, and future risk assessment.

---

## 🧠 Key Engineered Features
Our dataset (~270 players) relies on advanced feature engineering to capture true player value:
- **Goal Conversion Rate & Minutes Per Goal:** Efficiency metrics for attackers.
- **Performance Index:** A normalized score combining various on-ball and off-ball actions.
- **Injury Risk Score:** A predictive metric based on past physical reliability.
- **Sentiment Polarity:** Public perception scores to gauge marketability and reputation.

---

## ⚙️ Project Roadmap & Work Completed

### Phase 1: Data Aggregation & Cleaning (Weeks 1–2)
- [x] Multi-source data collection and alignment.
- [x] Player identity standardization across different APIs and datasets.
- [x] Missing value handling, imputation, and deduplication.
- [x] Initial base feature engineering.

### Phase 2: Advanced Processing (Weeks 3–4)
- [x] NLP Sentiment analysis using VADER and TextBlob.
- [x] Creation of advanced performance efficiency metrics.
- [x] Injury risk feature generation.
- [x] Finalized modeling-ready dataset.

### Phase 3: Modeling & Deployment (Current/Next Steps)
- [ ] Predictive ML modeling (Regression/XGBoost).
- [ ] Model evaluation and hyperparameter tuning.
- [x] Streamlit web application deployment.

---

## 🛠️ Tech Stack
| Category | Technologies |
|---|---|
| **Language** | Python |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **NLP** | VADER, TextBlob |
| **Deployment** | Streamlit |
| **Version Control** | Git & GitHub |

---

## 📁 Project Structure

```text
Transfer-IQ/
│
├── data/
│   ├── raw/             # Original, immutable data dumps
│   ├── interim/         # Intermediate data that has been transformed
│   └── processed/       # The final, canonical datasets for modeling
│
├── notebooks/           # Jupyter notebooks for EDA and prototyping
├── src/                 # Source code for data pipeline and feature engineering
├── reports/             # Generated analysis as HTML, PDF, or markdown
├── app.py               # Streamlit application entry point
└── README.md            # Project documentation
```

---

## 💻 Local Setup & Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/springboardmentor67/Transfer-IQ.git
   cd Transfer-IQ
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

---

## 👤 Author
**Abhishek Kumar**  
*Infosys Springboard Internship Project*
