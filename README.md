# Ad-Level Purchase Prediction (B2C Ads) — End-to-End ML + App
Forecast **purchases per ad** from planned campaign settings to drive **budget allocation,
guardrails, and ROAS planning**. Trains on ad/campaign configuration and historical event
logs; deploys a simple web app for what-if planning.
[![Python
3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Databricks/MLflow](https://img.shields.io/badge/MLflow-enabled-brightgreen)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/app-Streamlit-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
---
## 1) Overview
- **Goal:** Predict the **number of purchases** an ad will generate using only *pre-launch*
inputs (budget, duration, platform, creative type, targeting).
- **Why it matters:** Convert plans into forecasted outcomes to **reallocate budget**, set
**CPA/ROAS guardrails**, and plan **what-if** scenarios before spending.
- **Single target (MVP):** `Purchase` (count).
Later extensions: add `Impression`, `Click`, or decompose the funnel (CTR/CVR).
---
## 2) Data
Place the three CSVs in `data/raw/`:
- `campaigns.csv` — campaign metadata
- `campaign_id` (PK), `name`, `start_date`, `end_date`, `duration_days`, `total_budget`
- `ads.csv` — ad setup
- `ad_id` (PK), `campaign_id` (FK), `ad_platform`, `ad_type`, `target_gender`,
`target_age_group`, `target_interests`
- `ad_events.csv` — user events
- `event_id` (PK), `ad_id` (FK), `timestamp`, `day_of_week`, `time_of_day`, `event_type` ∈
{Impression, Click, Purchase, Like, Share, Comment}
### Label construction (ad level)
Aggregate `ad_events.csv` to counts per `ad_id` (pivot on `event_type`) and join back to
`ads.csv` and `campaigns.csv`.
**Targets:** `Impression`, `Click`, `Purchase` (MVP uses `Purchase`).
> This repo also supports a prepared file `data/processed/ad_level_model_ready.csv` with
engineered features and targets:
> - Numeric: `log_duration_days`, `log_total_budget`, `log_budget_per_day`, `n_interests`
> - One-hots: `ad_platform_*`, `ad_type_*`, `target_gender_*`, `target_age_group_*`
> - Multi-hot: `interest__*` (top-K interest tokens)
> - Targets: `Impression`, `Click`, `Purchase`
---
## 3) Problem Definition (Analytical Framing)
- **Business objective:** Maximize conversions per dollar by forecasting **purchases per ad**
ahead of launch to guide budget and targeting decisions.
- **ML task:** Supervised **regression** on non-negative counts (single target = `Purchase`).
- **Inputs at prediction time:** Only **planned** ad/campaign settings (no realized
performance).
- **Evaluation:** MAE, RMSE, SMAPE on purchases; report zero-rate and MAE on non-zero subset.
---
## 4) Modeling Strategies (pick one to start)
### A) Direct count regression (simple baseline)
- Model `Purchase` directly from features (tree boosting with **Poisson**/NB loss, or
SVR/Linear as baselines).
- Pros: Quick to build and deploy.
- Cons: Zero inflation can be tough.
### B) Hurdle (two-step) for zeros (recommended for purchases)
1. **Step 1:** Binary classifier — any purchase?
2. **Step 2:** Count model (Poisson/NB/GBM) on rows with purchases > 0.
- Pros: Handles many zeros, often improves accuracy on the tail.
> In class, start with **A (single target, direct regression)** for clarity. If zeros
dominate, move to **B**.
---
## 5) Repository Structure
```
.
■■ data/
■ ■■ raw/
■ ■ ■■ campaigns.csv
■ ■ ■■ ads.csv
■ ■ ■■ ad_events.csv
■ ■■ processed/
■ ■■ ad_level_events_summary.csv
■ ■■ ad_level_model_ready.csv
■■ notebooks/
■ ■■ 01_eda_and_featurization.ipynb
■ ■■ 02_train_eval_purchase_model.ipynb
■■ src/
■ ■■ prepare_data.py
■ ■■ train.py
■ ■■ evaluate.py
■ ■■ app.py # Streamlit app
■■ requirements.txt
■■ README.md
■■ LICENSE
```
---
## 6) Quickstart
### Environment
```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
### Prepare data (from raw → processed)
```bash
python -m src.prepare_data \
--campaigns data/raw/campaigns.csv \
--ads data/raw/ads.csv \
--events data/raw/ad_events.csv \
--out_ad_level data/processed/ad_level_model_ready.csv
```
### Train (single-target: purchases)
```bash
python -m src.train \
--train_csv data/processed/ad_level_model_ready.csv \
--target Purchase \
--model gbm_poisson \
--group_split campaign_id \
--mlflow_experiment "ad_purchase_mvp"
```
### Evaluate
```bash
python -m src.evaluate \
--test_csv data/processed/ad_level_model_ready.csv \
--target Purchase
```
### Run the app (what-if planner)
```bash
streamlit run src/app.py
```
---
## 7) Streamlit App (Demo Flow)
- **Inputs:** `duration_days`, `total_budget`, `ad_platform`, `ad_type`, `target_gender`,
`target_age_group`, `target_interests`
- **Outputs:** Predicted **Purchases**, derived **CPA** and **ROAS** given a user-entered
AOV.
- **What-if:** Budget slider and platform/targeting toggles; show changes in
Purchases/CPA/ROAS.
- **Explainability:** Feature importances (GBM), partial dependence, or SHAP for top drivers.
---
## 8) Evaluation & Splits
- **Leakage control:** Use only planned settings as features. Do not feed realized events as
inputs.
- **Split strategy:**
- Time-based by `start_date` (train earlier, test later), or
- Grouped split by `campaign_id` (no ads from the same campaign in both sets).
- **Metrics to report:** MAE, RMSE, SMAPE on Purchases; MAE on non-zero subset; % of ads with
absolute error ≤ K.
---
## 9) From Prediction to Business KPIs
With budget **B**, predicted purchases **■■**, and average order value **AOV**:
- **CPA** = `B / ■■`
- **ROAS** = `(■■ × AOV) / B`
- **POAS (margin ROAS)** = `(■■ × AOV × gross_margin) / B`
Use thresholds (e.g., **CPA ≤ $20**, **ROAS ≥ 2.0×**) to enable go/no-go, reallocate spend,
or adjust targeting before launch.
---
## 10) MLflow (optional but recommended)
- Track runs: parameters (features, model, seed), metrics (MAE/SMAPE), artifacts (plots), and
the serialized model.
- Register the best model for the app to load by name/version.
---
## 11) Reproducibility
- Fix random seeds and document library versions (see `requirements.txt`).
- Persist train/val/test indices used in final evaluation.
- Log feature schema (column order and dtypes) along with the model.
---
## 12) Roadmap
- Add **Impressions** and **Clicks** (multi-output) or a **funnel** (Impr → CTR → CVR).
- Quantile models for P10/P50/P90 purchase forecasts (risk-aware planning).
- Per-platform **AOV** tables to refine ROAS.
- Live data connector for post-launch monitoring and drift alerts.
---
## 13) Ethical & Practical Notes
- Use demographic targeting responsibly; avoid sensitive attributes and proxy bias.
- Validate model fairness by comparing error distributions across age/gender segments.
- Provide clear opt-out and transparency for targeting where required.
---
## 14) License
MIT — see `LICENSE`.
---
## Citations & Acknowledgments
This project structure is adapted from common ML engineering patterns (feature stores,
grouped splits, MLflow tracking) and simplified for instructional purposes.
