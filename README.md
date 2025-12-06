Social Media Ad Campaign Purchase Predictor

An interactive Streamlit web application that predicts the expected number of purchases for a social media ad campaign, using a regression model trained and served from Databricks MLflow.

This project is an educational end-to-end example covering:

Data aggregation from ad events and campaign metadata

Model training and experiment tracking in Databricks

Model serving via Databricks Serving endpoints

Frontend deployment with Streamlit Community Cloud








Table of Contents

Features

Project Structure

How It Works

Prerequisites

Local Setup

Configuration

Running the App

Usage

Model & Data Details

Limitations

Educational Extensions

License

Features
Web App (Streamlit)

Interactive UI to design a hypothetical social media ad campaign:

Campaign duration (days)

Total budget

Total number of ads

Platform mix (e.g., Instagram vs. Facebook)

Ad format mix (image, video, carousel)

Target gender mix

Number of unique interests

Single “Predict Purchases” button to send the configuration to a Databricks model endpoint.

Predicted number of purchases rounded to a whole number.

Guardrails such as:

If the total number of ads is 0, the app returns 0 predicted purchases (does not call the model).

Warnings when platform/format/gender counts exceed the total number of ads.

Notes & limitations section explaining how the model should and should not be used.

Responsive layout and custom styling tuned for both desktop and mobile.

Backend & Model

Model is trained in Databricks and tracked with MLflow.

Multiple models are trained and compared (Linear Regression, Ridge/Lasso, Random Forest, Gradient Boosting, XGBoost, KNN, Neural Networks, Ensembles).

Final selected model:

Ridge Regression with SelectKBest feature selection (e.g., top k features based on mutual information).

Model is deployed as a Databricks Serving Endpoint and consumed by the Streamlit app via REST API.

Educational Focus

Demonstrates an end-to-end ML workflow:

Data aggregation from synthetic ad event logs into campaign-level data.

Model training and selection using MLflow experiments.

Model serving via Databricks.

Building a simple production-style UI around the model using Streamlit.

Designed as a project for a Master’s in AI in Business course, with emphasis on:

Interpretable linear models

Business applications (pre-launch forecasting, budgeting, what-if analysis)

Clear communication of limitations

Project Structure
campaign_purchase_app/
│
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies for the app
├── README.md                     # This file
│
├── .streamlit/
│   └── secrets.toml.example      # Template for Databricks token and endpoint config
│
└── training/                     # (Optional) reference training assets
    ├── databricks_training_code.py   # Model training & MLflow logging (exported from Databricks)
    └── campaign_aggregation_notes.md # Documentation of feature engineering


Note: The primary training code and experiments live inside Databricks notebooks. This repo focuses on the app and configuration needed to call the served model, plus a reference of the training logic.

How It Works

Data aggregation (Databricks)
Event-level data (ad_events, ads, campaigns) is aggregated into campaign-level features such as:

duration_days, total_budget

start_month, end_month

num_ads, num_unique_interests

Counts of ads by platform, format, target gender, and age group

Model training (Databricks)
Multiple regression models are trained and logged with MLflow, for example:

Linear Regression + SelectKBest (different k values)

Ridge & Lasso Regression

Random Forest, Gradient Boosting, XGBoost

KNN Regression

Neural Networks (MLPRegressor) with different activation functions (ReLU, tanh, logistic)

Ensemble Voting Regressors

The final chosen model is a Ridge Regression + SelectKBest variant that achieves a good balance of:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² (explained variance)

Model serving (Databricks)

The best model is registered with MLflow / Unity Catalog.

A Databricks Serving Endpoint is created for that model.

The endpoint accepts JSON input and returns a single numeric prediction: expected number of purchases.

Streamlit app (this repo)

The app collects user inputs that map to the campaign-level features.

The inputs are transformed into a feature vector that matches the training schema.

A request is sent to the Databricks endpoint using the configured token and URL.

The predicted number of purchases is displayed in the UI.

Prerequisites

Python 3.10 or higher

pip (Python package manager)

A Databricks workspace with:

The synthetic social media dataset (or equivalent tables).

A trained and deployed MLflow model (Serving Endpoint).

A Databricks personal access token with permission to call the endpoint.

(Optional) Streamlit Community Cloud account if you want to host the app publicly.

Verify basic tools:

python --version
pip --version

Local Setup

Clone the repository

git clone https://github.com/your-username/campaign_purchase_app.git
cd campaign_purchase_app


Create and activate a virtual environment (recommended)

python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate


Install dependencies

pip install -r requirements.txt

Configuration

The app needs credentials to call your Databricks model endpoint:

DATABRICKS_TOKEN – personal access token

DATABRICKS_ENDPOINT_URL – serving endpoint invocation URL, e.g.
https://<workspace-url>/serving-endpoints/<endpoint-name>/invocations

Option 1: Streamlit secrets (for local and Streamlit Cloud)

Create .streamlit/secrets.toml:

DATABRICKS_TOKEN = "your_databricks_token_here"
DATABRICKS_ENDPOINT_URL = "https://your-workspace.cloud.databricks.com/serving-endpoints/your-endpoint/invocations"

Option 2: Environment variables (local only)
export DATABRICKS_TOKEN="your_databricks_token_here"
export DATABRICKS_ENDPOINT_URL="https://your-workspace.cloud.databricks.com/serving-endpoints/your-endpoint/invocations"


On Windows (PowerShell):

$env:DATABRICKS_TOKEN="your_databricks_token_here"
$env:DATABRICKS_ENDPOINT_URL="https://your-workspace.cloud.databricks.com/serving-endpoints/your-endpoint/invocations"


Important: Never commit real tokens to GitHub. Keep them in secrets.toml or environment variables.

Running the App

From the project root:

streamlit run streamlit_app.py


Streamlit will print a local URL, typically:

http://localhost:8501

Open that URL in your browser to use the app.

Usage

Set up the campaign

On the main page, you can configure:

Campaign duration (days)
Enter the planned duration of the campaign. The app may enforce a minimum (e.g., 30 days).

Total campaign budget
The overall spend allocated to the campaign.

Total number of ads
The total count of ad creatives across platforms and formats.

Number of unique interests
For example: travel, gaming, food, fitness, etc.

Platform & format mix

Specify how many ads will run on each platform (e.g., Instagram, Facebook).

Specify how many will be image, video, or carousel ads.

Specify the gender mix of targeted users.

The app includes checks so that:

The sum of platform ads does not exceed total ads.

The sum of format ads does not exceed total ads.

The sum of gender-targeted ads does not exceed total ads.
If any of these conditions are violated, the app shows a warning.

Predict purchases

Click “Predict Purchases” to send your configuration to the Databricks endpoint.

If total_ads = 0, the app returns 0 purchases and does not call the model.

Otherwise, it:

Builds a feature vector matching the training columns.

Sends a JSON payload to the serving endpoint.

Receives the prediction and displays the number of purchases (rounded to the nearest whole number).

Review limitations

The bottom section, “Notes & Limitations”, summarizes:

Synthetic nature of the data.

Missing information (product, price, creative quality).

That this is an educational tool, not a production forecasting system.

Model & Data Details
Data

Based on a synthetic dataset of social media ad campaigns.

Original source inspiration: Social Media Advertisement Performance dataset on Kaggle.

Tables:

ad_events – event-level logs (impressions, clicks, purchases, etc.)

ads – individual ad creatives with metadata (platform, type, targeting)

campaigns – campaign-level settings (start/end dates, duration, budget)

Target variable

num_purchases
Number of purchase events attributed to each campaign.

Example feature groups (campaign-level)

Core numeric features

duration_days

total_budget

start_month, end_month

num_ads

num_unique_interests

Aggregated categorical features

Counts of ads by platform:

e.g., ad_platform_Facebook, ad_platform_Instagram, etc.

Counts of ads by format:

e.g., ad_type_Image, ad_type_Video, ad_type_Carousel

Counts of ads by target gender:

e.g., target_gender_Male, target_gender_Female, target_gender_All

Counts of ads by age group:

e.g., target_age_group_18-24, 25-34, etc.

Best model (example)

Model type: Ridge Regression with SelectKBest feature selection

Key elements:

Standardization of numeric features.

Mutual information-based selection of the k most predictive features.

Ridge penalization (L2 regularization) to reduce overfitting and stabilize coefficients.

Metrics on synthetic test set (approximate):

RMSE ≈ 5–6 purchases

MAE ≈ 4–5 purchases

R² ≈ moderate fit

Interpretation:
On average, the model’s predictions are off by about 4–6 purchases per campaign in the synthetic data. This is reasonable for rough scenario analysis but not exact financial forecasting.

Limitations

This app is for education and experimentation, not for real-world budget allocation or forecasting. Key limitations:

Synthetic, single-company data
The model is trained on a fictional company’s simulated campaigns. Real companies, industries, and platforms may behave very differently.

Limited features (no product/price/creative details)
The model does not know:

What product is being sold.

The product’s price or margin.

The quality of the creative (copy, images, video editing, brand strength).

Correlation, not causation
If the model predicts more purchases with higher budget or more ads, that reflects patterns in the data—not guaranteed cause-and-effect in the real world.

No external factors
The model does not account for:

Seasonality or holidays

Competitor activity

Algorithm changes on platforms

Macro-economic shifts
These can significantly change performance in reality.

Limited training size
Original dataset included a relatively small number of campaigns, even if synthetic augmentation is used. Results should be seen as illustrative, not production-grade.

Single point estimate
The model returns one predicted purchase count. There is no uncertainty interval, best-case, or worst-case estimate.

Best for “similar” campaigns only
The model is most reliable for campaigns that look similar to those in the training data. Extreme durations, budgets, or very unusual setups may yield unreliable predictions.

Educational tool only
This app is built for a graduate course project. Any real business decisions should rely on proper experimentation, real data, and expert judgment.

Educational Extensions

Ideas for extending the project:

ROAS / CPA / Profit modeling

Add product price and cost data.

Predict revenue or profit alongside purchases.

Compute ROAS (Return on Ad Spend) and CPA (Cost per Acquisition).

Scenario comparison (A/B campaign setups)

Let users define two or more campaign setups.

Show predicted purchases side-by-side.

Uncertainty modeling

Add simple prediction intervals (e.g., via residual error analysis).

Show a range instead of a single point estimate.

Additional models / endpoints

Train alternative models (e.g., Gradient Boosting, XGBoost).

Add a model selector in the UI to switch between endpoints.

Logging and dashboards

Log user-generated scenarios (anonymously) to Databricks.

Create dashboards of the most common configurations users try.

License

This project is provided as educational material under the MIT License.
See the LICENSE file for full details.
