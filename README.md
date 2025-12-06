# Meta Ad Campaign Purchase Predictor
A Streamlit web app that predicts the expected **number of purchases** for a
Meta ad campaign using a linear regression model deployed on **Databricks MLflow**.
This app demonstrates an end-to-end workflow from data engineering and model training in Databricks to an
interactive prediction UI in Streamlit.
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-
blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.x-red.svg)](https://streamlit.io/)
[![Databricks](https://img.shields.io/badge/databricks-MLfloworange.svg)](https://www.databricks.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Model and Data](#model-and-data)
- [Limitations](#limitations)
- [Educational Context](#educational-context)
- [License](#license)
## Features
### User Interface
- Single-page **Streamlit** app with a clean, dark theme:
 - Black background and neon-purple (#8A00C4) inputs and controls
 - Large, prominent title: “Social Media Ad Campaign Purchase Predictor”
- Intuitive campaign configuration:
 - Campaign duration (days)
 - Total campaign budget
 - Total number of ads
 - Number of unique interests (e.g., “travel, gaming, food”)
 - Platform mix (Instagram vs Facebook ads)
 - Ad format mix (image, video, carousel ads)
 - Target gender mix (e.g., male / female / all)
- Clear, business-friendly output:
 - Predicted purchases shown as a **whole number**
 - Results displayed in a styled output box
### Validation and Guardrails
- If **total number of ads = 0**, the app returns **0 predicted purchases** and **does
not** call the model.
- Warning messages if:
 - Platform-specific ads exceed total number of ads
 - Ad-format counts exceed total number of ads
 - Gender-targeted ads exceed total number of ads
- Campaign duration has a minimum enforced (e.g., ≥ 30 days) to prevent unrealistic
inputs.
### Backend Integration
- Calls a **Databricks Serving Endpoint** that hosts the trained model.
- Uses a **Databricks personal access token** for secure authentication.
- Payload is constructed to match the training feature schema (campaign-level features).
## Project Structure
campaign_purchase_app/
■
■■■ streamlit_app.py # Main Streamlit application (UI + prediction logic)
■■■ requirements.txt # Python dependencies for the app
■■■ README.md # Project documentation (this file)
■
■■■ .streamlit/
 ■■■ secrets.toml.example # Example configuration for Databricks token and endpoint
### File Descriptions
- **streamlit_app.py**
 Main entry point for the application.
 - Defines all UI components (inputs, layout, notes & limitations).
 - Builds the feature vector from user input.
 - Sends a POST request to the Databricks serving endpoint and displays the prediction.
 - Includes basic validation logic and special handling for 0-ad campaigns.
- **requirements.txt**
 Contains the Python libraries needed to run the Streamlit app, for example:
 - streamlit
 - requests
 - pandas
 - numpy
- **.streamlit/secrets.toml.example**
 Template showing how to store:
 - DATABRICKS_TOKEN
 - DATABRICKS_ENDPOINT_URL
 Users should copy this to `.streamlit/secrets.toml` and fill in their own secure values.
## Prerequisites
Before running the app, you need:
- **Python 3.10 or higher** – Download from python.org
- **pip** – Python package manager (usually comes with Python)
- A **Databricks workspace** with:
 - Access to the synthetic social media advertising dataset (or equivalent tables)
 - A trained regression model logged with MLflow
 - A **Serving Endpoint** created for that model
- A **Databricks personal access token** with permission to invoke the serving endpoint
Optional but recommended:
- A **Streamlit Community Cloud** account if you want to host the app publicly.
### Verifying Prerequisites
python --version
pip --version
## Installation
### Step 1: Clone or Download the Repository
git clone https://github.com/your-username/campaign_purchase_app.git
cd campaign_purchase_app
Or download the ZIP from GitHub and extract it, then navigate into the folder.
### Step 2: Create a Virtual Environment (Recommended)
python -m venv .venv
Activate it (macOS / Linux):
source .venv/bin/activate
Activate it (Windows):
.venv\Scripts■ctivate
You should see (.venv) or similar in your terminal prompt.
### Step 3: Install Dependencies
pip install -r requirements.txt
## Configuration
The app needs two secret values to communicate with Databricks:
- DATABRICKS_TOKEN – your Databricks personal access token
- DATABRICKS_ENDPOINT_URL – the full invocation URL of your serving endpoint, typically:
https://<your-workspace>.cloud.databricks.com/serving-endpoints/<endpointname>/invocations
### Option 1: Using .streamlit/secrets.toml (Streamlit-friendly)
1. Copy the example file:
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
2. Edit .streamlit/secrets.toml:
DATABRICKS_TOKEN = "your_actual_databricks_token_here"
DATABRICKS_ENDPOINT_URL = "https://your-workspace.cloud.databricks.com/servingendpoints/your-endpoint/invocations"
Streamlit will load these values with st.secrets["DATABRICKS_TOKEN"] and
st.secrets["DATABRICKS_ENDPOINT_URL"].
### Option 2: Environment Variables (local runs)
Instead of secrets, you can export environment variables:
export DATABRICKS_TOKEN="your_actual_databricks_token_here"
export DATABRICKS_ENDPOINT_URL="https://your-workspace.cloud.databricks.com/servingendpoints/your-endpoint/invocations"
On Windows (PowerShell):
$env:DATABRICKS_TOKEN="your_actual_databricks_token_here"
$env:DATABRICKS_ENDPOINT_URL="https://your-workspace.cloud.databricks.com/servingendpoints/your-endpoint/invocations"
### Getting Your Databricks Values
- Token:
 Databricks workspace → profile (top right) → User Settings → Developer → Access tokens →
Generate new token.
- Serving Endpoint URL:
 Databricks workspace → Serving → click your endpoint → copy the Invocation URL.
### Security Notes
- Do not commit real tokens or secrets to GitHub.
- .streamlit/secrets.toml should be excluded from version control (add to .gitignore).
- Use separate tokens for development and production if needed.
## Running the Application
From the project root:
streamlit run streamlit_app.py
Streamlit will start a local server and show a URL such as:
http://localhost:8501
Open this URL in your browser to access the app.
To stop the app, press Ctrl + C in the terminal.
## Usage
1. Open the app in your browser (local or Streamlit Cloud URL).
2. Configure your campaign:
 - Enter campaign duration (days), with a minimum enforced (e.g., 30 days).
 - Enter total campaign budget.
 - Enter total number of ads.
 - Enter number of unique interests (e.g., “travel, gaming, food”).
3. Configure mix of ads:
 - Choose how many ads run on Instagram and Facebook.
 - Choose how many ads use each format (image, video, carousel).
 - Choose how many ads target each gender (male, female, all).
 If any of these counts exceed the total number of ads, the app shows a warning box.
4. Predict purchases:
 - If total number of ads = 0, the app immediately returns 0 purchases and skips the
model call.
 - Otherwise:
 - The app builds a feature vector that matches the training schema (e.g., counts by
platform, format, gender).
 - Sends a JSON payload to the Databricks serving endpoint.
 - Receives the predicted number of purchases and displays it as a rounded whole
number.
5. Review notes and limitations:
 - At the bottom of the app, a “Notes & Limitations” section explains the synthetic
nature of the data and key constraints of the model.
## Model and Data
### Data Source and Aggregation
The model is trained on a synthetic social media advertising dataset, inspired by the
Kaggle dataset:
Social Media Advertisement Performance
(https://www.kaggle.com/datasets/alperenmyung/social-media-advertisement-performance).
Raw tables in Databricks:
- ad_events – event-level logs (e.g., impressions, clicks, purchases)
- ads – ad-level information (platform, ad type, targeting attributes)
- campaigns – campaign-level configuration (start/end dates, budget, etc.)
These are transformed into a campaign-level training set. For each campaign, features
include:
- Core numeric features:
 - duration_days – campaign length (days)
 - total_budget – total budget
 - start_month, end_month – month of start and end
 - num_ads – number of ads in the campaign
 - num_unique_interests – count of distinct interests targeted
- Aggregated categorical features (via one-hot + sums per campaign):
 - Platform counts (e.g., ad_platform_Instagram, ad_platform_Facebook)
 - Format counts (e.g., ad_type_Image, ad_type_Video, ad_type_Carousel)
 - Gender counts (e.g., target_gender_Male, target_gender_Female, target_gender_All)
 - Age-group counts (e.g., target_age_group_18-24, 25-34, etc.)
The target variable is:
- num_purchases – the number of purchase events attributed to each campaign.
### Model Training (in Databricks)
Training is done in Databricks using scikit-learn and MLflow:
- Multiple models are trained and logged:
 - Linear Regression + SelectKBest (with different k values)
 - Ridge and Lasso Regression
 - Random Forest, Gradient Boosting, XGBoost
 - K-Nearest Neighbors Regression
 - Neural Networks (MLPRegressor) with different architectures and activation functions
 - Ensemble (VotingRegressor) combinations
The best-performing model chosen for deployment in this project is a Ridge Regression
pipeline with:
1. Median imputation and feature scaling
2. SelectKBest with mutual information to select top features (e.g., k=20)
3. Ridge Regression with tuned regularization alpha
Performance on the held-out synthetic test set:
- RMSE (root mean squared error): on the order of ~5–6 purchases
- MAE (mean absolute error): on the order of ~4–5 purchases
These metrics mean that, on average, the model’s predictions are within a few purchases of
the simulated “true” values in the test set.
The trained pipeline is logged to MLflow with:
- Input example
- Signature (schema)
- Artifacts (model, residual plots, metrics)
It is then exposed through a Databricks Serving Endpoint which the Streamlit app calls.
## Limitations
1. Synthetic, single-company data
 - The dataset simulates one fictional company’s campaigns.
 - Real companies, industries, and platforms may behave very differently.
2. Limited feature set
 - The model does not know about:
 - Product type or category
 - Product price or margins
 - Brand strength or creative quality
 - It only sees high-level campaign settings and targeting variables.
3. Correlation, not causation
 - The model learns patterns in historical synthetic data.
 - A higher predicted purchase count with higher budget or more ads does not guarantee
the same lift in real life.
4. No external factors
 - No seasonality, competitive activity, platform algorithm changes, or macroeconomic
conditions are modeled.
5. Data volume and coverage
 - The original campaign sample is relatively small; even with synthetic augmentation,
it may not cover all possible campaign setups.
 - Predictions can be unreliable for extreme or unusual inputs.
6. Point estimates only
 - The app returns a single predicted number of purchases.
 - It does not show uncertainty ranges or best/worst-case scenarios.
7. Best for “similar” campaigns
 - The model is most appropriate for campaigns that resemble those in the training data.
 - Very different setups (new platforms, unusual budget levels, very long durations)
should be interpreted with caution.
8. Educational use only
 - This project is built for coursework and experimentation.
 - Real budget decisions should combine this kind of modeling with expert judgment, live
experiments (A/B tests), and real performance data.
## Educational Context
This project demonstrates an end-to-end applied ML workflow:
- Data engineering and feature creation in Databricks (from raw event-level data to
campaign-level features).
- Model selection and tuning across multiple regression families using scikit-learn and
MLflow.
- Model deployment via a Databricks Serving Endpoint.
- Front-end integration with a Streamlit app that allows non-technical users to explore
“what-if” scenarios and see predicted purchase outcomes.
It is intended as an example of how machine learning can support marketing and campaign
planning in a business context, while also highlighting the importance of data quality,
feature coverage, and transparency about limitations.
## License
This project is provided as educational material under the MIT License.
You may use, modify, and share it for learning and non-production purposes.
