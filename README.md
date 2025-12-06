# Social Media Ad Campaign Purchase Predictor

An interactive **Streamlit** web application that predicts the expected **number of purchases** for a social media ad campaign, using a regression model trained and served from **Databricks MLflow**.  

This project is an educational end-to-end example covering:
- Data aggregation from ad events and campaign metadata  
- Model training and experiment tracking in Databricks  
- Model serving via Databricks Serving endpoints  
- Frontend deployment with Streamlit Community Cloud  

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-app-red.svg)](https://streamlit.io/)
[![Databricks](https://img.shields.io/badge/databricks-mlflow-orange.svg)](https://www.databricks.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Usage](#usage)
- [Model & Data Details](#model--data-details)
- [Limitations](#limitations)
- [Educational Extensions](#educational-extensions)
- [License](#license)

---

## Features

### Web App (Streamlit)

- Interactive UI to design a **hypothetical social media ad campaign**:
  - Campaign duration (days)
  - Total budget
  - Total number of ads
  - Platform mix (e.g., Instagram vs. Facebook)
  - Ad format mix (image, video, carousel)
  - Target gender mix
  - Number of unique interests
- Single **“Predict Purchases”** button to send the configuration to a Databricks model endpoint.
- Predicted number of purchases rounded to a whole number.
- Simple notes and limitations section describing how the model should and should not be used.
- Responsive layout and styling tuned for both desktop and mobile.

### Backend & Model

- Model is trained in **Databricks** and tracked with **MLflow**.
- Best-performing model (for the final version):
  - **Ridge Regression** with **SelectKBest** feature selection (e.g., top *k* features).
  - Evaluated with metrics like RMSE, MAE, and R².
- Model is deployed as a **Databricks Serving Endpoint** and consumed by the Streamlit app via REST API.

### Educational Focus

- Demonstrates an end-to-end ML workflow:
  1. Data aggregation from synthetic ad logs into campaign-level data.
  2. Training and selecting between multiple models (Linear Regression, Ridge/Lasso, Random Forest, Gradient Boosting, XGBoost, KNN, Neural Networks, Ensembles).
  3. Serving the best model via Databricks.
  4. Building a simple production-style UI around the model using Streamlit.
- Designed as a project for a **Master’s in AI in Business** course with an emphasis on interpretability and business applications.

---

## Project Structure

```text
campaign_purchase_app/
│
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies for the app
├── README.md                 # This file
│
├── .streamlit/
│   └── secrets.toml.example  # Template for Databricks token and endpoint config
│
└── training/
    ├── databricks_training_code.py or .dbc  # Model training & MLflow logging (Databricks)
    └── campaign_level_aggregation_notes.md  # (Optional) Documentation of feature engineering
