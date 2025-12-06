# Social Media Ad Campaign Purchase Predictor

A Streamlit web application that predicts the expected **number of purchases** for a social media ad campaign using a regression model deployed on **Databricks MLflow**.  
Built as part of a Master’s-level project in **AI in Business**, this app demonstrates an end-to-end workflow from data engineering and model training in Databricks to an interactive prediction UI in Streamlit.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.x-red.svg)](https://streamlit.io/)
[![Databricks](https://img.shields.io/badge/databricks-MLflow-orange.svg)](https://www.databricks.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

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

---

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

- If **total number of ads = 0**, the app returns **0 predicted purchases** and **does not** call the model.
- Warning messages if:
  - Platform-specific ads exceed total number of ads
  - Ad-format counts exceed total number of ads
  - Gender-targeted ads exceed total number of ads
- Campaign duration has a minimum enforced (e.g., ≥ 30 days) to prevent unrealistic inputs.

### Backend Integration

- Calls a **Databricks Serving Endpoint** that hosts the trained model.
- Uses a **Databricks personal access token** for secure authentication.
- Payload is constructed to match the training feature schema (campaign-level features).

---

## Project Structure

```text
campaign_purchase_app/
│
├── streamlit_app.py           # Main Streamlit application (UI + prediction logic)
├── requirements.txt           # Python dependencies for the app
├── README.md                  # Project documentation (this file)
│
└── .streamlit/
    └── secrets.toml.example   # Example configuration for Databricks token and endpoint

