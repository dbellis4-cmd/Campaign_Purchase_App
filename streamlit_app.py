import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

# --------------------------------------------------------
# Databricks serving endpoint
# --------------------------------------------------------
ENDPOINT_URL = "https://dbc-74bcb363-c4f6.cloud.databricks.com/serving-endpoints/ridge_regression/invocations"


# --------------------------------------------------------
# Helper: get Databricks token (from environment or secrets)
# --------------------------------------------------------
def get_databricks_token():
    tok = os.getenv("DATABRICKS_TOKEN")
    if tok:
        return tok
    return st.secrets["DATABRICKS_TOKEN"]


# --------------------------------------------------------
# Helper: load feature columns from Training_Columns.csv
# --------------------------------------------------------
@st.cache_resource
def load_feature_cols():
    df = pd.read_csv("Training_Columns.csv")
    first_col = df.columns[0]
    return df[first_col].tolist()


FEATURE_COLS = load_feature_cols()


def call_databricks_endpoint(df: pd.DataFrame) -> float:
    token = get_databricks_token()

    payload = {
        "dataframe_split": df.to_dict(orient="split")
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    resp = requests.post(ENDPOINT_URL, headers=headers, json=payload)
    resp.raise_for_status()
    js = resp.json()
    preds = js.get("predictions", None)
    if preds is None:
        raise RuntimeError(f"Unexpected response format: {js}")
    return float(preds[0])


# --------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------
st.title("Campaign Purchase Predictor")

st.write(
    "Set up a hypothetical ad campaign, then click **Predict purchases**. "
    "Behind the scenes, this calls a Ridge regression model served on Databricks."
)

st.sidebar.header("Campaign setup")

# Core campaign features
duration_days = st.sidebar.slider("Campaign duration (days)", 1, 365, 30)
total_budget = st.sidebar.number_input("Total budget ($)", min_value=0.0, value=1000.0, step=100.0)
num_ads = st.sidebar.number_input("Number of ads", min_value=1, value=5, step=1)
num_unique_interests = st.sidebar.number_input("Number of unique interests", min_value=1, value=3, step=1)

start_month = st.sidebar.selectbox("Start month (1–12)", list(range(1, 13)), index=0)
end_month = st.sidebar.selectbox("End month (1–12)", list(range(1, 13)), index=0)

# Platforms
st.subheader("Ad-platform breakdown (number of ads)")
facebook_ads = st.number_input("Facebook ads", min_value=0, step=1, value=0)
instagram_ads = st.number_input("Instagram ads", min_value=0, step=1, value=0)

# Ad types
st.subheader("Ad-type breakdown (number of ads)")
carousel_ads = st.number_input("Carousel ads", min_value=0, step=1, value=0)
image_ads    = st.number_input("Image ads", min_value=0, step=1, value=0)
stories_ads  = st.number_input("Stories ads", min_value=0, step=1, value=0)
video_ads    = st.number_input("Video ads", min_value=0, step=1, value=0)

# Genders
st.subheader("Target gender breakdown (number of ads)")
male_ads       = st.number_input("Male-targeted ads", min_value=0, step=1, value=0)
female_ads     = st.number_input("Female-targeted ads", min_value=0, step=1, value=0)
all_gender_ads = st.number_input("All-gender ads", min_value=0, step=1, value=0)

# Age groups
st.subheader("Target age-group breakdown (number of ads)")
age_18_24_ads = st.number_input("Age 18–24 ads", min_value=0, step=1, value=0)
age_25_34_ads = st.number_input("Age 25–34 ads", min_value=0, step=1, value=0)
age_35_44_ads = st.number_input("Age 35–44 ads", min_value=0, step=1, value=0)
age_all_ads   = st.number_input("All-age ads",   min_value=0, step=1, value=0)

st.caption("Any feature not shown here is set to zero. The model still uses all the columns it was trained on.")


if st.button("Predict purchases"):
    row = {c: 0 for c in FEATURE_COLS}

    # Core numeric features
    if "duration_days" in row:
        row["duration_days"] = duration_days
    if "total_budget" in row:
        row["total_budget"] = total_budget
    if "num_ads" in row:
        row["num_ads"] = num_ads
    if "num_unique_interests" in row:
        row["num_unique_interests"] = num_unique_interests
    if "start_month" in row:
        row["start_month"] = start_month
    if "end_month" in row:
        row["end_month"] = end_month

    # Platforms
    if "ad_platform_Facebook" in row:
        row["ad_platform_Facebook"] = facebook_ads
    if "ad_platform_Instagram" in row:
        row["ad_platform_Instagram"] = instagram_ads

    # Ad types
    if "ad_type_Carousel" in row:
        row["ad_type_Carousel"] = carousel_ads
    if "ad_type_Image" in row:
        row["ad_type_Image"] = image_ads
    if "ad_type_Stories" in row:
        row["ad_type_Stories"] = stories_ads
    if "ad_type_Video" in row:
        row["ad_type_Video"] = video_ads

    # Genders
    if "target_gender_Male" in row:
        row["target_gender_Male"] = male_ads
    if "target_gender_Female" in row:
        row["target_gender_Female"] = female_ads
    if "target_gender_All" in row:
        row["target_gender_All"] = all_gender_ads

    # Age groups
    if "target_age_group_18-24" in row:
        row["target_age_group_18-24"] = age_18_24_ads
    if "target_age_group_25-34" in row:
        row["target_age_group_25-34"] = age_25_34_ads
    if "target_age_group_35-44" in row:
        row["target_age_group_35-44"] = age_35_44_ads
    if "target_age_group_All" in row:
        row["target_age_group_All"] = age_all_ads

    X_new = pd.DataFrame([row])

    with st.spinner("Calling Databricks endpoint..."):
        try:
            pred = call_databricks_endpoint(X_new)

            st.markdown("### Predicted purchases")
            st.metric("Estimated purchases", f"{pred:.1f}")

            st.markdown("#### Features sent to the model")
            st.dataframe(X_new)
        except Exception as e:
            st.error(f"Error calling Databricks endpoint: {e}")
