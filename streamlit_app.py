import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

# --------------------------------------------------------
# Page config
# --------------------------------------------------------
st.set_page_config(
    page_title="Campaign Purchase Predictor",
    page_icon="📈",
    layout="wide",
)

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
    """Send a one-row DataFrame to the Databricks serving endpoint."""
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
# Simple CSS tweak for a cleaner look
# --------------------------------------------------------
st.markdown(
    """
    <style>
    .main > div {
        max-width: 1100px;
        margin: 0 auto;
    }
    .stMetric {
        background-color: #f5f7fb;
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------
# Header
# --------------------------------------------------------
st.markdown("### 📊 Campaign Purchase Predictor")
st.markdown(
    """
    Configure a hypothetical digital ad campaign and click **Predict purchases**.
    The input features are sent to a Ridge regression model served on **Databricks**,
    which returns the predicted number of purchases at the campaign level.
    """
)

st.sidebar.header("Campaign setup")

# --------------------------------------------------------
# Sidebar – core campaign knobs
# --------------------------------------------------------
with st.sidebar.expander("Campaign basics", expanded=True):
    duration_days = st.slider("Campaign duration (days)", 1, 365, 30)
    total_budget = st.number_input("Total budget ($)", min_value=0.0, value=1000.0, step=100.0)
    num_ads = st.number_input("Number of ads", min_value=1, value=5, step=1)
    num_unique_interests = st.number_input("Number of unique interests", min_value=1, value=3, step=1)

with st.sidebar.expander("Timing", expanded=False):
    start_month = st.selectbox("Start month (1–12)", list(range(1, 13)), index=0)
    end_month = st.selectbox("End month (1–12)", list(range(1, 13)), index=0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Tip:**\n\nTry different budget and mix of platforms/ad types to see how the expected purchases change."
)

# --------------------------------------------------------
# Main layout – three sections in columns
# --------------------------------------------------------
col_platforms, col_types = st.columns(2)

with col_platforms:
    st.subheader("Ad-platform mix")
    st.caption("Number of ads running on each platform.")
    facebook_ads = st.number_input("Facebook ads", min_value=0, step=1, value=0)
    instagram_ads = st.number_input("Instagram ads", min_value=0, step=1, value=0)

with col_types:
    st.subheader("Ad-format mix")
    st.caption("How your creatives are split across formats.")
    carousel_ads = st.number_input("Carousel ads", min_value=0, step=1, value=0)
    image_ads    = st.number_input("Image ads", min_value=0, step=1, value=0)
    stories_ads  = st.number_input("Stories ads", min_value=0, step=1, value=0)
    video_ads    = st.number_input("Video ads", min_value=0, step=1, value=0)

st.markdown("---")

col_gender, col_age = st.columns(2)

with col_gender:
    st.subheader("Target gender mix")
    st.caption("Number of ads aimed at each gender segment.")
    male_ads       = st.number_input("Male-targeted ads", min_value=0, step=1, value=0)
    female_ads     = st.number_input("Female-targeted ads", min_value=0, step=1, value=0)
    all_gender_ads = st.number_input("All-gender ads", min_value=0, step=1, value=0)

with col_age:
    st.subheader("Target age-group mix")
    st.caption("Number of ads aimed at each age band.")
    age_18_24_ads = st.number_input("Age 18–24 ads", min_value=0, step=1, value=0)
    age_25_34_ads = st.number_input("Age 25–34 ads", min_value=0, step=1, value=0)
    age_35_44_ads = st.number_input("Age 35–44 ads", min_value=0, step=1, value=0)
    age_all_ads   = st.number_input("All-age ads",   min_value=0, step=1, value=0)

st.markdown(
    "<small>Any feature not controlled in the UI is set to 0. "
    "The model still uses the full feature vector it was trained on.</small>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------
# Build feature row
# --------------------------------------------------------
def build_feature_row():
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

    return pd.DataFrame([row])


# --------------------------------------------------------
# Predict button + output
# --------------------------------------------------------
st.markdown("### 🔮 Run prediction")

left, right = st.columns([1, 2])

with left:
    run = st.button("Predict purchases", type="primary")

if run:
    X_new = build_feature_row()

    with st.spinner("Calling Databricks model endpoint..."):
        try:
            pred = call_databricks_endpoint(X_new)

            with left:
                st.metric("Estimated purchases", f"{pred:.1f}")

            with right:
                st.markdown("##### Features sent to the model")
                st.dataframe(X_new, use_container_width=True)

        except Exception as e:
            st.error(f"Error calling Databricks endpoint: {e}")
else:
    st.info("Adjust the campaign settings, then click **Predict purchases** to see the model’s estimate.")

