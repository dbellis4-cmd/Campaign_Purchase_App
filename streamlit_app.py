import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta

# --------------------------------------------------------
# Page config
# --------------------------------------------------------
st.set_page_config(
    page_title="Meta Ad Campaign Purchase Predictor",
    page_icon="💸",
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

    payload = {"dataframe_split": df.to_dict(orient="split")}

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
# Global CSS – black background, neon-purple inputs, slider colors,
# hide sidebar, purple button + output card
# --------------------------------------------------------
st.markdown(
    """
    <style>
    /* ----- Global theme overrides (kills the orange/red) ----- */
    :root {
        --primary-color: #8A00C4;
        --primary-color-hover: #8A00C4;
        --secondary-background-color: #000000;
    }

    .stApp {
        background: #000000;
        color: #f5f5f5;
    }

    /* Hide sidebar entirely */
    div[data-testid="stSidebar"] { display: none !important; }

    /* Center main content */
    main .block-container {
        max-width: 1100px;
        margin: 0 auto;
    }

    /* Number inputs container (purple box) */
    div[data-testid="stNumberInput"] > div {
        background-color: #8A00C4 !important;
        border-radius: 0.5rem;
    }
    div[data-testid="stNumberInput"] input {
        background-color: #8A00C4 !important;
        color: #ffffff !important;
    }
    /* +/- buttons on number inputs */
    div[data-testid="stNumberInput"] button {
        background-color: #8A00C4 !important;
        color: #ffffff !important;
        border: none !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        background-color: #A536FF !important;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background-color: #8A00C4 !important;
        color: #ffffff !important;
        border-radius: 0.5rem;
    }

    /* Generic text inputs */
    div[data-baseweb="input"] input {
        background-color: #8A00C4 !important;
        color: #ffffff !important;
    }

    /* ----- Slider: kill the orange ----- */

/* Remove default colored rail */
div[data-baseweb="slider"] > div {
    background-color: transparent !important;
}

/* First segment (left / filled part) */
div[data-baseweb="slider"] > div > div:nth-child(1) {
    background-color: #8A00C4 !important;  /* purple filled bar */
}

/* Second segment (right / empty part) */
div[data-baseweb="slider"] > div > div:nth-child(2) {
    background-color: #FFFFFF !important;  /* white remaining bar */
}

/* Number above the handle */
div[data-baseweb="slider"] span {
    color: #ffffff !important;            /* purple value text */
}


    /* Metric background (output card) */
    .stMetric {
        background-color: #8A00C4 !important;
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
    }

    /* Spacer before Notes – default (desktop) */
.notes-spacer {
    height: 60px;
}

/* Mobile tweaks */
@media (max-width: 768px) {
    .notes-spacer {
        height: 30px;  /* shorter gap on mobile */
    }
}


   
    /* ... your existing styles ... */

    /* Customize info / alert boxes (st.info, st.warning, st.error) */
    div[data-testid="stAlert"] {
        background-color: #120024 !important;  /* dark purple background */
        border-radius: 0.75rem !important;
        border: 1px solid #8A00C4 !important;  /* neon purple border */
        color: #ffffff !important;
    }

    /* Make the text inside slightly brighter */
    div[data-testid="stAlert"] p {
        color: #ffffff !important;
    }
    


    /* Primary button: purple pill */
    div.stButton > button {
        background-color: #8A00C4 !important;
        color: #ffffff !important;
        border-radius: 999px !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
    }
    div.stButton > button:hover {
        background-color: #A536FF !important;
    }

    /* Main title – default (desktop) */
.main-title {
    font-size: 2.6rem;
    margin-bottom: 0.4rem;
}

/* Mobile tweaks */
@media (max-width: 768px) {
    .main-title {
        font-size: 1.8rem !important;  /* smaller on mobile */
        line-height: 1.2;
    }
}

    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------
# Header
# --------------------------------------------------------
st.markdown(
    "<h1 class='main-title'>📊 Meta Ad Campaign Purchase Predictor</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
This app uses a regression model trained on a synthetic social media advertising dataset to estimate how many purchases a campaign might generate. It is meant for learning and scenario testing, **not** for real business forecasting.

**How to use the app:**
1. Set your campaign basics (duration, budget, and number of unique interests).  
2. Choose how many ads you will run on each platform (Facebook and Instagram).  
3. Distribute those ads across formats (carousel, image, stories, video) and audience segments (gender and age groups).  
4. Click **Predict Purchases** to see the model’s estimated number of purchases for that setup.

Predictions are based only on these campaign settings and do not include product details, prices, or real-world factors like competitors or seasonality, so results should be interpreted as approximate and educational.
    """
)


# Spacer between intro and core settings
st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
# You can increase/decrease 40px as you like




# --------------------------------------------------------
# Core campaign settings (fully on main page)
# --------------------------------------------------------
st.subheader("Core Campaign Settings")

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

c1, c2, c3, c4 = st.columns(4)

with c1:
    duration_days = st.number_input(
        "Campaign duration (days)",
        min_value=30,
        max_value=365,
        value=30,
        step=1,
    )

with c2:
    total_budget = st.number_input(
        "Total Budget ($)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        format="%.2f",
    )

with c3:
    num_unique_interests = st.number_input(
        "Number of Unique Interests",
        min_value=1,
        value=3,
        step=1,
    )
    st.caption("Ex: travel, gaming, food, etc.")


with c4:
    start_month_name = st.selectbox("Start Month", MONTH_NAMES, index=0)
    start_month = MONTH_NAMES.index(start_month_name) + 1

    # Compute end month from duration (assume year 2024, starting on day 1)
    start_date = datetime(2024, start_month, 1)
    end_date = start_date + timedelta(days=duration_days)
    end_month = end_date.month
    end_month_name = MONTH_NAMES[end_month - 1]

    st.caption(f"End Month (auto from duration): {end_month_name}")

st.markdown("---")

# --------------------------------------------------------
# Platform + format mix
# --------------------------------------------------------
col_platforms, col_types = st.columns(2)

with col_platforms:
    st.subheader("Ad-Platform Mix")
    st.caption("Number of ads on each platform. Total ads is computed automatically.")
    facebook_ads = st.number_input("Facebook Ads", min_value=0, step=1, value=0)
    instagram_ads = st.number_input("Instagram Ads", min_value=0, step=1, value=0)

    total_ads_computed = facebook_ads + instagram_ads
    st.markdown(f"**Total Number of Ads (computed):** {int(total_ads_computed)}")

with col_types:
    st.subheader("Ad-format mix")
    st.caption("Split your creatives across formats.")

    carousel_ads = st.number_input(
        "Carousel ads", min_value=0, step=1, value=0
    )
    image_ads = st.number_input(
        "Image ads", min_value=0, step=1, value=0
    )
    stories_ads = st.number_input(
        "Stories ads", min_value=0, step=1, value=0
    )
    video_ads = st.number_input(
        "Video ads", min_value=0, step=1, value=0
    )

    # New: total format ads vs total ads
    format_total = carousel_ads + image_ads + stories_ads + video_ads
    total_ads_computed = facebook_ads + instagram_ads

    st.caption(
        f"Total format-specific ads: {format_total} "
        f"(out of {int(total_ads_computed)} total ads)"
    )

    if format_total > total_ads_computed:
        st.warning(
            f"You assigned {format_total} ads across formats but only "
            f"{int(total_ads_computed)} total ads. "
            "Please reduce one of the format counts."
        )



st.markdown("---")

# --------------------------------------------------------
# Targeting mix
# --------------------------------------------------------
col_gender, col_age = st.columns(2)

with col_gender:
    st.subheader("Target Gender Mix")
    st.caption("Number of ads aimed at each gender segment.")
    male_ads = st.number_input("Male-targeted ads", min_value=0, step=1, value=0)
    female_ads = st.number_input("Female-targeted ads", min_value=0, step=1, value=0)
    all_gender_ads = st.number_input("All-gender ads", min_value=0, step=1, value=0)

    gender_total = male_ads + female_ads + all_gender_ads
    total_ads_computed = facebook_ads + instagram_ads

    st.caption(f"Total gender-targeted ads: {gender_total} (out of {int(total_ads_computed)} total ads)")

    if gender_total > total_ads_computed:
        st.warning(
            f"You assigned {gender_total} gender-targeted ads but only "
            f"{int(total_ads_computed)} total ads. Please reduce one of the values."
        )


with col_age:
    st.subheader("Target Age-Group Mix")
    st.caption("Number of ads aimed at each age band.")
    age_18_24_ads = st.number_input("Age 18–24 ads", min_value=0, step=1, value=0)
    age_25_34_ads = st.number_input("Age 25–34 ads", min_value=0, step=1, value=0)
    age_35_44_ads = st.number_input("Age 35–44 ads", min_value=0, step=1, value=0)
    age_all_ads = st.number_input("All-age ads", min_value=0, step=1, value=0)

st.markdown(
    "<small>Any feature not controlled in the UI is set to 0. "
    "The model still uses the full feature vector it was trained on.</small>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------
# Build feature row (num_ads auto from FB + IG)
# --------------------------------------------------------
def build_feature_row():
    row = {c: 0 for c in FEATURE_COLS}

    # Core numeric features
    if "duration_days" in row:
        row["duration_days"] = duration_days
    if "total_budget" in row:
        row["total_budget"] = total_budget
    if "num_unique_interests" in row:
        row["num_unique_interests"] = num_unique_interests

    # Auto-computed number of ads
    num_ads_calc = facebook_ads + instagram_ads
    if "num_ads" in row:
        row["num_ads"] = num_ads_calc

    # Months (numeric)
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
# --------------------------------------------------------
# Predict button + output
# --------------------------------------------------------
st.markdown("### 🔮 Run Prediction")

left, right = st.columns([1, 2])

with left:
    run = st.button("Predict Purchases", type="primary")

if run:
    # Short-circuit: if there are 0 total ads, do NOT call the model
    total_ads = facebook_ads + instagram_ads

    if total_ads == 0:
        with left:
            # show whole number 0
            st.metric("Estimated Purchases", "0")
        with right:
            st.info(
                "Total number of ads is **0**, so the model was not called. "
                "Increase the number of Facebook or Instagram ads to get a prediction."
            )
    else:
        X_new = build_feature_row()
        with st.spinner("Calling Databricks model endpoint..."):
            try:
                pred = call_databricks_endpoint(X_new)
                pred_rounded = int(round(pred))  # round to nearest whole number

                with left:
                    # display as whole number (no decimals)
                    st.metric("Estimated Purchases", f"{pred_rounded}")

                with right:
                    st.markdown("##### Features sent to the model")
                    st.dataframe(X_new, use_container_width=True)

            except Exception as e:
                st.error(f"Error calling Databricks endpoint: {e}")
else:
    st.info("Adjust the campaign settings above, then click **Predict Purchases** to see the model’s estimate.")

# Spacer between Predict section and Notes
st.markdown("<div class='notes-spacer'></div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div class='notes-spacer'></div>", unsafe_allow_html=True)
st.markdown("### ℹ️ Important notes and limitations")


st.markdown(
    """

- **Synthetic, single-company data**  
  This model is trained on a **synthetic dataset** for one fictional company. Real businesses and platforms may behave very differently, so predictions are **rough scenario estimates**, not exact forecasts.  
  Dataset used: https://www.kaggle.com/datasets/alperenmyung/social-media-advertisement-performance  

- **Small sample + synthetic augmentation**  
  The original data contained **only 50 campaigns**. Extra synthetic campaigns were created by slightly changing those original campaigns. This helps the model train but does **not** make the data truly representative of the real world.

- **Limited inputs**  
  The model only sees high-level campaign settings (duration, budget, number of ads, number of unique interests, platform mix, ad formats, and basic gender/age targeting).  
  It does **not** know the product, price, brand strength, creative quality, competition, or platform changes.

- **Duration is a weak driver here**  
  In this dataset, campaign duration had **low predictive power** after controlling for other features. Changing duration alone may have **little or no impact** on predicted purchases, even though in real life duration would usually matter more.

- **Correlation, not causation; no external factors**  
  The model learns patterns in past data, not cause-and-effect. It does not account for seasonality, holidays, competitors, or the wider economy. Actual performance may be **higher or lower** than the prediction.

- **Use for similar campaigns and comparisons**  
  Predictions are most reasonable for campaigns similar to those in the dataset. Very unusual setups (extreme budgets, very long durations, uncommon mixes) may be unreliable.  
  Treat the output as a **single point estimate** and mainly use it to compare “Scenario A vs Scenario B” within this synthetic world.

- **Behavior when total ads = 0**  
  If the **total number of ads is 0**, the app returns **0 predicted purchases** and does not call the model.

- **Educational tool only**  
  This app is meant for **learning and experimentation**, not for real budget decisions. Real decisions should also rely on expert judgment, tests (A/B experiments), and real performance data.

"""
)

