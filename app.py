import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Healthcare Cost Predictor", page_icon="🏥", layout="centered")

# -----------------------------------------------------------------
# Load the trained pipeline (preprocessing + model bundled together)
# -----------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("healthcare_cost_model.joblib")

model = load_model()

st.title("🏥 Healthcare Cost Predictor")
st.write(
    "Estimate annual insurance charges based on personal and lifestyle "
    "factors, using a Gradient Boosting model trained on historical data."
)

st.divider()

# -----------------------------------------------------------------
# Input form
# -----------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.5, step=0.1)
    children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

st.divider()

if st.button("Predict Cost", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region,
    }])

    prediction = model.predict(input_df)[0]

    st.metric("Predicted Annual Healthcare Cost", f"${prediction:,.2f}")

    if smoker == "yes":
        st.warning(
            "Smoker status has the largest impact on predicted cost — "
            "it typically accounts for the majority of the difference in charges."
        )

    with st.expander("See input used for this prediction"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption(
    "Model: Gradient Boosting Regressor · Trained on 1,338 records "
    "(age, sex, BMI, children, smoker status, region → charges)."
)
