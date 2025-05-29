import streamlit as st
import pandas as pd
from test3 import predict_fatigue_and_link_with_explanation  # or from ex_ai import ...

st.set_page_config(page_title="Thyroid Fatigue Predictor", layout="centered")

st.title("🔬 Thyroid & Fatigue Risk Predictor")
st.markdown("Enter your hormone test values to assess fatigue risk and potential thyroid condition.")

# User input form
with st.form("input_form"):
    tsh = st.number_input("TSH (mIU/L)", min_value=0.0, step=0.1)
    t3 = st.number_input("T3 (ng/dL)", min_value=0.0, step=0.1)
    tt4 = st.number_input("TT4 (μg/dL)", min_value=0.0, step=0.1)
    fti = st.number_input("FTI", min_value=0.0, step=0.1)
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        user_input = {"TSH": tsh, "T3": t3, "TT4": tt4, "FTI": fti}
        result = predict_fatigue_and_link_with_explanation(user_input)

        st.success("✅ Prediction Complete!")
        st.write("**🧠 Thyroid Type:**", result["thyroid_type"])
        st.write("**🔋 Fatigue Risk:**", result["fatigue_risk"])
        st.write("**📌 Combined Diagnosis:**", result["combined_diagnosis"])
        st.markdown("**💡 Explanation:**")
        st.info(result["explanation"])

    except Exception as e:
        st.error(f"Error in processing: {e}")
