import streamlit as st
import matplotlib.pyplot as plt
from predict import predict_diabetes_with_explanation

#page config
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("Diabetes Prediction App")
st.markdown("### Enter patient details to predict diabetes risk with SHAP explanation")

# Input form (arranged in two columns)
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=6, help="Number of times pregnant")
    glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, value=148.0, help="Plasma glucose concentration")
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, value=72.0)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, value=35.0)

with col2:
    insulin = st.number_input("Insulin (μU/ml)", min_value=0.0, value=0.0)
    bmi = st.number_input("BMI", min_value=0.0, value=33.6, help="Body Mass Index")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.627, step=0.001)
    age = st.number_input("Age", min_value=1, max_value=120, value=50)

# Predict button
if st.button(" Predict Diabetes Risk", type="primary"):
    input_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    with st.spinner("Analyzing..."):
        result = predict_diabetes_with_explanation(input_dict)

    # Showing results
    st.success("Prediction Complete!")

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric("Prediction", result["message"])
    with col_res2:
        st.metric("Probability of Diabetes", f"{result['probability']:.2%}")

    #Top Contributors
    st.subheader(" Top 5 Contributing Features (SHAP Values)")
    for i, contrib in enumerate(result["top_contributors"], 1):
        impact = "↑ Increases risk" if contrib['SHAP_Value'] > 0 else "↓ Decreases risk"
        st.write(f"**{i}. {contrib['Feature']}**: {contrib['SHAP_Value']:.4f} {impact}")

    # SHAP Waterfall Plot
    st.subheader(" SHAP Waterfall Explanation")
    st.pyplot(result["shap_fig"])

    # Clearing figures (to save memory)
    plt.close(result["shap_fig"])

# footer
st.markdown("---")
st.caption("Built with Streamlit | Model: XGBoost + LightGBM Ensemble with SHAP Explainability")