import pandas as pd
import numpy as np
import joblib
import shap

fatigue_model = joblib.load("fatigue_rf_model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")


clinical_ranges = {
    'TSH': {'low': 0.4, 'high': 4.0},
    'T3': {'low': 80, 'high': 180},
    'TT4': {'low': 5.0, 'high': 12.0},
    'FTI': {'low': 4.0, 'high': 11.0}
}

def classify_thyroid_clinically(row):
    hypo_score, hyper_score = 0, 0
    for hormone, limits in clinical_ranges.items():
        val = row[hormone]
        if pd.isna(val): 
            continue
        if val < limits['low']:
            hypo_score += 1
        elif val > limits['high']:
            hyper_score += 1
    if hypo_score > hyper_score:
        return "Hypothyroidism"
    elif hyper_score > hypo_score:
        return "Hyperthyroidism"
    else:
        return "Indeterminate"

def explain_fatigue_prediction_with_thyroid(model, input_scaled, feature_names, thyroid_type):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

   
    if isinstance(shap_values, list):
        shap_val = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        shap_val = shap_values[0]

    explanation_parts = []
    for i, feature in enumerate(feature_names):
        raw_value = shap_val[i]
        
        contribution = float(raw_value) if not isinstance(raw_value, np.ndarray) else float(raw_value.ravel()[0])
        impact = "increases" if contribution > 0 else "decreases"
        explanation_parts.append(f"{feature} ({input_scaled[0, i]:.2f}) {impact} fatigue risk by {abs(contribution):.3f}")

    
    explanation_text = (
        f"The fatigue prediction model indicates that your fatigue risk is because of '{thyroid_type}'. "
        f"This thyroid condition plays a major role in your fatigue risk.\n"
        "Regarding hormone contributions: "
        + ", ".join(explanation_parts)
        + "."
    )
    return explanation_text

def predict_fatigue_and_link_with_explanation(user_inputs):
    hormones = ['TSH', 'T3', 'TT4', 'FTI']
    input_df = pd.DataFrame([user_inputs])

    fatigue_input = input_df[hormones]
    fatigue_imputed = imputer.transform(fatigue_input)
    fatigue_scaled = scaler.transform(fatigue_imputed)
    fatigue_pred = fatigue_model.predict(fatigue_scaled)[0]
    fatigue_class = label_encoder.inverse_transform([fatigue_pred])[0]

    thyroid_type = classify_thyroid_clinically(input_df.iloc[0])
    explanation = explain_fatigue_prediction_with_thyroid(fatigue_model, fatigue_scaled, hormones, thyroid_type)

    if fatigue_class in ["Moderate", "High"] and thyroid_type != "Indeterminate":
        combined = f"Fatigue likely due to {thyroid_type}"
    elif fatigue_class in ["Moderate", "High"]:
        combined = "Fatigue present but thyroid type is unclear"
    else:
        combined = "Fatigue not significantly linked to thyroid levels"

    return {
        "thyroid_type": thyroid_type,
        "fatigue_risk": fatigue_class,
        "combined_diagnosis": combined,
        "explanation": explanation
    }

if __name__ == "__main__":
    print("🔬 Thyroid Fatigue Predictor\n")

    try:
        tsh = float(input("Enter TSH value: "))
        t3 = float(input("Enter T3 value: "))
        tt4 = float(input("Enter TT4 value: "))
        fti = float(input("Enter FTI value: "))

        user_data = {'TSH': tsh, 'T3': t3, 'TT4': tt4, 'FTI': fti}
        result = predict_fatigue_and_link_with_explanation(user_data)

        print("\n✅ Prediction Results:")
        print(f"🧠 Thyroid Type: {result['thyroid_type']}")
        print(f"🔋 Fatigue Risk: {result['fatigue_risk']}")
        print(f"📌 Combined Diagnosis: {result['combined_diagnosis']}")
        print(f"\n💡 Explanation:\n{result['explanation']}")

    except Exception as e:
        print(f"\n⚠️ Error processing input: {e}")
