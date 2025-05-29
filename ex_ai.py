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
    hypo_score = 0
    hyper_score = 0
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

def explain_fatigue_prediction(fatigue_model, input_data_scaled, feature_names):
    explainer = shap.TreeExplainer(fatigue_model)
    shap_values = explainer.shap_values(input_data_scaled)

    if isinstance(shap_values, list):
        shap_val_instance = shap_values[1][0]
    else:
        shap_val_instance = shap_values[0]

    explanation_lines = []
    for i, feature in enumerate(feature_names):
        contribution = shap_val_instance[i]
        # Ensure contribution is scalar
        if isinstance(contribution, (np.ndarray, list)):
            contribution = float(np.array(contribution).flatten()[0])
        impact = "increases" if contribution > 0 else "decreases"
        explanation_lines.append(f"{feature} ({input_data_scaled[0][i]:.2f}) {impact} fatigue risk by {abs(contribution):.3f}")

    explanation_text = " | ".join(explanation_lines)
    return explanation_text

def predict_fatigue_and_link_with_explanation(input_df):
    # Select hormone columns
    hormones = ['TSH', 'T3', 'TT4', 'FTI']
    fatigue_input = input_df[hormones]

    # Fatigue risk prediction
    fatigue_input_imputed = imputer.transform(fatigue_input)
    fatigue_input_scaled = scaler.transform(fatigue_input_imputed)
    fatigue_pred = fatigue_model.predict(fatigue_input_scaled)[0]
    fatigue_class = label_encoder.inverse_transform([fatigue_pred])[0]

   
    thyroid_type = classify_thyroid_clinically(input_df.iloc[0])

    explanation = explain_fatigue_prediction(fatigue_model, fatigue_input_scaled, hormones)

   
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
    test_sample = pd.DataFrame([{
        'TSH': 0.5,
        'T3': 300,
        'TT4': 15,
        'FTI': 3.7
    }])

    result = predict_fatigue_and_link_with_explanation(test_sample)
    print(result)
