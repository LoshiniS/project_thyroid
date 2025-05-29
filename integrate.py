

import pandas as pd
import numpy as np
import joblib

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

def predict_fatigue_and_link(input_df):
    # Select hormone columns
    hormones = ['TSH', 'T3', 'TT4', 'FTI']
    fatigue_input = input_df[hormones]

    # Fatigue risk prediction
    fatigue_input_imputed = imputer.transform(fatigue_input)
    fatigue_input_scaled = scaler.transform(fatigue_input_imputed)
    fatigue_pred = fatigue_model.predict(fatigue_input_scaled)[0]
    fatigue_class = label_encoder.inverse_transform([fatigue_pred])[0]

    
    thyroid_type = classify_thyroid_clinically(input_df.iloc[0])

    
    if fatigue_class in ["Moderate", "High"] and thyroid_type != "Indeterminate":
        combined = f"Fatigue likely due to {thyroid_type}"
    elif fatigue_class in ["Moderate", "High"]:
        combined = "Fatigue present but thyroid type is unclear"
    else:
        combined = "Fatigue not significantly linked to thyroid levels"

    return {
        "thyroid_type": thyroid_type,
        "fatigue_risk": fatigue_class,
        "combined_diagnosis": combined
    }

# Example usage
if __name__ == "__main__":
    
    test_sample = pd.DataFrame([{
        'TSH': 0.5,     # High TSH
        'T3': 300,       # Low T3
        'TT4': 15,     # Low TT4
        'FTI': 3.7      # Low FTI
    }])

    result = predict_fatigue_and_link(test_sample)
    print(result)
