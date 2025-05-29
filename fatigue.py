import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('hypothyroid.csv')
df.replace('?', np.nan, inplace=True)

hormones = ['TSH', 'T3', 'TT4', 'FTI']

for col in hormones:
    df[col] = pd.to_numeric(df[col], errors='coerce')

normal_ranges = {
    'TSH': (0.4, 4.0),
    'T3': (80, 180),
    'TT4': (5.0, 12.0),
    'FTI': (4.0, 11.0)
}

def calculate_fatigue_risk(row):
    abnormal_count = 0
    for hormone in hormones:
        low, high = normal_ranges[hormone]
        val = row[hormone]
        if pd.isna(val):
            continue  
        if val < low or val > high:
            abnormal_count += 1
    risk_pct = (abnormal_count / len(hormones)) * 100
    return risk_pct

df['fatigue_risk_percent'] = df.apply(calculate_fatigue_risk, axis=1)

def risk_category(risk):
    if risk == 0:
        return 'Low'
    elif risk <= 50:
        return 'Moderate'
    else:
        return 'High'

df['fatigue_risk_category'] = df['fatigue_risk_percent'].apply(risk_category)

df_model = df.dropna(subset=hormones + ['fatigue_risk_category']).copy()

X = df_model[hormones]

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

le = LabelEncoder()
y = le.fit_transform(df_model['fatigue_risk_category'])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
import joblib

joblib.dump(rf, 'fatigue_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model and preprocessors saved.")
