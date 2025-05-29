import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Lasso
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

df = pd.read_csv("hypothyroid.csv")
df.replace('?', np.nan, inplace=True)
df = df.infer_objects()

if df['TBG'].isnull().all():
    df.drop(columns=['TBG'], inplace=True)

df['binaryClass'] = df['binaryClass'].map({'P': 0, 'N': 1})  # 0: Hypothyroid, 1: Hyperthyroid

def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    
    return ColumnTransformer([
        ('cat', categorical_pipeline, categorical_cols),
        ('num', numerical_pipeline, numerical_cols)
    ])

X = df.drop("binaryClass", axis=1)
y = df["binaryClass"]

preprocessor = build_preprocessor(X)

smote_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

smote_pipeline.fit(X_train, y_train)
y_pred = smote_pipeline.predict(X_test)

print("\n[SMOTE-Balanced Classifier Results]")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, target_names=["Hypothyroid", "Hyperthyroid"]
))

joblib.dump(smote_pipeline, 'thyroid_classifier.pkl')
print("\nClassifier model saved as 'thyroid_classifier.pkl'")

clinical_thresholds = {
    'TSH': {'low': 0.4, 'high': 4.0},
    'T3': {'low': 80, 'high': 180},
    'TT4': {'low': 5.0, 'high': 12.0},
    'FTI': {'low': 4.0, 'high': 11.0}
}

def classify_thyroid_clinically(row):
    hypo_score = hyper_score = 0
    for hormone, limits in clinical_thresholds.items():
        val = row.get(hormone)
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

df_reg = df.dropna(subset=['TSH'])
X_reg = df_reg.drop(columns=["binaryClass", "TSH"])
y_reg = df_reg["TSH"]

preprocessor_reg = build_preprocessor(X_reg)

lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor_reg),
    ('regressor', Lasso(alpha=0.01))
])

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lasso_pipeline.fit(X_reg_train, y_reg_train)
predicted_TSH = np.clip(lasso_pipeline.predict(X_reg_test), 0, 20)


print("\n[TSH Prediction - Lasso Regression]")
print("MAE:", mean_absolute_error(y_reg_test, predicted_TSH))
rmse = np.sqrt(mean_squared_error(y_reg_test, predicted_TSH))
print("RMSE:", rmse)


clinical_df = X_reg_test.copy()
clinical_df['TSH_pred'] = predicted_TSH

def classify_from_predicted_tsh(val):
    if val < clinical_thresholds['TSH']['low']:
        return "Hyperthyroidism"
    elif val > clinical_thresholds['TSH']['high']:
        return "Hypothyroidism"
    return "Indeterminate"

clinical_df['ClinicalDiagnosis'] = clinical_df['TSH_pred'].apply(classify_from_predicted_tsh)

print("\nDiagnosis counts from predicted TSH:")
print(clinical_df['ClinicalDiagnosis'].value_counts())

print("\nSample Clinical Diagnoses from Predicted TSH:")
print(clinical_df[['TSH_pred', 'ClinicalDiagnosis']].head(100))

# Save regressor
joblib.dump(lasso_pipeline, 'tsh_regressor.pkl')
print("\nTSH prediction model saved as 'tsh_regressor.pkl'")
