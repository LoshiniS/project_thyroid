import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('hypothyroid.csv')

df.drop(columns=[col for col in df.columns if 'measured' in col], inplace=True)

df.replace('?', np.nan, inplace=True)

bool_cols = [
    'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant',
    'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych'
]

for col in bool_cols:
    df[col] = df[col].map({'t': 1, 'f': 0})


for col in ['sex', 'referral source']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

df['binaryClass'] = df['binaryClass'].map({'P': 1, 'N': 0})

num_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
X = df.drop(columns=['binaryClass'])
y = df['binaryClass']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(" Preprocessing Complete")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

