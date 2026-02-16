# train.py

import os
import joblib
import pandas as pd

from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# =====================
# Carga
# =====================
data = load_penguins()

# =====================
# Limpieza
# =====================
num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
for col in num_cols:
    data[col].fillna(data[col].median(), inplace=True)

data['sex'].fillna(data['sex'].mode()[0], inplace=True)

# =====================
# Encoding target
# =====================
le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])

# =====================
# One-hot
# =====================
data = pd.get_dummies(data, columns=['island', 'sex'], drop_first=True)

# =====================
# Feature engineering
# =====================
data['bill_ratio'] = data['bill_length_mm'] / data['bill_depth_mm']
data['mass_per_flipper_length'] = data['body_mass_g'] / data['flipper_length_mm']

# =====================
# Split
# =====================
X = data.drop('species', axis=1)
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================
# SMOTE
# =====================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# =====================
# Modelos
# =====================
rf = RandomForestClassifier(random_state=42)
rf.fit(X_resampled, y_resampled)

lr = LogisticRegression(max_iter=3000)
lr.fit(X_resampled, y_resampled)

# =====================
# Guardar
# =====================
models_dir = r"C:\Users\MSI\taller_penguins\models"
os.makedirs(models_dir, exist_ok=True)

joblib.dump(rf, os.path.join(models_dir, "rf_model.joblib"))
joblib.dump(lr, os.path.join(models_dir, "lr_model.joblib"))
joblib.dump(le, os.path.join(models_dir, "label_encoder.joblib"))

print("Modelos y encoder guardados en /models")
