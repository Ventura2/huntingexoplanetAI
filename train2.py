import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tabpfn import TabPFNClassifier
import joblib

# Load the dataset
data = pd.read_csv("cumulative_2025.10.04_02.30.22.csv")
print(data.head())

# Keep only numeric and categorical relevant features
features = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
    'koi_srad', 'koi_smass', 'koi_impact', 'koi_teq',
    'koi_insol', 'koi_model_snr', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co'
]

X = data[features]
y = data['koi_disposition']


# Encode target labels numerically
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into train test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Handle NaNs (TabPFN can't handle NaNs)
X = X.fillna(X.mean())

#clf = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)  # or 'cuda' if you have GPU
clf = TabPFNClassifier(device='cuda')  # or 'cuda' if you have GPU

clf.fit(X_train, y_train)

#Predict 
y_pred = clf.predict(X_test)
print('Test Accuracy:', accuracy_score(y_test, y_pred))

#Show result
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(confusion_matrix(y_test, y_pred))


# by probability

probs = clf.predict_proba(X_test)
data_eval = X_test.copy()
data_eval['true_label'] = le.inverse_transform(y_test)
data_eval['predicted_label'] = le.inverse_transform(y_pred)
data_eval['prob_planet'] = probs[:, list(le.classes_).index("CONFIRMED")]
print(data_eval.head())


# Guardar
joblib.dump(clf, "tabpfn_model.pkl")

# Cargar después
#clf_cargado = joblib.load("tabpfn_model.pkl")