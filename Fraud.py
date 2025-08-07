# ====== 1. Import Libraries ======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, precision_recall_curve, 
                            average_precision_score, accuracy_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ====== 2. Load & Preprocess Data ======
# Load dataset (place 'creditcard.csv' in the same folder)
data = pd.read_csv(r'C:\Users\DELL\Downloads\Liveprojects\creditcard.csv')

# Check class imbalance
print("\nClass Distribution (0: Legit, 1: Fraud):")
print(data['Class'].value_counts())

# Scale 'Time' and 'Amount'
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data['Time'] = StandardScaler().fit_transform(data['Time'].values.reshape(-1, 1))

# Split features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Split into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====== 3. Handle Class Imbalance (SMOTE) ======
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Balance fraud to 50% of legit
X_res, y_res = smote.fit_resample(X_train, y_train)

print("\nResampled Class Distribution:")
print(pd.Series(y_res).value_counts())

# ====== 4. Train Models ======
# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_res, y_res)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_res, y_res)

# ====== 5. Evaluate Models ======
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\n=== {model_name} Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nROC AUC Score:", roc_auc_score(y_test, y_prob))
    
    # Plot Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (AP={average_precision_score(y_test, y_prob):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

# Evaluate both models
evaluate_model(lr, X_test, y_test, "Logistic Regression")
evaluate_model(rf, X_test, y_test, "Random Forest")

# ====== 6. Feature Importance (Random Forest) ======
importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Random Forest - Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.tight_layout()
plt.show()

# ====== 7. Save Best Model ======
import joblib
joblib.dump(rf, 'fraud_detection_model.pkl')
print("\nModel saved as 'fraud_detection_model.pkl'")

# ====== 8. Test Prediction Example ======
sample_transaction = X_test.iloc[0:1]  # Take 1 test transaction
prediction = rf.predict(sample_transaction)
prob_fraud = rf.predict_proba(sample_transaction)[0][1]

print("\nSample Transaction Prediction:")
print(f"Predicted Class: {prediction[0]} (0=Legit, 1=Fraud)")
print(f"Fraud Probability: {prob_fraud:.4f}")