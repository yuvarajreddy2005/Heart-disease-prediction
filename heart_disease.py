import pandas as pd
import numpy as np
import joblib
import speech_recognition as sr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)

# === Step 1: Load Dataset ===
data = pd.read_csv("dataset.csv")
print("📄 Columns in dataset:", list(data.columns))

# === Step 2: Preprocessing ===
X = data.drop("target", axis=1)
y = data["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 3: Train Model ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# === Step 4: Evaluation ===
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy}")

# === Step 5: Voice + Manual Input Function ===
def get_number_from_voice(prompt, allow_float=False):
    r = sr.Recognizer()
    for _ in range(2):
        with sr.Microphone() as source:
            print(f"{prompt} 🎙️")
            audio = r.listen(source)
        try:
            response = r.recognize_google(audio)
            print(f"🗣️ You said: {response}")
            if allow_float:
                return float(response)
            return int(response)
        except:
            print("❌ Could not understand")
    return None

def get_string_from_voice(prompt, valid_options):
    r = sr.Recognizer()
    for _ in range(2):
        with sr.Microphone() as source:
            print(f"{prompt} 🎙️")
            audio = r.listen(source)
        try:
            response = r.recognize_google(audio).lower()
            print(f"🗣️ You said: {response}")
            for option in valid_options:
                if option in response:
                    return option
        except:
            print("❌ Could not understand")
    return None

# === Step 6: Collect User Input ===
def collect_user_data():
    try:
        age = get_number_from_voice("Say your age")
        if age is None: raise Exception

        sex = get_string_from_voice("Say your gender (male or female)", ["male", "female"])
        sex = 1 if sex == "male" else 0

        cp = get_string_from_voice(
            "Say chest pain type: typical, atypical, non-anginal, or asymptomatic",
            ["typical", "atypical", "non-anginal", "asymptomatic"]
        )
        cp_map = {"typical": 0, "atypical": 1, "non-anginal": 2, "asymptomatic": 3}
        cp = cp_map.get(cp)

        trestbps = get_number_from_voice("Say resting blood pressure")
        chol = get_number_from_voice("Say cholesterol level")

        # ⌨️ Forced Manual Input for clarity
        fbs = int(input("⌨️ Type 1 if fasting blood sugar > 120, else 0: "))
        restecg = int(input("⌨️ Type ECG result (0 = normal, 1 = abnormal, 2 = LVH): "))

        thalach = get_number_from_voice("Say your max heart rate")
        exang = int(input("⌨️ Type 1 if exercise angina present, else 0: "))

        oldpeak = get_number_from_voice("Say ST depression like one point four", allow_float=True)

        slope = get_string_from_voice(
            "Say ST slope: upward, flat, or downward",
            ["upward", "flat", "downward"]
        )
        slope_map = {"upward": 0, "flat": 1, "downward": 2}
        slope = slope_map.get(slope)

        if None in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]:
            raise Exception("❌ One or more inputs were invalid.")

        user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope]])
        return user_input
    except:
        print("❌ Failed to collect user input or make prediction.")
        return None

# === Step 7: Predict on User Input ===
user_data = collect_user_data()
if user_data is not None:
    scaled_input = scaler.transform(user_data)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1] * 100

    print("\n🎯 Prediction Result:", "❗ Heart Disease Detected ❗" if prediction == 1 else "✅ No Heart Disease")
    print(f"📊 Risk Score: {round(probability, 2)} / 100")

    if probability >= 70:
        print("🚦 Risk Level: High Risk 🔴")
    elif probability >= 40:
        print("🚦 Risk Level: Medium Risk 🟠")
    else:
        print("🚦 Risk Level: Low Risk 🟢")

# === Step 8: Accuracy Visualization ===

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 5))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
