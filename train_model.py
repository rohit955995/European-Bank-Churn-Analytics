import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 1. LOAD DATASET
# =========================

df = pd.read_csv("data/European_Bank.csv")

print("Dataset loaded:", df.shape)


# =========================
# 2. FEATURES & TARGET
# =========================

features = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

X = df[features]
y = df["Exited"]


# =========================
# 3. CATEGORICAL COLUMNS
# =========================

categorical_features = [
    "Geography",
    "Gender"
]

numerical_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]


# =========================
# 4. PREPROCESSING
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)


# =========================
# 5. MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# =========================
# 6. PIPELINE
# =========================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# =========================
# 7. TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================
# 8. TRAIN
# =========================

print("Training model...")

pipeline.fit(X_train, y_train)


# =========================
# 9. EVALUATION
# =========================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL ACCURACY")
print("==============================")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print("==============================")
print(classification_report(y_test, y_pred))


# =========================
# 10. SAVE MODEL
# =========================

joblib.dump(pipeline, "model.joblib")

print("\n==============================")
print("MODEL SAVED SUCCESSFULLY")
print("==============================")
print("File: model.joblib")