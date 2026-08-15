import pandas as pd
import os
import sys
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# IMPORT PREPROCESSING


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_and_preprocess_data



# LOAD PREPROCESSED DATA


df = load_and_preprocess_data()

print("\n========== DATA LOADED ==========")
print("Shape:", df.shape)



# REMOVE UNNECESSARY COLUMNS


drop_columns = [
    "Exited",
    "CustomerId",
    "Surname",
    "Year"
]

X = df.drop(columns=drop_columns, errors="ignore")
y = df["Exited"]



# IDENTIFY FEATURES


categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\n========== FEATURES ==========")
print("Categorical:", categorical_features)
print("Numeric:", numeric_features)



# PREPROCESSING PIPELINE


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)



# RANDOM FOREST MODEL


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced"
)



# COMPLETE PIPELINE


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)



# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n========== TRAINING MODEL ==========")

pipeline.fit(
    X_train,
    y_train
)



# PREDICTION


y_pred = pipeline.predict(X_test)



# MODEL EVALUATION


accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========== MODEL RESULTS ==========")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)



# SAVE MODEL


model_path = "model.joblib"

joblib.dump(
    pipeline,
    model_path
)

print("\n========== MODEL SAVED ==========")
print(
    f"Model saved successfully: {model_path}"
)
