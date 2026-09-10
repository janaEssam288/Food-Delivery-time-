# ==============================================================================
# Food Delivery Time Prediction — full ML pipeline
# ==============================================================================

import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

# ------------------------------------------------------------------------------
# 1. Extraction & Loading Data
# ------------------------------------------------------------------------------
zip_path = "archive (1).zip"
extract_dir = "archive_extracted"

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

DATA_DIR = None
if os.path.exists(extract_dir):
    for root, dirs, files in os.walk(extract_dir):
        if "train.csv" in files:
            DATA_DIR = root
            break

if DATA_DIR is None:
    DATA_DIR = "."  # local fallback

print(f"Loading data from: {DATA_DIR}")
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

# ------------------------------------------------------------------------------
# 2. Data Cleaning Function
# ------------------------------------------------------------------------------
def clean_dataframe(df):
    """Strip whitespace, turn 'NaN' text into real missing values,
    and remove stray text prefixes/suffixes."""
    df = df.copy()

    # 1) Strip whitespace & fix 'NaN' strings
    text_cols = df.select_dtypes(include="object").columns
    for c in text_cols:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"NaN": np.nan, "nan": np.nan})

    # 2) Text prefix cleanups
    if "Weatherconditions" in df.columns:
        df["Weatherconditions"] = df["Weatherconditions"].str.replace("conditions ", "", regex=False)
    if "Time_taken(min)" in df.columns:
        df["Time_taken(min)"] = (
            df["Time_taken(min)"].str.replace("(min) ", "", regex=False).astype(float)
        )

    # 3) Convert text-stored numbers
    for c in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

train_c = clean_dataframe(train)
test_c = clean_dataframe(test)

# ------------------------------------------------------------------------------
# 3. Evaluation Helper Function
# ------------------------------------------------------------------------------
def evaluate(model, X, y):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 3)}

# ------------------------------------------------------------------------------
# 4. Preparing Features and Target
# ------------------------------------------------------------------------------
# Drop missing targets if any
train_c = train_c.dropna(subset=["Time_taken(min)"])

X = train_c.drop(columns=["ID", "Delivery_person_ID", "Time_taken(min)"], errors="ignore")
y = train_c["Time_taken(min)"]

# Identify column types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# ------------------------------------------------------------------------------
# 5. Preprocessing Pipelines & Train/Test Split
# ------------------------------------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------------------------
# 6. Model Training & Ablation Test (City Feature)
# ------------------------------------------------------------------------------
# Model A: With City
pipe_with_city = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])
pipe_with_city.fit(X_train, y_train)
score_with_city = evaluate(pipe_with_city, X_test, y_test)

# Model B: Without City
cat_cols_no_city = [c for c in cat_cols if c != "City"]
preprocessor_no_city = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols_no_city),
])

X_train_no_city = X_train.drop(columns=["City"], errors="ignore")
X_test_no_city = X_test.drop(columns=["City"], errors="ignore")

pipe_no_city = Pipeline([
    ("prep", preprocessor_no_city),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])
pipe_no_city.fit(X_train_no_city, y_train)
score_no_city = evaluate(pipe_no_city, X_test_no_city, y_test)

# Results Comparison
print("\n--- Model Evaluation Results ---")
print("With City:   ", score_with_city)
print("Without City:", score_no_city)