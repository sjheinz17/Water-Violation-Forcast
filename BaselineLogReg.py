# #%%
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, roc_auc_score

# # --- 1. INDEPENDENT CLEANING FUNCTION ---
# def clean_water_data(df_in):
#     """Standardizes cleaning for any year's dataframe"""
#     df = df_in.copy()
    
#     # Rename columns to standard format
#     column_mapping = {
#         "Water System Number": "Water System Primary Key",
#         "System No": "Water System Primary Key",
#         "System Number": "Water System Primary Key",
#         "rst": "Result",
#         "mcl": "MCL",
#         "finding": "Result"
#     }
#     df = df.rename(columns=column_mapping)
#     # Remove duplicates created by renaming
#     df = df.loc[:, ~df.columns.duplicated()]

#     # Basic filtering
#     if "Water System Primary Key" not in df.columns:
#         return pd.DataFrame()
#     if "System Status" in df.columns:
#         df = df[df["System Status"] == "A"]
    
#     df = df.dropna(subset=["Result", "MCL", "Water System Primary Key"])
    
#     # Numeric Coercion
#     cols_to_num = ["Result", "MCL", "Reporting Level", "Population Served", "Service Connections"]
#     for c in cols_to_num:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")
            
#     # Non-Detect Handling
#     if "Less Than Reporting Level" in df.columns and "Reporting Level" in df.columns:
#         mask_nd = df["Less Than Reporting Level"] == "Y"
#         df.loc[mask_nd, "Result"] = df["Reporting Level"] / 2.0
        
#     return df

# #%%
# # --- 2. LOAD ONLY 2024 & 2025 (NAIVE APPROACH) ---
# # We deliberately ignore 2020-2023 to simulate a "Simple" model
# print("Loading baseline data (2024 & 2025 only)...")
# try:
#     df_2024 = pd.read_csv("2024.csv")
#     df_2025 = pd.read_csv("2025.csv")
# except FileNotFoundError:
#     print("Error: Please ensure 2024.csv and 2025.csv are in the folder.")

# df_2024_clean = clean_water_data(df_2024)
# df_2025_clean = clean_water_data(df_2025)

# # --- 3. FEATURE ENGINEERING (SINGLE YEAR) ---
# features_group = ["Water System Primary Key", "Analyte Name"]



# # Features from 2024
# X_features = (
#     df_2024_clean
#     .groupby(features_group)
#     .agg(
#         prior_year_mean=("Result", "mean"),
#         prior_year_max=("Result", "max"),
#         prior_year_failures=("Result", lambda x: (x > df_2024_clean.loc[x.index, "MCL"]).sum())
#     )
#     .reset_index()
# )

# # Metadata
# system_meta = (
#     df_2024_clean
#     .sort_values("Population Served", ascending=False)
#     .groupby("Water System Primary Key")
#     .agg(
#         principal_county=("Principal County Served", "first"),
#         classification=("Water System Classification", "first"),
#         pop_served=("Population Served", "max"),
#         facility_water_type=("Facility Water Type", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown")
#     )
#     .reset_index()
# )
# X_features = X_features.merge(system_meta, on="Water System Primary Key", how="left")

# # Target from 2025
# y_target = (
#     df_2025_clean
#     .groupby(features_group)
#     .agg(future_violation=("Result", lambda x: (x > df_2025_clean.loc[x.index, "MCL"]).max()))
#     .reset_index()
# )
# y_target["future_violation"] = y_target["future_violation"].astype(int)

# # Merge
# naive_df = X_features.merge(y_target, on=features_group, how="inner")
# naive_df = naive_df.fillna(0) # Simple imputation

# # --- 4. PREPARE THE PIPELINE ---

# num_cols = ["prior_year_mean", "prior_year_max", "prior_year_failures", "pop_served"]
# cat_cols = ["principal_county", "classification", "facility_water_type"]

# # Handle Missing Values in the DataFrame first (Cleanest approach)
# naive_df[num_cols] = naive_df[num_cols].fillna(0)
# naive_df[cat_cols] = naive_df[cat_cols].fillna("Unknown")

# # CRITICAL FIX: Force all categorical columns to string to prevent TypeError
# for col in cat_cols:
#     naive_df[col] = naive_df[col].astype(str)

# # Define Preprocessor
# preprocessor = ColumnTransformer([
#     ("num", StandardScaler(), num_cols),
#     ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
# ])

# # Define Naive Model (Logistic Regression)
# logreg_pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("logreg", LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
# ])

# # --- 5. TRAIN & EVALUATE ---
# X = naive_df[num_cols + cat_cols]
# y = naive_df["future_violation"]

# # Simple 80/20 split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# print("Training Naive Baseline...")
# logreg_pipeline.fit(X_train, y_train)

# y_pred = logreg_pipeline.predict(X_test)
# y_proba = logreg_pipeline.predict_proba(X_test)[:, 1]

# print("\n=== NAIVE BASELINE RESULTS (LogReg, No Stacking) ===")
# print(classification_report(y_test, y_pred))
# print(f"Baseline AUC: {roc_auc_score(y_test, y_proba):.4f}")



#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score

# --- 1. DATA LOADING (2021-2024 for Training, 2025 for Testing) ---

# We will aggregate 2021-2024 into one big training set
train_years = [2021, 2022, 2023, 2024]
test_year = 2025

def clean_water_data(df_in):
    """Standardizes cleaning"""
    df = df_in.copy()
    column_mapping = {
        "Water System Number": "Water System Primary Key",
        "System No": "Water System Primary Key",
        "System Number": "Water System Primary Key",
        "rst": "Result", "mcl": "MCL", "finding": "Result"
    }
    df = df.rename(columns=column_mapping)
    df = df.loc[:, ~df.columns.duplicated()] # Fix duplicate col error

    if "Water System Primary Key" not in df.columns: return pd.DataFrame()
    if "System Status" in df.columns: df = df[df["System Status"] == "A"]
    
    df = df.dropna(subset=["Result", "MCL", "Water System Primary Key"])
    
    cols_to_num = ["Result", "MCL", "Reporting Level", "Population Served", "Service Connections"]
    for c in cols_to_num:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
            
    if "Less Than Reporting Level" in df.columns and "Reporting Level" in df.columns:
        mask_nd = df["Less Than Reporting Level"] == "Y"
        df.loc[mask_nd, "Result"] = df["Reporting Level"] / 2.0
        
    return df

# Helper to build (X -> y) for a specific year pair
def create_dataset(past_year, future_year):
    try:
        df_past = pd.read_csv(f"{past_year}.csv")
        df_future = pd.read_csv(f"{future_year}.csv")
    except FileNotFoundError:
        print(f"Skipping {past_year}->{future_year} (File not found)")
        return pd.DataFrame()

    df_past = clean_water_data(df_past)
    df_future = clean_water_data(df_future)

    # FEATURES (From Past)
    features = (
        df_past.groupby(["Water System Primary Key", "Analyte Name"])
        .agg(
            prior_year_mean=("Result", "mean"),
            prior_year_max=("Result", "max"),
            # NOTE: We REMOVED 'prior_year_failures' to make this model simpler/worse
        ).reset_index()
    )
    
    # METADATA
    meta = (
        df_past.sort_values("Population Served", ascending=False)
        .groupby("Water System Primary Key")
        .agg(
            principal_county=("Principal County Served", "first"),
            classification=("Water System Classification", "first"),
            pop_served=("Population Served", "max"),
            facility_water_type=("Facility Water Type", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown")
        ).reset_index()
    )
    
    # TARGET (From Future)
    target = (
        df_future.groupby(["Water System Primary Key", "Analyte Name"])
        .agg(future_violation=("Result", lambda x: (x > df_future.loc[x.index, "MCL"]).max()))
        .reset_index()
    )
    target["future_violation"] = target["future_violation"].astype(int)

    # MERGE
    return features.merge(meta, on="Water System Primary Key").merge(target, on=["Water System Primary Key", "Analyte Name"])

# --- 2. BUILD DATASETS ---

# Train on history (2021->2022, 2022->2023, 2023->2024)
print("Building Historical Training Set...")
train_dfs = []
for y in train_years[:-1]: # Stop one before the end to allow for the +1 year target
    print(f"  - Training Pair: {y} predicting {y+1}")
    ds = create_dataset(y, y+1)
    train_dfs.append(ds)

full_train = pd.concat(train_dfs)

# Test strictly on 2024 -> 2025
print("Building 2025 Test Set...")
final_test = create_dataset(2024, 2025)

# --- 3. MODELING (NAIVE LOGREG) ---

# Notice: 'prior_year_failures' is GONE.
num_cols = ["prior_year_mean", "prior_year_max", "pop_served"]
cat_cols = ["principal_county", "classification", "facility_water_type"]

# Force strings for categories
full_train[cat_cols] = full_train[cat_cols].fillna("Unknown").astype(str)
final_test[cat_cols] = final_test[cat_cols].fillna("Unknown").astype(str)
full_train[num_cols] = full_train[num_cols].fillna(0)
final_test[num_cols] = final_test[num_cols].fillna(0)

# Define X and y
X_train = full_train[num_cols + cat_cols]
y_train = full_train["future_violation"]

X_test = final_test[num_cols + cat_cols]
y_test = final_test["future_violation"]

# Pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

logreg = Pipeline([
    ("preprocessor", preprocessor),
    # Regularization (C=0.1) limits complexity further, making it 'dumber'
    ("logreg", LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)) 
])

# Train
print("\nTraining Naive Baseline (History predicting Future)...")
logreg.fit(X_train, y_train)

# Evaluate
y_pred = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)[:, 1]

print("\n=== NAIVE BASELINE RESULTS (Without 'Prior Failure' Feature) ===")
print(classification_report(y_test, y_pred))
print(f"Baseline AUC: {roc_auc_score(y_test, y_proba):.4f}")
# %%
