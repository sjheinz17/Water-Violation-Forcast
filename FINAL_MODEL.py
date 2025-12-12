#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, validation_curve
from sklearn.metrics import classification_report, roc_auc_score
from scipy.stats import randint
import matplotlib.pyplot as plt

# --- 1. DATA LOADING & CLEANING ---

# Load all years (assuming files are named 2020.csv, 2021.csv, etc.)
years = [2020, 2021, 2022, 2023, 2024, 2025]
data_frames = {}
#%%
print("Loading data files...")
for y in years:
    # Replace with your actual file paths if they differ
    try:
        data_frames[y] = pd.read_csv(f"{y}.csv")
        print(f"Loaded {y}.csv")
    except FileNotFoundError:
        print(f"Warning: Could not find {y}.csv")

#%%

def clean_water_data(df_in):
    """Standardizes cleaning for any year's dataframe"""
    df = df_in.copy()
    
    # --- FIX 1: Rename columns using a map ---
    column_mapping = {
        "Water System Number": "Water System Primary Key",
        "System No": "Water System Primary Key",
        "System Number": "Water System Primary Key",
        "rst": "Result",
        "mcl": "MCL",
        "finding": "Result"
    }
    df = df.rename(columns=column_mapping)

    # --- FIX 2: Remove Duplicate Columns ---
    # If a file had BOTH "Water System Number" and "Water System Primary Key",
    # the rename above creates two columns with the same name. We keep only the first.
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Check if the critical key exists now
    if "Water System Primary Key" not in df.columns:
        print(f"ERROR: Could not find system ID column. Available columns: {list(df.columns)}")
        return pd.DataFrame() 

    # 1. Filter active systems
    if "System Status" in df.columns:
        df = df[df["System Status"] == "A"]
    
    # Drop critical missing data
    df = df.dropna(subset=["Result", "MCL", "Water System Primary Key"])
    
    # 2. Coerce Numeric Columns
    cols_to_num = ["Result", "MCL", "Reporting Level", "Population Served", "Service Connections"]
    for c in cols_to_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    # 3. Handle Non-Detects (1/2 Reporting Level)
    if "Less Than Reporting Level" in df.columns and "Reporting Level" in df.columns:
        mask_nd = df["Less Than Reporting Level"] == "Y"
        df.loc[mask_nd, "Result"] = df["Reporting Level"] / 2.0
        
    return df

# Apply cleaning to all loaded frames
for y in data_frames:
    data_frames[y] = clean_water_data(data_frames[y])


# --- 2. THE "TIME MACHINE" FUNCTION ---

def create_lagged_dataset(df_past, df_future):
    """
    Takes a Past Year (Features) and a Future Year (Targets)
    and merges them into a single dataset for training.
    """
    
    features_group = ["Water System Primary Key", "Analyte Name"]

    # A. FEATURES (From the Past)
    features_past = (
        df_past
        .groupby(features_group)
        .agg(
            prior_year_mean=("Result", "mean"),
            prior_year_max=("Result", "max"),
            prior_year_std=("Result", "std"),
            prior_year_count=("Result", "count"),
            # Did they fail in the PAST year?
            prior_year_failures=("Result", lambda x: (x > df_past.loc[x.index, "MCL"]).sum())
        )
        .reset_index()
    )

    # B. METADATA (From the Past)
    system_meta = (
        df_past
        .sort_values("Population Served", ascending=False)
        .groupby("Water System Primary Key")
        .agg(
            principal_county=("Principal County Served", "first"),
            classification=("Water System Classification", "first"),
            pop_served=("Population Served", "max"),
            service_connections=("Service Connections", "max"),
            facility_water_type=("Facility Water Type", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown")
        )
        .reset_index()
    )
    
    # Merge Metadata
    X_df = features_past.merge(system_meta, on="Water System Primary Key", how="left")

    # C. TARGETS (From the Future)
    target_future = (
        df_future
        .groupby(features_group)
        .agg(
            # Did they fail in the FUTURE year?
            future_violation=("Result", lambda x: (x > df_future.loc[x.index, "MCL"]).max())
        )
        .reset_index()
    )
    target_future["future_violation"] = target_future["future_violation"].astype(int)

    # D. COMBINE
    merged_df = X_df.merge(
        target_future, 
        on=["Water System Primary Key", "Analyte Name"], 
        how="inner"
    )
    
    return merged_df

# --- 3. STACKING THE YEARS (BUILDING HISTORY) ---

train_datasets = []

# We create pairs: (2020->21), (2021->22), (2022->23), (2023->24)
training_pairs = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]

print("Building Training Stack (Historical Data)...")
for past_year, future_year in training_pairs:
    if past_year in data_frames and future_year in data_frames:
        print(f"  - Stacking: {past_year} predicting {future_year}")
        ds = create_lagged_dataset(data_frames[past_year], data_frames[future_year])
        train_datasets.append(ds)

# Concatenate all historical pairs into one massive training set
full_train_df = pd.concat(train_datasets, ignore_index=True)

# Build the FINAL TEST set: strictly 2024 predicting 2025
print("Building Test Set (2024 -> 2025)...")
final_test_df = create_lagged_dataset(data_frames[2024], data_frames[2025])

print(f"Training Rows: {len(full_train_df)}")
print(f"Testing Rows: {len(final_test_df)}")

# --- 4. PREPARE VECTORS ---

num_features = [
    "prior_year_mean", "prior_year_max", "prior_year_std", 
    "prior_year_count", "prior_year_failures",
    "pop_served", "service_connections"
]
cat_features = ["principal_county", "classification", "facility_water_type"]

# Handle NaNs created by lag (e.g., std dev of 1 sample)
full_train_df[num_features] = full_train_df[num_features].fillna(0)
full_train_df[cat_features] = full_train_df[cat_features].fillna("Unknown")

final_test_df[num_features] = final_test_df[num_features].fillna(0)
final_test_df[cat_features] = final_test_df[cat_features].fillna("Unknown")

# Define X and y
X_train = full_train_df[num_features + cat_features]
y_train = full_train_df["future_violation"]

X_test = final_test_df[num_features + cat_features]
y_test = final_test_df["future_violation"]

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

# Base Pipeline
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
])

# --- 5. HYPERPARAMETER TUNING (RandomizedSearch) ---

param_dist = {
    'classifier__n_estimators': randint(100, 500),
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': randint(2, 20),
    'classifier__min_samples_leaf': randint(1, 10),
    'classifier__class_weight': ['balanced', 'balanced_subsample', None]
}

random_search = RandomizedSearchCV(
    clf,                            
    param_distributions=param_dist,
    n_iter=15,                      # Reduced slightly for speed since dataset is larger now
    cv=3,                           
    scoring='roc_auc',              
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nStarting Hyperparameter Tuning on Stacked History...")
random_search.fit(X_train, y_train)

print(f"Best AUC Score (CV): {random_search.best_score_:.4f}")
print("Best Parameters found:", random_search.best_params_)

best_model = random_search.best_estimator_

# --- 6. VALIDATION CURVE (Complexity Check) ---

param_range = [2, 5, 10, 15, 20, 30] # Trimmed range for speed
print("\nGenerating complexity curve (this takes a moment)...")

train_scores, test_scores = validation_curve(
    best_model, # Use the best architecture found
    X_train, 
    y_train, 
    param_name="classifier__max_depth", 
    param_range=param_range,
    cv=3, 
    scoring="roc_auc", 
    n_jobs=-1
)

# Plotting
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.title("Validation Curve: Impact of Tree Depth (Multi-Year Training)")
plt.xlabel("Max Depth")
plt.ylabel("ROC-AUC Score")
plt.ylim(0.5, 1.1)
lw = 2
plt.plot(param_range, train_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange")
plt.plot(param_range, test_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# --- 7. INTERPRETATION & FINAL EVALUATION ---

params = random_search.best_params_
score = random_search.best_score_

print(f"\n=== OPTIMIZED MODEL REPORT ===")
print(f"Final Tuned AUC Score: {score:.4f}")
print("-" * 30)

# Interpret Tree Depth
depth = params['classifier__max_depth']
print(f"1. Max Depth: {depth}")
if depth is None:
    print("   -> INTERPRETATION: Unlimited depth. The model found complex historical patterns.")
elif depth < 15:
    print("   -> INTERPRETATION: Shallow trees. The model prioritized preventing overfitting.")
else:
    print("   -> INTERPRETATION: Moderate depth. Balanced approach.")

# Interpret Class Weight
weight = params['classifier__class_weight']
print(f"2. Class Weight: {weight}")
if weight is not None:
    print("   -> INTERPRETATION: The model is heavily penalizing missed violations (Good for safety).")
else:
    print("   -> INTERPRETATION: The historical signal was strong enough without artificial weighting.")

# Final Prediction on 2025 Data
print("\n=== FINAL PERFORMANCE (Predicting 2025 Violations) ===")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))


#%%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 1. Get probabilities for the positive class (Violations)
# (best_model is your tuned Random Forest from the previous step)
y_proba = best_model.predict_proba(X_test)[:, 1]

# 2. Calculate the Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# 3. Plot
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

# Formatting
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (False Alarms)', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curve: Predicting 2025 Water Violations', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)

plt.show()
# %%

# Parameter range to test
param_range = [2, 5, 10, 15, 20, 30]

# Compute validation curve
train_scores, val_scores = validation_curve(
    estimator=best_model,                 # tuned pipeline
    X=X_train,
    y=y_train,
    param_name="classifier__max_depth",    # must match pipeline name
    param_range=param_range,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

# Aggregate results
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, label="Training ROC-AUC", linewidth=2)
plt.fill_between(
    param_range,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.2
)

plt.plot(param_range, val_mean, label="Validation ROC-AUC", linewidth=2)
plt.fill_between(
    param_range,
    val_mean - val_std,
    val_mean + val_std,
    alpha=0.2
)

plt.xlabel("Max Tree Depth")
plt.ylabel("ROC-AUC")
plt.title("Validation Curve: Random Forest Model Complexity")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
