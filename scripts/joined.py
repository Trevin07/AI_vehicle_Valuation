import pandas as pd
import numpy as np
from datetime import datetime

# 1. Configuration: Target columns for the final file
STANDARD_COLS = [
    'VEHICLE_PRICE', 'BRAND', 'MODEL', 'YEAR', 'TRANSMISSION', 'FUEL_TYPE'
]

def process_and_standardize(df, mapping, dataset_name):
    """
    Cleans and standardizes a dataset:
    - Joins Trim to Model ONLY if Trim is not '0' or empty.
    - Standardizes column names via mapping.
    - Reports column mismatches (Missing required columns).
    """
    if df is None or df.empty:
        print(f"--- {dataset_name} is empty or not provided, skipping. ---")
        return pd.DataFrame(columns=STANDARD_COLS)

    print(f"\n--- Checking {dataset_name} ---")
    temp_df = df.copy()

    # Step A: Model and Trim Joining Logic
    # Identify Trim column (handles common variations)
    trim_cols = ['Trim / Edition', 'trim', 'Edition', 'Trim']
    found_trim = next((c for c in trim_cols if c in temp_df.columns), None)
    
    # Identify Model column
    model_candidates = ['model', 'Model', 'MODEL', 'MODEL_NAME']
    found_model = next((c for c in model_candidates if c in temp_df.columns), None)

    if found_model:
        temp_df[found_model] = temp_df[found_model].fillna('').astype(str).str.strip()
        
        if found_trim:
            print(f"-> Found trim column '{found_trim}'. Checking values for joining...")
            temp_df[found_trim] = temp_df[found_trim].fillna('').astype(str).str.strip()
            
            # Logic: Only join if trim is NOT '0' and NOT empty
            # If it's '0', we ignore the trim and just keep the model name.
            temp_df[found_model] = np.where(
                (temp_df[found_trim] != '0') & (temp_df[found_trim] != ''),
                temp_df[found_model] + " " + temp_df[found_trim],
                temp_df[found_model]
            )
            temp_df[found_model] = temp_df[found_model].str.strip()

    # Step B: Rename columns to standard names
    temp_df = temp_df.rename(columns=mapping)

    # Step C: Mismatch Detection & Warning
    missing = [col for col in STANDARD_COLS if col not in temp_df.columns]
    if missing:
        print(f"⚠️  WARNING: {dataset_name} is missing these standard columns: {missing}")
        for col in missing:
            temp_df[col] = np.nan
    else:
        print(f"✅ {dataset_name} matched all standard columns.")

    # Return only the 7 standard columns in the correct order
    return temp_df.reindex(columns=STANDARD_COLS)

# --- 2. Define Mappings (Customized to your provided column headers) ---
map1 = {
    'vehicle_price': 'VEHICLE_PRICE', 'brand': 'BRAND', 'model': 'MODEL',
    'year': 'YEAR', 'mileage_km': 'MILEAGE_KM', 'transmission': 'TRANSMISSION',
    'fuel_type': 'FUEL_TYPE'
}

map2_3 = {
    'vehicle_price': 'VEHICLE_PRICE', 'brand': 'BRAND', 'model': 'MODEL',
    'year': 'YEAR', 'mileage_km': 'MILEAGE_KM', 'Transmission': 'TRANSMISSION',
    'fuel_type': 'FUEL_TYPE'
}

# --- 3. Load Your Datasets ---

data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')
data3 = pd.read_csv('data3.csv')

# --- 4. Process and Combine ---
df1_clean = process_and_standardize(data1, map1, "Dataset 1")
df2_clean = process_and_standardize(data2, map2_3, "Dataset 2")
df3_clean = process_and_standardize(data3, map2_3, "Dataset 3")

final_df = pd.concat([df1_clean, df2_clean, df3_clean], ignore_index=True)

# --- 5. Final Quality Filtering ---
initial_len = len(final_df)

# Filter 1: Remove numeric-only models (Must contain at least one letter)
final_df = final_df[final_df['MODEL'].str.contains('[a-zA-Z]', na=False, regex=True)]
after_model_filter = len(final_df)

# Filter 2: Keep only records where YEAR >= 2000
# (Converts years to numbers first; invalid/empty years become NaN and are removed)
final_df['YEAR'] = pd.to_numeric(final_df['YEAR'], errors='coerce')
final_df = final_df[final_df['YEAR'] >= 2000]
final_len = len(final_df)
final_df['PROCESSED_DATE'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Summary of cleaning
print(f"\n--- Cleaning Summary ---")
print(f"Dropped due to numeric-only model: {initial_len - after_model_filter}")
print(f"Dropped due to year < 2000: {after_model_filter - final_len}")
print(f"Total records remaining: {final_len}")

# --- 6. Save to Excel ---
final_df.to_csv('Standardized_Joined_Data.csv', index=False)
print("\n🚀 SUCCESS: Final file saved as 'Standardized_Joined_Data.csv'")