import pandas as pd
import numpy as np
import os

# --- Configuration ---
FILE_PATH = "Standardized_Joined_Data.csv"

# UPDATED: 'year' has been added to the grouping criteria.
# These columns are now assumed to be in lowercase and stripped of whitespace
# after cleaning the DataFrame headers and data inside the function.
GROUPING_COLS = ['BRAND','MODEL','YEAR','TRANSMISSION','FUEL_TYPE']

# The target column is also assumed to be lowercase and stripped.
TARGET_COL = 'VEHICLE_PRICE'
CONSOLIDATED_COL = 'Fixed_value'
MIN_RECORDS_FOR_MEDIAN = 20

def conditional_agg(series):
    """
    Applies conditional aggregation logic:
    - If group size is < MIN_RECORDS_FOR_MEDIAN, return the mean.
    - If group size is >= MIN_RECORDS_FOR_MEDIAN, return the median.
    """
    if len(series) < MIN_RECORDS_FOR_MEDIAN:
        return series.mean()
    else:
        return series.median()

def calculate_representative_value(file_path):
    """
    Loads the data, *removes price outliers using the 1.5*IQR rule*,
    calculates the representative value for each unique vehicle type
    based on the conditional aggregation rule, and returns the result.
    """
    print(f"Loading data from {file_path}...")
    
    # 1. Load the dataset
    try:
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # --- CRITICAL CLEANING STEP 1: Headers ---
        original_columns = list(df.columns)
        df.columns = df.columns.str.strip().str.lower()
        print("Successfully cleaned column names to lowercase and stripped whitespace.")
        
        # Map the required columns to the cleaned column names (which are now lowercase)
        cleaned_grouping_cols = [col.strip().lower() for col in GROUPING_COLS]
        cleaned_target_col = TARGET_COL.strip().lower()
        
        # --- CRITICAL CLEANING STEP 2: Data Values (Internal Consistency) ---
        for col in cleaned_grouping_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().str.lower()
        
        print("Successfully cleaned data values in grouping columns (converted to lowercase).")
        
        # Ensure target column is numeric
        df[cleaned_target_col] = pd.to_numeric(df[cleaned_target_col], errors='coerce')
        
        # Drop rows with NaN in the required columns
        df.dropna(subset=cleaned_grouping_cols + [cleaned_target_col], inplace=True)
        
    except Exception as e:
        # This block prints the exact error.
        print(f"Error loading or processing CSV: {e}. ")
        print("This usually means the original column names in the CSV do not match the expected names.")
        if 'KeyError' in str(e) and hasattr(df, 'columns'):
            print(f"Available columns (original headers): {original_columns}")
            print(f"Available columns (cleaned/lowercase): {list(df.columns)}")
        return None
    
    initial_records_post_nan_drop = len(df)
    print(f"Total records loaded after initial cleaning and dropping NaN: {initial_records_post_nan_drop}")
    
    #
    # --- NEW: IQR FILTERING STEP (Outlier Removal) ---
    if initial_records_post_nan_drop > 0:
        q1 = df[cleaned_target_col].quantile(0.25)
        q3 = df[cleaned_target_col].quantile(0.75)
        iqr = q3 - q1
        
        # Define the bounds for outlier detection using the 1.5 * IQR rule
        lower_bound = q1 - 1.8 * iqr
        upper_bound = q3 + 1.8 * iqr
        
        # Filter the DataFrame, keeping prices within the bounds.
        # Use .copy() to avoid SettingWithCopyWarning
        df_filtered = df[(df[cleaned_target_col] >= lower_bound) & 
                        (df[cleaned_target_col] <= upper_bound)].copy()
        
        records_removed = initial_records_post_nan_drop - len(df_filtered)
        
        print(f"\n--- IQR Filtering on '{cleaned_target_col.upper()}' ---")
        print(f"Q1: {q1:,.2f}, Q3: {q3:,.2f}, IQR: {iqr:,.2f}")
        print(f"Lower Bound (Q1 - 1.8*IQR): {lower_bound:,.2f}")
        print(f"Upper Bound (Q3 + 1.8*IQR): {upper_bound:,.2f}")
        print(f"Records removed by IQR filter (Outliers): {records_removed}")
        print(f"Total records remaining: {len(df_filtered)}\n")
        
        # Update the main DataFrame reference
        df = df_filtered
    else:
        print("No records left after initial cleaning. Skipping IQR filtering.")
    
    # --- APPLY PRICE CEILING LOGIC BEFORE AGGREGATION ---
    
    #CHANGED-----


    if len(df) > 0:
        print("\n--- Applying Price Ceiling Logic (BEFORE aggregation) ---")
        initial_count_pre_ceiling = len(df)
        
        # A. Apply Floor Logic (> 2M)
        floor_mask = df[cleaned_target_col] > 2000000
        removed_by_floor = initial_count_pre_ceiling - floor_mask.sum()
        df = df[floor_mask].copy()
        
        # B. Apply Year-based Ceilings
        def check_ceiling_raw(row):
            year = row['year']
            price = row[cleaned_target_col]
            
            if year >= 2015 and price > 30000000:
                return False
            if 2005 <= year < 2015 and price > 25000000:
                return False
            if 2000 <= year < 2005 and price > 20000000:
                return False
            
            return True
        
        ceiling_mask = df.apply(check_ceiling_raw, axis=1)
        removed_by_ceiling = len(df) - ceiling_mask.sum()
        df = df[ceiling_mask].copy()
        
        #CHANGE ENDED------

        print(f"Records removed (Price < 2,000,000): {removed_by_floor}")
        print(f"Records removed (Year-based Ceilings): {removed_by_ceiling}")
        print(f"Records remaining for aggregation: {len(df)}\n")
    
    # 2. Group the data and apply the custom function
    # Check if there are still records after filtering
    if len(df) == 0:
        print("No records remaining after filtering for aggregation.")
        return None
    
    print(f"Grouping data by {GROUPING_COLS} and applying conditional aggregation...")
    
    # Group by the defining characteristics and apply the custom function to the target column
    consolidated_values = df.groupby(cleaned_grouping_cols).agg(
        # Calculate the size of the group
        Record_Count=(cleaned_grouping_cols[0], 'size'),
        # Calculate the representative value using the custom conditional_agg function
        **{CONSOLIDATED_COL: pd.NamedAgg(column=cleaned_target_col, aggfunc=conditional_agg)}
    ).reset_index()
    
    # Add a column indicating which aggregation method was used for clarity
    consolidated_values['Aggregation_Method'] = np.where(
        consolidated_values['Record_Count'] < MIN_RECORDS_FOR_MEDIAN,
        'MEAN (FEWER THAN 20 RECORDS)',
        'MEDIAN (20 OR MORE RECORDS)'
    )
    
    # --- FINAL CLEANING, TEXT SCRUBBING ---
    print("\n--- Final Processing: Text Cleaning ---")
    
    # 1. CLEAN SYMBOLS: Remove unwanted characters (#, $, %, etc.) and replace - / with space
    # This regex [^a-zA-Z0-9\s] means "everything that is NOT a letter, number, or space"
    for col in cleaned_grouping_cols:
        if consolidated_values[col].dtype == 'object':
            # Replace - and / with space first
            consolidated_values[col] = consolidated_values[col].str.replace(r'[-/]', ' ', regex=True)
            # Remove all other symbols (keeping only letters, numbers, and spaces)
            consolidated_values[col] = consolidated_values[col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
            # Clean up extra internal whitespace
            consolidated_values[col] = consolidated_values[col].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 1.5. REMOVE BRAND & DEDUPLICATE: Remove Brand name from Model and fix repeats
    def clean_model_redundancy(row):
        brand = str(row['brand']).strip().lower()
        model_str = str(row['model']).strip().lower()
        
        # Convert year to int first, then to string for comparison
        
        #CHANGED-----------
        try:
            year_int = int(row['year'])
            year_str = str(year_int)
        except:
            year_str = str(row['year']).strip()
        
        words = model_str.split()
        cleaned_words = []
        seen = set()
        
        for word in words:
            # Skip if word is brand OR a repeat OR if word equals the year
            if word != brand and word not in seen and word != year_str:
                cleaned_words.append(word)
                seen.add(word)
        

        #CHANGE ENDED-----


        return " ".join(cleaned_words)
    
    print("Cleaning MODEL names: removing brands, symbols, years, and duplicates...")
    consolidated_values['model'] = consolidated_values.apply(clean_model_redundancy, axis=1)
    
    # 2. Convert to UPPERCASE
    cols_to_uppercase = cleaned_grouping_cols + ['Aggregation_Method']
    for col in cols_to_uppercase:
        if col in consolidated_values.columns and consolidated_values[col].dtype == 'object':
            consolidated_values[col] = consolidated_values[col].str.upper()
    
    # --- FIX DUPLICATES: Drop duplicates based on grouping columns ---
   
   
    #CHANGED---------
    print("\n--- Removing duplicate vehicle entries ---")
    pre_dedup_count = len(consolidated_values)
    consolidated_values = consolidated_values.drop_duplicates(subset=cleaned_grouping_cols, keep='first')
    duplicates_removed = pre_dedup_count - len(consolidated_values)
    print(f"Duplicate records removed: {duplicates_removed}")

    #CHANGE ENDED--------

    print(f"Final unique records: {len(consolidated_values)}")
    
    # Final Column Renaming to UPPERCASE
    new_cols = {col: col.upper() for col in cleaned_grouping_cols}
    consolidated_values.rename(columns=new_cols, inplace=True)
    
    return consolidated_values

# Execute the function
consolidated_df = calculate_representative_value(FILE_PATH)

if consolidated_df is not None:
    # Optional: Save the resulting unique vehicle list with representative values
    output_file = "after_pre.xlsx"
    consolidated_df.to_excel(output_file, index=False)
    print(f"\nFinal unique vehicle types and their representative values saved to {output_file}")