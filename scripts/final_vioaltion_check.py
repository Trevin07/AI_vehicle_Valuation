import pandas as pd

df = pd.read_excel("after_pre.xlsx")

# 1. Check for Floor violations (< 2,000,000)
floor_violations = df[df['Fixed_value'] < 2000000]
num_floor_violations = len(floor_violations)

# 2. Check for Year-based Ceiling violations
def violates_ceiling(row):
    year = row['YEAR']
    price = row['Fixed_value']
    
    # Logic: 
    if year >= 2015 and price > 30000000:
        return True
    if 2005 <= year < 2015 and price > 25000000:
        return True
    if 2000 <= year < 2005 and price > 20000000:
        return True
    return False

# Identify ceiling violators
ceiling_mask = df.apply(violates_ceiling, axis=1)
ceiling_violations = df[ceiling_mask]
num_ceiling_violations = len(ceiling_violations)

# --- REPORTING ---
print(f"Violation Report for 'after_pre.xlsx':")
print(f"----------------------------------------")
print(f"Records violating 2M Floor: {num_floor_violations}")
print(f"Records violating Year Ceilings: {num_ceiling_violations}")

# 3. Drop Violators
if num_floor_violations > 0 or num_ceiling_violations > 0:
   
    df_cleaned = df[(df['Fixed_value'] >= 2000000) & (~ceiling_mask)].copy()
    
    print(f"Total records removed: {len(df) - len(df_cleaned)}")
    print(f"Records remaining: {len(df_cleaned)}")
    
    # Save to CSV file
    df_cleaned.to_csv("after_filling_final.csv", index=False)
    print("Cleaned data saved to 'after_filling_final.csv'")
else:
    print("No violations found. Dataset is already clean.")
    # Save the original data if no violations
    df.to_csv("after_filling_final.csv", index=False)
    print("Original data saved to 'after_filling_final.csv'")