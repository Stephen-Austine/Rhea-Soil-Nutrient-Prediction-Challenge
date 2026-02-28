import pandas as pd

train = pd.read_csv("Train.csv")

print("=== Column Names ===")
for col in train.columns:
    print(f"'{col}'")

print("\n=== pH-related columns ===")
for col in train.columns:
    if 'ph' in col.lower() or 'pH' in col:
        print(f"✓ Found: '{col}'")
        print(f"  Non-NaN values: {train[col].count()}")
        print(f"  Sample values: {train[col].dropna().head().values}")

print("\n=== Nutrient columns ===")
nutrients = ['N', 'P', 'K', 'Ca', 'Mg']
for n in nutrients:
    if n in train.columns:
        print(f"✓ {n}: {train[n].count()} non-NaN values")
    else:
        print(f"✗ {n}: NOT FOUND")