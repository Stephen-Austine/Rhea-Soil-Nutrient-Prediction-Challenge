import pandas as pd

# Load training data
train = pd.read_csv("Train.csv")

print("=== Train.csv Columns ===")
for col in train.columns:
    print(col)

print("\n=== Data Sample ===")
print(train[['N', 'P', 'K', 'ph']].head())

print("\n=== Nutrient Columns Presence ===")
nutrients = ['N', 'P', 'K', 'Ca', 'Mg']
for nutrient in nutrients:
    if nutrient in train.columns:
        print(f"✓ {nutrient} found")
        print(f"  Non-NaN values: {train[nutrient].count()}")
        print(f"  Sample values: {train[nutrient].dropna().head().values}")
    else:
        print(f"✗ {nutrient} NOT found")

print("\n=== pH Column ===")
if 'ph' in train.columns:
    print(f"✓ ph found")
    print(f"  Non-NaN values: {train['ph'].count()}")
    print(f"  Sample values: {train['ph'].dropna().head().values}")
else:
    print("✗ ph NOT found")
