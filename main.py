import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Students Social Media Addiction.csv")

# Show first 5 rows
print("FIRST 5 ROWS:")
print(df.head())

# Dataset information
print("\nDATASET INFO:")
print(df.info())

# Missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset cleaned successfully!")