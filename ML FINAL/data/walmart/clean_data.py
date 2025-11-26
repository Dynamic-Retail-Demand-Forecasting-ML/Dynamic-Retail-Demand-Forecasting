import pandas as pd
import numpy as np

print("=" * 60)
print("WALMART SALES DATA CLEANING (SEPARATE FILES)")
print("=" * 60)

# Load datasets
print("\n[1/5] Loading datasets...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
features = pd.read_csv('features.csv', na_values=['NA'])
stores = pd.read_csv('stores.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Features shape: {features.shape}")
print(f"Stores shape: {stores.shape}")

# ============================================================================
# CLEAN FEATURES.CSV
# ============================================================================
print("\n[2/5] Cleaning features.csv...")
print("  - Checking null values before cleaning:")
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
for col in markdown_cols:
    null_count = features[col].isnull().sum()
    print(f"    {col}: {null_count} nulls ({null_count/len(features)*100:.1f}%)")

