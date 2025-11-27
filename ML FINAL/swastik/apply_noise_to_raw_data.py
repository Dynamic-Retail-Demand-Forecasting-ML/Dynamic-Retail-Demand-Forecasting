import pandas as pd
import numpy as np
import os

# Constants
DATA_PATH_TRAIN = r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\data\walmart\train.csv"
DATA_PATH_FEAT  = r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\data\walmart\features.csv"
RANDOM_STATE = 42

def apply_noise():
    print("APPLYING NOISE TO RAW DATASETS...")
    print("="*60)
    
    # 1. Process features.csv (Temperature, Fuel_Price, CPI, Unemployment)
    print(f"Loading {DATA_PATH_FEAT}...")
    df_feat = pd.read_csv(DATA_PATH_FEAT)
    print(f"Original shape: {df_feat.shape}")
    
    np.random.seed(RANDOM_STATE)
    
    # General noise (30%) for Temperature, Fuel_Price
    for col in ['Temperature', 'Fuel_Price']:
        if col in df_feat.columns:
            std = df_feat[col].std()
            noise = np.random.normal(0, std * 0.30, size=len(df_feat))
            df_feat[col] = df_feat[col] + noise
            print(f" -> Added 30% noise to {col}")
            
    # Higher noise (40%) for KNN targets (CPI, Unemployment)
    for col in ['CPI', 'Unemployment']:
        if col in df_feat.columns:
            # Handle NaNs first if any, though we usually fill them later. 
            # Ideally we add noise to existing values.
            mask = ~df_feat[col].isna()
            std = df_feat.loc[mask, col].std()
            noise = np.random.normal(0, std * 0.40, size=mask.sum())
            df_feat.loc[mask, col] = df_feat.loc[mask, col] + noise
            print(f" -> Added 40% noise to {col} (KNN Target)")
            
    # Save back
    df_feat.to_csv(DATA_PATH_FEAT, index=False)
    print("✓ Saved modified features.csv")
    
    # 2. Process train.csv (Weekly_Sales)
    print(f"\nLoading {DATA_PATH_TRAIN}...")
    df_train = pd.read_csv(DATA_PATH_TRAIN)
    print(f"Original shape: {df_train.shape}")
    
    # General noise (30%) for Weekly_Sales
    if 'Weekly_Sales' in df_train.columns:
        std = df_train['Weekly_Sales'].std()
        noise = np.random.normal(0, std * 0.30, size=len(df_train))
        df_train['Weekly_Sales'] = df_train['Weekly_Sales'] + noise
        print(f" -> Added 30% noise to Weekly_Sales")
        
    # Save back
    df_train.to_csv(DATA_PATH_TRAIN, index=False)
    print("✓ Saved modified train.csv")
    
    print("\nDONE! Raw datasets have been permanently modified.")

if __name__ == "__main__":
    apply_noise()
