import pandas as pd
import numpy as np
import os

def make_data_raw(input_file, output_file):
    print(f"Reading from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except PermissionError:
        print(f"\nERROR: Permission denied for {input_file}.")
        print("The file is likely open in Excel or another program, or locked by OneDrive.")
        print("Please CLOSE the file and try again.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 1. Modify Temperature
    # Adding random noise to Temperature to make it 'rawer' / less precise
    print("Adding noise to Temperature...")
    if 'Temperature' in df.columns:
        # Add Gaussian noise
        noise = np.random.normal(0, 5, df['Temperature'].shape) 
        df['Temperature'] = df['Temperature'] + noise
        
        # Randomly set 5% to NaN
        mask = np.random.random(df['Temperature'].shape) < 0.05 
        df.loc[mask, 'Temperature'] = np.nan

    # 2. Modify MarkDown1 - MarkDown5
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    
    for col in markdown_cols:
        if col in df.columns:
            print(f"Modifying {col}...")
            # Add noise to non-null values
            not_null_mask = df[col].notnull()
            noise = np.random.normal(0, 50, not_null_mask.sum())
            df.loc[not_null_mask, col] = df.loc[not_null_mask, col] + noise
            
            # Randomly drop 10% more values
            drop_mask = np.random.random(df[col].shape) < 0.10
            df.loc[drop_mask, col] = np.nan

    print(f"Saving modified data to {output_file}...")
    try:
        df.to_csv(output_file, index=False)
        print("Done!")
    except Exception as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    # Paths
    input_csv = r"D:\features.csv"
    output_csv = r"D:\features.csv"
    
    make_data_raw(input_csv, output_csv)
