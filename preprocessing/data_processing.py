"""
Data Processing Pipeline for Cyber Insurance Analytics
Preserves all intermediate outputs for transparency
"""

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

# Configuration
LABEL_DATA_PATH = 'classification/data/label_w_year.csv'
CYBER_DATA_PATH = 'classification/data/cyber_cleaned.csv'
OUTPUT_PATH = 'classification/data/label_cyber_year.csv'

def main():
    # Initialize settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 50)
    print("="*60 + "\nDATA PROCESSING PIPELINE STARTED\n" + "="*60)

    # Phase 1: Load raw data
    print("\n>>> PHASE 1: LOADING RAW DATA")
    label_df = pd.read_csv(LABEL_DATA_PATH, index_col=False, low_memory=False)
    print(f"Original data shape: {label_df.shape}")
    print("\nFirst 5 records:")
    print(label_df.head())
    
    # Phase 2: Data quality analysis
    print("\n>>> PHASE 2: DATA QUALITY ANALYSIS")
    print("\nData summary:")
    print(label_df.info())
    
    # Missing value analysis
    missing_counts = label_df.isnull().sum()
    missing_pct = (missing_counts / len(label_df)) * 100
    print("\nMissing value report:")
    print(pd.DataFrame({'Missing Count': missing_counts, 
                      'Percentage (%)': missing_pct.round(2)}))

    # Phase 3: Data cleaning
    print("\n>>> PHASE 3: DATA CLEANING")
    label_clean = label_df[label_df['COMPANY_ID'].notnull()].copy()
    print(f"Cleaned data shape (non-null COMPANY_ID): {label_clean.shape}")
    # Remove records where ACCIDENT_YEAR is missing
    label_clean = label_clean.dropna(subset=['ACCIDENT_YEAR'])
    print(f"Records after removing missing ACCIDENT_YEAR: {label_clean.shape}")
    print(f"Unique COMPANY_ID count: {label_clean.COMPANY_ID.nunique()}")

    # Phase 4: Duplicate analysis
    print("\n>>> PHASE 4: DUPLICATE ANALYSIS")
    column_groups = [
        ['COMPANY_ID', 'CASE_TYPE_LG'],
        ['COMPANY_ID', 'CASE_TYPE_LG', 'ACCIDENT_YEAR'],
        ['COMPANY_ID', 'CASE_TYPE_LG', 'ACCIDENT_YEAR', 'TOTAL_AMOUNT'],
        ['COMPANY_ID', 'CASE_TYPE_LG', 'ACCIDENT_YEAR', 'TOTAL_AMOUNT', 'MSCAD_ID']
    ]

    duplicate_stats = {}
    for cols in column_groups:
        group = label_clean.groupby(cols)
        duplicate_groups = group.size().reset_index(name='COUNT')
        duplicate_stats[len(cols)] = (duplicate_groups['COUNT'] > 1).sum()
    
    print("\nDuplicate group statistics:")
    print(pd.DataFrame({
        'Columns Matched': list(duplicate_stats.keys()),
        'Duplicate Groups': list(duplicate_stats.values())
    }))

    # Phase 5: Multi-stage deduplication
    print("\n>>> PHASE 5: DEDUPLICATION")
    
    # First pass (5 columns)
    print("\nFirst deduplication pass (5 columns)...")
    cols_5 = ['COMPANY_ID', 'CASE_TYPE_LG', 'ACCIDENT_YEAR', 'TOTAL_AMOUNT', 'MSCAD_ID']
    label_dedup = label_clean.drop_duplicates(subset=cols_5, keep='first')
    print(f"Records after first pass: {len(label_dedup)} "
          f"(Reduced by {len(label_clean)-len(label_dedup)})")

    # Remove MSCAD_ID
    label_dedup = label_dedup.drop(columns='MSCAD_ID')
    print("\nData shape after MSCAD_ID removal:", label_dedup.shape)

    # Second pass (4 columns)
    print("\nSecond deduplication pass (4 columns)...")
    cols_4 = ['COMPANY_ID', 'CASE_TYPE_LG', 'ACCIDENT_YEAR', 'TOTAL_AMOUNT']
    n_before = len(label_dedup)
    label_dedup = label_dedup.drop_duplicates(subset=cols_4, keep='first')
    print(f"Records after second pass: {len(label_dedup)} "
          f"(Reduced by {n_before - len(label_dedup)})")

    # Phase 6: Temporal analysis
    print("\n>>> PHASE 6: TEMPORAL ANALYSIS")
    year_dist = label_dedup['ACCIDENT_YEAR'].value_counts().sort_index()
    print("\nYear distribution summary:")
    print(year_dist.describe())
    
    plt.figure(figsize=(12, 6))
    year_dist.plot(kind='bar', title='Claim Distribution by Year')
    plt.xlabel('Accident Year')
    plt.ylabel('Claim Count')
    plt.tight_layout()
    plt.savefig('classification/data/claims_by_year.png', dpi=150)
    plt.close()

    # Phase 7: Feature encoding
    print("\n>>> PHASE 7: FEATURE ENCODING")
    mlb = MultiLabelBinarizer()
    case_types = [[ct] for ct in label_dedup['CASE_TYPE_LG']]
    encoded = mlb.fit_transform(case_types)
    encoded_df = pd.DataFrame(encoded, columns=mlb.classes_)
    
    print("\nEncoded features:")
    print(f"New columns: {list(mlb.classes_)}")
    print("Encoded data sample:")
    print(encoded_df.head())

    # Merge encoded features
    final_df = pd.concat([label_dedup.reset_index(drop=True), 
                        encoded_df.reset_index(drop=True)], axis=1)
    final_df = final_df.drop(columns='CASE_TYPE_LG')
    print("\nMerged data shape:", final_df.shape)

    # Phase 8: Data aggregation
    print("\n>>> PHASE 8: DATA AGGREGATION")
    group_cols = ['COMPANY_ID', 'ACCIDENT_YEAR']
    agg_rules = {'TOTAL_AMOUNT': 'sum'}
    for col in encoded_df.columns:
        agg_rules[col] = 'sum'
    
    grouped = final_df.groupby(group_cols).agg(agg_rules).reset_index()
    print(f"Aggregated data shape: {grouped.shape}")
    print("\nAggregated data sample:")
    print(grouped.head(3))

    # Phase 9: Cyber data merge
    print("\n>>> PHASE 9: CYBER DATA INTEGRATION")
    cyber = pd.read_csv(CYBER_DATA_PATH, low_memory=False)
    print(f"Cyber data shape: {cyber.shape}")
    
    merged = cyber.merge(
        grouped,
        left_on='CYBER_COMPANY_ID',
        right_on='COMPANY_ID',
        how='inner'
    )
    
    merged = merged.drop(columns=['TOTAL_AMOUNT'])
    print(f"Merged dataset shape: {merged.shape}")
    print(f"Unique companies in merged data: {merged['COMPANY_ID'].nunique()}")

    # Phase 10: Save results
    print("\n>>> PHASE 10: SAVING RESULTS")
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Data saved to: {OUTPUT_PATH}")
    
    # Final validation
    print("\nPipeline validation:")
    print(f"Final record count: {len(merged)}")
    print(f"Feature count: {merged.shape[1]}")
    print("Final data sample (last 8 columns):")
    print(merged.iloc[:2, -8:])

if __name__ == "__main__":
    main()