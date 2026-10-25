import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess_data():
    # Define input and output paths
    input_path = "classification/data/label_cyber_year.csv"
    output_dir = "classification/data"
    test_size = 0.2  # Proportion of the dataset to allocate for testing
    random_state = 42  # Seed for reproducibility

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load raw data
    print("Loading raw data...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Raw data dimensions: {df.shape}")

    # 2. Data Cleaning
    print("\nPerforming data cleaning:")
    # Remove unnecessary columns
    df = df.drop(columns=["CARPE_ID", "CYBER_COMPANY_ID"])
    print(f"Data dimensions after cleaning: {df.shape}")

    # 3. Feature Engineering
    print("\nPerforming feature transformation:")
    # Define grouping and target columns
    group_cols = ['COMPANY_ID', 'ACCIDENT_YEAR', 'TOTAL_AMOUNT']
    category_cols = [col for col in df.columns if col not in group_cols]
    
    # Convert categorical features to binary (0 or 1)
    df[category_cols] = np.where(df[category_cols] > 0, 1, 0)
    print("Categorical feature binarization completed")

    # 4. Split Features and Targets
    print("\nSplitting features and labels:")
    df = df.drop(columns='COMPANY_ID')  # Remove COMPANY_ID as it's not a feature
    target_cols = ['Data Breach', 'Privacy Violation', 
                   'Extortion/Fraud', 'IT Error', 'Other']
    features = [col for col in df.columns if col not in target_cols]
    
    X = df[features]  # Feature matrix
    y = df[target_cols]  # Target labels

    # 5. Train-Test Split
    print(f"\nSplitting dataset (Test set proportion: {test_size})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state
    )

    # 6. Save Processed Data
    print("\nSaving processed data:")
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "Y_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "Y_test.csv"), index=False)
    
    print("\nData preprocessing completed!")
    print(f"Training set dimensions: {X_train.shape}")
    print(f"Test set dimensions: {X_test.shape}")

if __name__ == "__main__":
    preprocess_data()
