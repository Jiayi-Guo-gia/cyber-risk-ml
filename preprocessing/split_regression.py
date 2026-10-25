import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data(
    input_path="regression/data/label_cyber_year.csv",
    output_dir="regression/data",
    test_size=0.2,
    random_state=42
):
    """
    Preprocess data for multi-label classification, perform data checks,
    split into train/test sets, and save results to CSV files.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_dir (str): Directory to save processed datasets
    test_size (float): Proportion of data to use for testing (default: 0.2)
    random_state (int): Random seed for reproducibility (default: 42)
    """
    
    # Load the dataset
    df = pd.read_csv(input_path)
    
    # Initial data checks
    print("=== Initial Dataset Information ===")
    print(f"Shape: {df.shape}")
    print("\nMissing values per column (before preprocessing):")
    print(df.isna().sum())
    print("\nData types and non-null counts:")
    df.info()
    
    # Create working copy and drop columns
    df1 = df.copy()
    columns_to_drop = ["CARPE_ID", "CYBER_COMPANY_ID", "COMPANY_ID"]
    df1 = df1.drop(columns=[col for col in columns_to_drop if col in df1.columns])
    
    # Post-drop checks
    print("\n\n=== After Column Removal ===")
    print(f"New shape: {df1.shape}")
    print("\nRemaining missing values:")
    print(df1.isna().sum())
    
    # Handle missing values
    df1 = df1.fillna(0)
    print("\n\n=== After Missing Value Handling ===")
    print("Total remaining missing values:", df1.isna().sum().sum())
    
    # Show final data sample
    print("\nFinal Data Sample (last 8 columns):")
    print(df1.iloc[:, -8:].head(1))  # Display first row of last 8 columns
    
    # Define features and targets
    category_cols = ['Data Breach', 'Privacy Violation', 
                    'Extortion/Fraud', 'IT Error', 'Other']
    features = [col for col in df1.columns if col not in category_cols]
    
    X = df1[features]
    y = df1[category_cols]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Show split shapes
    print("\n\n=== Dataset Split Shapes ===")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "Y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "Y_test.csv"), index=False)
    
    print("\n=== Preprocessing Summary ===")
    print(f"Processed data saved to: {output_dir}")
    print("Final feature columns:", len(features))
    print("Target columns:", category_cols)

# Example usage
if __name__ == "__main__":
    preprocess_data()