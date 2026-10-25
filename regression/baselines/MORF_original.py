import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    """
    Main function to load data, train Multi-Output Random Forest Regression, evaluate the model, and save results.
    """
    # Ensure the results directory exists
    results_dir = "results/"
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    X_train = pd.read_csv("data/X_train.csv")
    Y_train = pd.read_csv("data/Y_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    Y_test = pd.read_csv("data/Y_test.csv")
    print("Data loaded successfully.")

    # Initialize Multi-Output Random Forest Regression
    print("Initializing Multi-Output Random Forest Regression...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)  # Using multiple jobs for speed
    morf = MultiOutputRegressor(rf)
    print("Multi-Output Random Forest Regression initialized.")

    # Train the model
    print("Training Multi-Output Random Forest Regression model...")
    morf.fit(X_train, Y_train)
    print("Training completed.")

    # Make predictions
    print("Making predictions...")
    Y_pred = morf.predict(X_test)
    print("Predictions completed.")

    # Evaluate the model
    print("Evaluating model...")
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, Y_pred)
    print("Evaluation completed.")

    # Print final evaluation metrics
    print("\n=== Final Evaluation Metrics ===")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame(Y_pred, columns=Y_test.columns)
    predictions_df.to_csv(os.path.join(results_dir, "morf_predictions.csv"), index=False)
    print(f"Predictions saved to {os.path.join(results_dir, 'morf_predictions.csv')}")

    # Save metrics summary to a CSV file
    metrics_df = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "R^2 Score"],
        "Value": [mse, rmse, r2]
    })
    metrics_df.to_csv(os.path.join(results_dir, "morf_metrics.csv"), index=False)
    print(f"Metrics saved to {os.path.join(results_dir, 'morf_metrics.csv')}")

