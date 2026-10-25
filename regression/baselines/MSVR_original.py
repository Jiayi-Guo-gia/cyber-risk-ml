import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

# MSVR Implementation
class MSVR(BaseEstimator, RegressorMixin):
    def __init__(self, kernel='rbf', degree=3, gamma=None, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.Beta = None
        self.NSV = None
        self.xTrain = None

    def fit(self, x, y):
         # Convert y to a NumPy array if it's a DataFrame\
        if hasattr(y, 'values'):
            y = y.to_numpy()
        self.xTrain = x.copy()
        C = self.C
        epsi = self.epsilon
        tol = self.tol

        n_m = np.shape(x)[0]  # Number of samples
        n_d = np.shape(x)[1]  # Input data dimensionality
        n_k = np.shape(y)[1]  # Output data dimensionality (output variables)

        print("Computing kernel matrix...")
        H = pairwise_kernels(x, x, metric=self.kernel, filter_params=True,
                             degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        print("Kernel matrix computed.")
        self.Beta = np.zeros((n_m, n_k))

        E = y - np.dot(H, self.Beta)
        E = np.array(E)
        u = np.sqrt(np.sum(E**2, 1, keepdims=True))

        RMSE = []
        RMSE_0 = np.sqrt(np.mean(u**2))
        RMSE.append(RMSE_0)
        print(f"Initial RMSE: {RMSE_0}")

        i1 = np.where(u > epsi)[0]
        a = 2 * C * (u - epsi) / u
        print(f"Number of points with error > epsilon: {len(i1)}")

        L = np.zeros(u.shape)
        L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2

        Lp = []
        BetaH = np.dot(np.dot(self.Beta.T, H), self.Beta)
        Lp_0 = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
        Lp.append(Lp_0)
        print(f"Initial Loss: {Lp_0}")

        eta = 1
        k = 1
        hacer = 1
        val = 1

        while(hacer):
            Beta_a = self.Beta.copy()
            E_a = E.copy()
            u_a = u.copy()
            i1_a = i1.copy()

            M1 = H[i1][:, i1] + np.diagflat(1/a[i1]) + 1e-10 * np.eye(len(a[i1]))
            print("Computing betas...")
            sal1 = np.dot(np.linalg.inv(M1), y[i1])

            eta = 1
            self.Beta = np.zeros(self.Beta.shape)
            self.Beta[i1] = sal1.copy()

            E = y - np.dot(H, self.Beta)
            u = np.sqrt(np.sum(E**2, 1)).reshape(n_m, 1)
            i1 = np.where(u >= epsi)[0]
            print(f"Number of points with error > epsilon: {len(i1)}")

            L = np.zeros(u.shape)
            L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2

            BetaH = np.dot(np.dot(self.Beta.T, H), self.Beta)
            Lp_k = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
            Lp.append(Lp_k)
            print(f"Loss at iteration {k}: {Lp_k}")

            while(Lp[k] > Lp[k-1]):
                print(f"Adjusting eta (eta = {eta})")
                eta = eta/10
                i1 = i1_a.copy()

                self.Beta = np.zeros(self.Beta.shape)
                self.Beta[i1] = eta*sal1 + (1-eta)*Beta_a[i1]

                E = y - np.dot(H, self.Beta)
                u = np.sqrt(np.sum(E**2, 1)).reshape(n_m, 1)
                i1 = np.where(u >= epsi)[0]

                L = np.zeros(u.shape)
                L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2
                BetaH = np.dot(np.dot(self.Beta.T, H), self.Beta)
                Lp_k = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
                Lp[k] = Lp_k

                if(eta < 1e-16):
                    print("Eta too small, stopping...")
                    Lp[k] = Lp[k-1] - 1e-15
                    self.Beta = Beta_a.copy()
                    u = u_a.copy()
                    i1 = i1_a.copy()
                    hacer = 0

            a_a = a.copy()
            a = 2 * C * (u - epsi) / u

            RMSE_k = np.sqrt(np.mean(u**2))
            RMSE.append(RMSE_k)
            print(f"RMSE at iteration {k}: {RMSE_k}")

            if((Lp[k-1]-Lp[k])/Lp[k-1] < tol):
                hacer = 0

            k = k + 1

            if(len(i1) == 0):
                hacer = 0
                self.Beta = np.zeros(self.Beta.shape)
                val = -1

        self.NSV = len(i1)
        print(f"Number of support vectors: {self.NSV}")

    def predict(self, x):
        H = pairwise_kernels(x, self.xTrain, metric=self.kernel, filter_params=True,
                             degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        yPred = np.dot(H, self.Beta)
        return yPred


def main():
    """
    Main function to load data, train GPU-accelerated MSVR, evaluate the model, and save results.
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
    # Save column names before converting to NumPy arrays
    Y_test_columns = Y_test.columns
    # Convert DataFrames to NumPy arrays using .values
    X_train = X_train.values
    Y_train = Y_train.values
    X_test = X_test.values
    Y_test = Y_test.values

    print("Data loaded successfully.")

    # Initialize MSVR
    print("Initializing MSVR...")
    msvr = MSVR(kernel='rbf', C=1.0, epsilon=0.1, gamma=0.1)
    print("MSVR initialized.")

    # Train the model
    print("Training MSVR model...")
    msvr.fit(X_train, Y_train)
    print("Training completed.")

    # Make predictions
    print("Making predictions...")
    Y_pred = msvr.predict(X_test)
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
    predictions_df = pd.DataFrame(Y_pred, columns=Y_test_columns)
    predictions_df.to_csv(os.path.join(results_dir, "msvr_predictions.csv"), index=False)
    print(f"Predictions saved to {os.path.join(results_dir, 'msvr_predictions.csv')}")

    # Save metrics summary to a CSV file
    metrics_df = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "R^2 Score"],
        "Value": [mse, rmse, r2]
    })
    metrics_df.to_csv(os.path.join(results_dir, "msvr_metrics.csv"), index=False)
    print(f"Metrics saved to {os.path.join(results_dir, 'msvr_metrics.csv')}")


if __name__ == "__main__":
    main()
