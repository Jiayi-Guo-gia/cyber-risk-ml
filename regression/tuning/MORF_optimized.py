import os
import sys
import json
import numpy as np
import pandas as pd
import optuna
from scipy.spatial import distance
from sklearn.ensemble import RandomForestRegressor 
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from optuna.samplers import TPESampler

# Configuration updated for RF
CONFIG = {
    "data_path": {
        "X_train": "data/X_train.csv",
        "Y_train": "data/Y_train.csv", 
        "X_test": "data/X_test.csv",
        "Y_test": "data/Y_test.csv"
    },
    "output": {
        "model_params": "results/RF/RF_best_params.json",
        "train_metrics": "results/RF/RF_train_metrics.json",
        "test_metrics": "results/RF/RF_test_metrics.json",
        "train_predictions": "results/RF/RF_train_preds.csv",
        "test_predictions": "results/RF/RF_test_preds.csv",
        "study_db": "sqlite:///results/RF/rf_study.db"
    },
    "optimization": {
        "n_trials": 100,
        "cv_folds": 5,
        "random_state": 42,
        "metric": "MSE"
    }
}

def load_data():
    """Load and validate datasets with error handling"""
    print("\nLoading data...", flush=True)
    start = datetime.now()
    
    try:
        datasets = {
            "X_train": pd.read_csv(CONFIG["data_path"]["X_train"]),
            "Y_train": pd.read_csv(CONFIG["data_path"]["Y_train"]),
            "X_test": pd.read_csv(CONFIG["data_path"]["X_test"]),
            "Y_test": pd.read_csv(CONFIG["data_path"]["Y_test"])
        }
        
        label_order = datasets["Y_train"].columns.tolist()
        
        print(f"Data loaded in {datetime.now()-start}", flush=True)
        print(f"Training shape: {datasets['X_train'].shape}", flush=True)
        print(f"Testing shape: {datasets['X_test'].shape}", flush=True)
        return datasets, label_order
        
    except Exception as e:
        print(f"\nError loading data: {str(e)}", file=sys.stderr, flush=True)
        raise

def calculate_metrics(y_true, y_pred, label_order):
    """Calculate regression metrics with label names and global Euclidean distance"""
    try:
        # Convert to numpy arrays if needed
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate per-label metrics
        metrics = {
            "MSE": mean_squared_error(y_true, y_pred, multioutput='raw_values'),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values')),
            "MAE": mean_absolute_error(y_true, y_pred, multioutput='raw_values'),
            "CC": [pearsonr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])]
        }

        # Calculate RRMSE
        y_std = np.std(y_true, axis=0)
        metrics["RRMSE"] = metrics["RMSE"] / y_std

        # Calculate global Euclidean distance
        distance_matrix = distance.cdist(y_true, y_pred, 'euclidean')
        eu_dist_global = distance_matrix.diagonal().mean()

        # Convert to label-based dictionary
        per_label_metrics = {
            label: {
                "MSE": float(metrics["MSE"][i]),
                "RMSE": float(metrics["RMSE"][i]),
                "MAE": float(metrics["MAE"][i]),
                "CC": float(metrics["CC"][i]),
                "RRMSE": float(metrics["RRMSE"][i])
            }
            for i, label in enumerate(label_order)
        }

        # Calculate averages
        avg_metrics = {
            "MSE": float(np.mean(metrics["MSE"])),
            "RMSE": float(np.mean(metrics["RMSE"])),
            "MAE": float(np.mean(metrics["MAE"])),
            "CC": float(np.mean(metrics["CC"])),
            "RRMSE": float(np.mean(metrics["RRMSE"])),
            "EU_DIST": float(eu_dist_global)
        }

        return {
            "per_label": per_label_metrics,
            "average": avg_metrics
        }

    except Exception as e:
        print(f"\nError calculating metrics: {str(e)}", file=sys.stderr, flush=True)
        raise

def objective(trial):
    """Optuna optimization objective with enhanced logging"""
    trial_id = trial.number + 1
    trial_start = datetime.now()
    
    print(f"\n{'='*60}", flush=True)
    print(f"Starting Trial {trial_id}/{CONFIG['optimization']['n_trials']}", flush=True)
    print(f"Start Time: {trial_start.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Trial ID: {trial.number}", flush=True)
    
    try:
        data, label_order = load_data()
        X, Y = data["X_train"], data["Y_train"]
       # Random Forest hyperparameter space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=100),
            'max_depth': trial.suggest_int('max_depth', 4, 12), 
            'min_samples_split': trial.suggest_int('min_samples_split', 10, 50), 
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 30),  
            'max_features': trial.suggest_float('max_features', 0.4, 0.8),
            'bootstrap': True,
            'random_state': CONFIG["optimization"]["random_state"],  
            'n_jobs': -1
        }

        kf = KFold(n_splits=CONFIG["optimization"]['cv_folds'])
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            fold_start = datetime.now()
            print(f"\nProcessing Fold {fold}/{CONFIG['optimization']['cv_folds']}", flush=True)
            
            try:
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                Y_train_fold, Y_val_fold = Y.iloc[train_idx], Y.iloc[val_idx]

                # Directly use RandomForestRegressor
                model = RandomForestRegressor(**params)
                model.fit(X_train_fold, Y_train_fold)
                val_preds = model.predict(X_val_fold)
                
                fold_score = mean_squared_error(Y_val_fold, val_preds)
                fold_scores.append(fold_score)
                
                fold_duration = datetime.now() - fold_start
                print(f"Fold {fold} MSE: {fold_score:.4f}", flush=True)
                print(f"Fold Duration: {fold_duration}", flush=True)

            except Exception as e:
                print(f"\nFold {fold} failed: {str(e)}", file=sys.stderr, flush=True)
                raise

        trial_duration = datetime.now() - trial_start
        avg_score = np.mean(fold_scores)
        
        print(f"\nTrial {trial_id} Completed in {trial_duration}", flush=True)
        print(f"Average MSE: {avg_score:.4f}", flush=True)
        
        try:
            print(f"Best Value Until Now: {trial.study.best_value:.4f}", flush=True)
        except ValueError:
            print("No valid trials completed yet", flush=True)
        
        print(f"{'='*60}", flush=True)
        return avg_score

    except Exception as e:
        print(f"\nTrial {trial_id} failed: {str(e)}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise optuna.exceptions.TrialPruned()

def evaluate_model(params, X_train, Y_train, X_test, Y_test, label_order):
    """Final model evaluation with error handling"""
    print("\nEvaluating final model...", flush=True)
    start = datetime.now()
    
    try:
        model = RandomForestRegressor(**params)
        model.fit(X_train, Y_train)
        
        predictions = {
            "train": model.predict(X_train),
            "test": model.predict(X_test)
        }
        
        metrics = {
            "train": calculate_metrics(Y_train.values, predictions["train"], label_order),
            "test": calculate_metrics(Y_test.values, predictions["test"], label_order)
        }
        
        for split in ["train", "test"]:
            pd.DataFrame(predictions[split], columns=label_order).to_csv(
                CONFIG["output"][f"{split}_predictions"], index=False
            )
        
        print(f"Evaluation completed in {datetime.now()-start}", flush=True)
        return metrics
        
    except Exception as e:
        print(f"\nEvaluation failed: {str(e)}", file=sys.stderr, flush=True)
        raise

def main():
    """Main optimization workflow with comprehensive error handling"""
    try:
        optuna.logging.set_verbosity(optuna.logging.INFO)
        print("\nStarting Optimization Process", flush=True)
        print(f"Target Metric: {CONFIG['optimization']['metric']}", flush=True)
        print(f"Total Trials: {CONFIG['optimization']['n_trials']}", flush=True)
        
        # Create output directories
        os.makedirs(os.path.dirname(CONFIG["output"]["model_params"]), exist_ok=True)
        db_path = CONFIG["output"]["study_db"].replace("sqlite:///", "")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        data, label_order = load_data()
        
        # Study initialization
        try:
            study = optuna.load_study(
                study_name="RF_Optimization",
                storage=CONFIG["output"]["study_db"]
            )
            print(f"Loaded existing study: {study.study_name}", flush=True)
        except:
            study = optuna.create_study(
                direction="minimize",
                storage=CONFIG["output"]["study_db"],
                sampler=TPESampler(seed=CONFIG["optimization"]['random_state']),
                study_name="RF_Optimization"
            )
            print(f"Created new study: {study.study_name}", flush=True)
        
        print(f"\nDatabase URL: {CONFIG['output']['study_db']}", flush=True)
        print("Initializing optimization...\n", flush=True)
        
        # Calculate remaining trials
        remaining_trials = CONFIG["optimization"]["n_trials"] - len(study.trials)
        if remaining_trials > 0:
            print(f"Running {remaining_trials} additional trials", flush=True)
            study.optimize(objective, n_trials=remaining_trials)
        else:
            print("Optimization already completed - loading best trial", flush=True)
        
        # Save results
        best_params = study.best_params
        with open(CONFIG["output"]["model_params"], "w") as f:
            json.dump(best_params, f, indent=2)
        
        metrics = evaluate_model(
            best_params,
            data["X_train"], data["Y_train"],
            data["X_test"], data["Y_test"],
            label_order
        )
        
        for split in ["train", "test"]:
            with open(CONFIG["output"][f"{split}_metrics"], "w") as f:
                json.dump(metrics[split], f, indent=2)
        
        print("\nOptimization Complete", flush=True)
        print(f"Best Trial Value (Avg MSE): {study.best_value:.4f}", flush=True)
        print(f"Best Parameters Saved: {CONFIG['output']['model_params']}", flush=True)
        
    except Exception as e:
        print(f"\nCritical error in main: {str(e)}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()