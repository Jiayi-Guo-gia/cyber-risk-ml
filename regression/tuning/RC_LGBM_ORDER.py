import os
import sys
import json
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from scipy.spatial import distance
from sklearn.multioutput import RegressorChain
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from optuna.samplers import TPESampler
from itertools import permutations

# Configuration updated for LGBM
CONFIG = {
    "data_path": {
        "X_train": "data/X_train.csv",
        "Y_train": "data/Y_train.csv", 
        "X_test": "data/X_test.csv",
        "Y_test": "data/Y_test.csv"
    },
    "output": {
        "model_params": "results/ORDER/LGBM_RC_best_params.json",
        "chain_order": "results/ORDER/LGBM_RC_best_chain_order.json",  
        "train_metrics": "results/ORDER/LGBM_RC_train_metrics.json",
        "test_metrics": "results/ORDER/LGBM_RC_test_metrics.json",
        "train_predictions": "results/ORDER/LGBM_RC_train_preds.csv",
        "test_predictions": "results/ORDER/LGBM_RC_test_preds.csv",
        "study_db": "sqlite:///results/ORDER/lgbm_rc_study.db"
    },
    "optimization": {
        "n_trials": 100,
        "cv_folds": 5,
        "random_state": 42,
        "metric": "MSE"
    }
}


N_LABELS = 5
ALL_PERMUTATIONS = list(permutations(range(N_LABELS)))
print(f"Total permutations: {len(ALL_PERMUTATIONS)}")

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
        
        # Validate target order and count
        label_order = datasets["Y_train"].columns.tolist()
        if len(label_order) != N_LABELS:
            raise ValueError(f"Expected {N_LABELS} output targets, got {len(label_order)}")
            
        print(f"Data loaded in {datetime.now()-start}", flush=True)
        print(f"Training shape: {datasets['X_train'].shape}", flush=True)
        print(f"Testing shape: {datasets['X_test'].shape}", flush=True)
        return datasets, label_order
        
    except Exception as e:
        print(f"\nError loading data: {str(e)}", file=sys.stderr, flush=True)
        raise

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics with safe division"""
    try:
        # Convert to numpy arrays and flatten
        y_true_flat = y_true.to_numpy().flatten()
        y_pred_flat = np.array(y_pred).flatten()

        metrics = {
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "CC": pearsonr(y_true_flat, y_pred_flat)[0],  # Use converted arrays
        }

        y_std = np.std(y_true_flat)                          
        metrics.update({
            "RRMSE": metrics["RMSE"] / (y_std) if y_std != 0 else float('inf'),           
        })

        return metrics
        
    except Exception as e:
        print(f"\nError calculating metrics: {str(e)}", file=sys.stderr, flush=True)
        raise

def objective(trial):
    """Optuna optimization objective with enhanced logging"""
    trial_id = trial.number + 1
    trial_start = datetime.now()
    
    print(f"\n{'='*60}", flush=True)
    print(f"Starting Trial {trial_id}/{CONFIG['optimization']['n_trials']}", flush=True)
    
    try:
        data, label_order = load_data()
        X, Y = data["X_train"], data["Y_train"]
        
        # LightGBM hyperparameter space 
        params = {
            'objective': 'regression',
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'num_leaves': trial.suggest_int('num_leaves', 100, 180, step=10),
            'max_depth': trial.suggest_int('max_depth', 8, 16),
            'learning_rate': trial.suggest_float('learning_rate', 0.002, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0, step=0.05),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9, step=0.05),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 60, 100),
            'random_state': CONFIG["optimization"]["random_state"],
            'n_jobs': -1,
            'verbosity': -1
        }

        
  
        perm_idx = trial.suggest_int('permutation_index', 0, len(ALL_PERMUTATIONS)-1)
        chain_order = list(ALL_PERMUTATIONS[perm_idx])
        print(f"Chain Order: {chain_order}")

        kf = KFold(n_splits=CONFIG["optimization"]["cv_folds"])
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            fold_start = datetime.now()
            print(f"\nProcessing Fold {fold}/{CONFIG['optimization']['cv_folds']}", flush=True)
            
            try:
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                Y_train_fold, Y_val_fold = Y.iloc[train_idx], Y.iloc[val_idx]

                # Initialize model with optimized chain order
                base_model = lgb.LGBMRegressor(**params)
                model = RegressorChain(
                    base_model,
                    order=chain_order,
                    random_state=CONFIG["optimization"]["random_state"]
                )
                
                model.fit(X_train_fold, Y_train_fold)
                val_preds = model.predict(X_val_fold)
                
                # Calculate per-label MSE and average
                label_mse = [
                    mean_squared_error(Y_val_fold[label], val_preds[:, idx])
                    for idx, label in enumerate(label_order)
                ]
                fold_score = np.mean(label_mse)
                fold_scores.append(fold_score)
                
                fold_duration = datetime.now() - fold_start
                print(f"Fold {fold} Avg MSE: {fold_score:.4f}", flush=True)
                print(f"Fold Duration: {fold_duration}", flush=True)

            except Exception as e:
                print(f"\nFold {fold} failed: {str(e)}", file=sys.stderr, flush=True)
                raise

        trial_duration = datetime.now() - trial_start
        avg_score = np.mean(fold_scores)
        
        print(f"\nTrial {trial_id} Completed in {trial_duration}", flush=True)
        print(f"Average MSE: {avg_score:.4f}", flush=True)
        print(f"{'='*60}", flush=True)
        return avg_score

    except Exception as e:
        print(f"\nTrial failed: {str(e)}", file=sys.stderr, flush=True)
        raise optuna.exceptions.TrialPruned()

def evaluate_model(best_params, best_chain_order, X_train, Y_train, X_test, Y_test, label_order):
    """Final model evaluation with best parameters"""
    print("\nEvaluating final model...", flush=True)
    start = datetime.now()
    
    try:
        # Initialize best model with best chain order
        base_model = lgb.LGBMRegressor(**best_params)
        model = RegressorChain(
            base_model,
            order=best_chain_order,
            random_state=CONFIG["optimization"]["random_state"]
        )
        
        model.fit(X_train, Y_train)
        
        predictions = {
            "train": model.predict(X_train),
            "test": model.predict(X_test)
        }
        
        metrics = {"train": {"labels": {}, "average": {}}, "test": {"labels": {}, "average": {}}}
        
        for split in ["train", "test"]:
            true_labels = Y_train if split == "train" else Y_test
            for idx, label in enumerate(label_order):
                metrics[split]["labels"][label] = calculate_metrics(
                    true_labels[label], predictions[split][:, idx]
                )
            
            # Calculate average metrics
            metrics[split]["average"] = {
                metric: np.mean([m[metric] for m in metrics[split]["labels"].values()])
                for metric in ["MSE", "RMSE", "MAE", "CC", "RRMSE"]
            }
            eu_dist = distance.cdist(true_labels, predictions[split], 'euclidean').diagonal().mean()
            metrics[split]["average"]["EU_DIST"] = eu_dist
        
        # Save predictions with correct column order
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
        
        # Create output directories
        os.makedirs(os.path.dirname(CONFIG["output"]["model_params"]), exist_ok=True)
        db_path = CONFIG["output"]["study_db"].replace("sqlite:///", "")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        data, label_order = load_data()

        # Study initialization with resume support
        try:
            # Try to load existing study
            study = optuna.load_study(
                study_name="RC_LGBM_Optimization",
                storage=CONFIG["output"]["study_db"]
            )
            print(f"Resuming existing study with {len(study.trials)} completed trials", flush=True)
        except (KeyError, ValueError):
            # Create new study if not exists
            study = optuna.create_study(
                direction="minimize",
                sampler=TPESampler(seed=CONFIG["optimization"]["random_state"]),
                storage=CONFIG["output"]["study_db"],
                study_name="RC_LGBM_Optimization"
            )
            print("Initializing new study", flush=True)
        
        # Calculate remaining trials
        remaining_trials = CONFIG["optimization"]["n_trials"] - len(study.trials)
        if remaining_trials > 0:
            print(f"Running {remaining_trials} additional trials", flush=True)
            study.optimize(objective, n_trials=remaining_trials)
        else:
            print("Optimization already completed - loading best trial", flush=True)
        
        # Save best parameters
        with open(CONFIG["output"]["model_params"], "w") as f:
            json.dump(study.best_params, f, indent=2)
            
        # Get and save the best chain order
        perm_idx = study.best_params['permutation_index']
        best_chain_order = list(ALL_PERMUTATIONS[perm_idx])
        with open(CONFIG["output"]["chain_order"], "w") as f:
            json.dump(best_chain_order, f, indent=2)
        print(f"Best chain order: {best_chain_order}")
            
        # Final evaluation with best chain order
        metrics = evaluate_model(
            {k: v for k, v in study.best_params.items() if k != 'permutation_index'},
            best_chain_order,
            data["X_train"], data["Y_train"],
            data["X_test"], data["Y_test"],
            label_order
        )
        
        # Save metrics
        with open(CONFIG["output"]["train_metrics"], "w") as f:
            json.dump(metrics["train"], f, indent=2)
        with open(CONFIG["output"]["test_metrics"], "w") as f:
            json.dump(metrics["test"], f, indent=2)
            
        print("\nOptimization Complete")
        print(f"Best Trial Value (Avg MSE): {study.best_value:.4f}")
        print(f"Best Chain Order: {best_chain_order}")

    except Exception as e:
        print(f"\nCritical error: {str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()