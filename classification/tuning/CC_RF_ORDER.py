import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import optuna
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    jaccard_score,
    hamming_loss,
    f1_score
)
from sklearn.multioutput import ClassifierChain
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from optuna.samplers import TPESampler
from itertools import permutations

# Configure logging
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

CONFIG = {
    "data_path": {
        "X_train": "data/X_train.csv",
        "Y_train": "data/Y_train.csv",
        "X_test": "data/X_test.csv",
        "Y_test": "data/Y_test.csv"
    },
    "output": {
        "model_params": "results/ORDER/RF_CC_best_model_params.json",
        "thresholds": "results/ORDER/RF_CC_best_thresholds.json",
        "chain_order": "results/ORDER/RF_CC_best_chain_order.json",  # path for saving the chain order
        "train_metrics": "results/ORDER/RF_CC_train_metrics.json",
        "train_report": "results/ORDER/RF_CC_train_classification_report.txt",
        "train_predictions": "results/ORDER/RF_CC_train_predictions.csv",
        "test_metrics": "results/ORDER/RF_CC_test_metrics.json",
        "test_report": "results/ORDER/RF_CC_test_classification_report.txt",
        "test_predictions": "results/ORDER/RF_CC_test_predictions.csv",
        "study_db": "sqlite:///results/ORDER/rf_cc_optuna_study.db"
    },
    "optimization": {
        "n_trials": 100,
        "cv_folds": 5,
        "random_state": 42,
        "metric": "weighted"
    },
    "label_order": [
        'Data Breach',
        'Privacy Violation', 
        'Extortion/Fraud',
        'IT Error',
        'Other',
    ]
}

# Precompute all possible permutations
ALL_PERMUTATIONS = list(permutations(range(len(CONFIG["label_order"]))))
print(f"Total permutations: {len(ALL_PERMUTATIONS)}")

def load_data():
    print("\nLoading data...")
    start = datetime.now()
    datasets = {
        "X_train": pd.read_csv(CONFIG["data_path"]["X_train"]),
        "Y_train": pd.read_csv(CONFIG["data_path"]["Y_train"])[CONFIG["label_order"]],
        "X_test": pd.read_csv(CONFIG["data_path"]["X_test"]),
        "Y_test": pd.read_csv(CONFIG["data_path"]["Y_test"])[CONFIG["label_order"]]
    }
    
    assert list(datasets["Y_train"].columns) == CONFIG["label_order"], "Label mismatch in Y_train"
    assert list(datasets["Y_test"].columns) == CONFIG["label_order"], "Label mismatch in Y_test"
    
    print(f"Data loaded in {datetime.now() - start}")
    print(f"Training shape: {datasets['X_train'].shape}, Test shape: {datasets['X_test'].shape}")
    return datasets

def objective(trial):
    start_time = datetime.now()
    print(f"\nStarting Trial {trial.number+1}/{CONFIG['optimization']['n_trials']}")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    data = load_data()
    X, Y = data["X_train"], data["Y_train"]
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': CONFIG["optimization"]["random_state"],
        'n_jobs': -1,
        'verbose': 0
    }
    
    thresholds = [
        trial.suggest_float(f'threshold_{i}', 0.2, 0.8)
        for i in range(Y.shape[1])
    ]
    
    # Directly optimize the permutation of label order
    perm_idx = trial.suggest_int('permutation_index', 0, len(ALL_PERMUTATIONS)-1)
    chain_order = list(ALL_PERMUTATIONS[perm_idx])
    print(f"Chain Order: {chain_order}")
    
    mskf = MultilabelStratifiedKFold(
        n_splits=CONFIG["optimization"]["cv_folds"],
        shuffle=True,
        random_state=CONFIG["optimization"]["random_state"]
    )
    
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(mskf.split(X, Y)):
        print(f"\nProcessing Fold {fold+1}/{CONFIG['optimization']['cv_folds']}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = Y.iloc[train_idx], Y.iloc[val_idx]

        model = ClassifierChain(
            RandomForestClassifier(**params),
            order=chain_order,  # use the optimized order
            random_state=CONFIG["optimization"]["random_state"]
        )
        model.fit(X_train, y_train)
        
        val_probas = model.predict_proba(X_val)
        val_preds = (val_probas >= thresholds).astype(int)
        
        fold_score = f1_score(y_val, val_preds, average=CONFIG["optimization"]["metric"])
        cv_scores.append(fold_score)
        print(f"Fold {fold+1} Score: {fold_score:.4f}")

    mean_score = np.mean(cv_scores)
    print(f"\nTrial {trial.number} Completed in {datetime.now() - start_time}")
    print(f"Average Score: {mean_score:.4f}")
    return mean_score

def evaluate_configuration(params, thresholds, chain_order):
    print("\nStarting Final Evaluation")
    eval_start = datetime.now()
    data = load_data()
    
    print(f"Using chain order: {chain_order}")
    
    # ===== Train Set Evaluation =====
    print("\nEvaluating on Training Set...")
    train_start = datetime.now()
    
    # Train final model
    train_model = ClassifierChain(
        RandomForestClassifier(**params),
        order=chain_order,  # use the optimized order
        random_state=CONFIG["optimization"]["random_state"]
    )
    train_model.fit(data["X_train"], data["Y_train"])
    
    # Get training predictions
    train_probas = train_model.predict_proba(data["X_train"])
    train_preds = (train_probas >= thresholds).astype(int)
    
    # Save training predictions
    pd.DataFrame(train_preds, columns=CONFIG["label_order"]).to_csv(
        CONFIG["output"]["train_predictions"], index=False
    )
    
    # Generate training report
    train_report = classification_report(
        data["Y_train"], train_preds,
        target_names=CONFIG["label_order"],
        digits=4
    )
    
    # Save training report
    with open(CONFIG["output"]["train_report"], 'w') as f:
        f.write("Training Set Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(train_report)
    
    # Calculate training metrics
    train_metrics = {
        "weighted_f1": f1_score(data["Y_train"], train_preds, average='weighted'),
        "macro_f1": f1_score(data["Y_train"], train_preds, average='macro'),
        "micro_f1": f1_score(data["Y_train"], train_preds, average='micro'),
        "sample_f1": f1_score(data["Y_train"], train_preds, average='samples'),
        "jaccard": jaccard_score(data["Y_train"], train_preds, average='samples'),
        "hamming_loss": hamming_loss(data["Y_train"], train_preds)
    }
    
    # Print training results
    print("\nTraining Classification Report:")
    print(train_report)
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric:<15}: {value:.4f}")
    print(f"Training evaluation time: {datetime.now() - train_start}")

    # ===== Test Set Evaluation =====
    print("\nEvaluating on Test Set...")
    test_start = datetime.now()
    
    # Get test predictions (using same model)
    test_probas = train_model.predict_proba(data["X_test"])
    test_preds = (test_probas >= thresholds).astype(int)
    
    # Save test predictions
    pd.DataFrame(test_preds, columns=CONFIG["label_order"]).to_csv(
        CONFIG["output"]["test_predictions"], index=False
    )
    
    # Generate test report
    test_report = classification_report(
        data["Y_test"], test_preds,
        target_names=CONFIG["label_order"],
        digits=4
    )
    
    # Save test report
    with open(CONFIG["output"]["test_report"], 'w') as f:
        f.write("Test Set Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(test_report)
    
    # Calculate test metrics
    test_metrics = {
        "weighted_f1": f1_score(data["Y_test"], test_preds, average='weighted'),
        "macro_f1": f1_score(data["Y_test"], test_preds, average='macro'),
        "micro_f1": f1_score(data["Y_test"], test_preds, average='micro'),
        "sample_f1": f1_score(data["Y_test"], test_preds, average='samples'),
        "jaccard": jaccard_score(data["Y_test"], test_preds, average='samples'),
        "hamming_loss": hamming_loss(data["Y_test"], test_preds)
    }
    
    # Print test results
    print("\nTest Classification Report:")
    print(test_report)
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric:<15}: {value:.4f}")
    print(f"Test evaluation time: {datetime.now() - test_start}")

    # Save all metrics
    metrics = {"train": train_metrics, "test": test_metrics}
    with open(CONFIG["output"]["train_metrics"], 'w') as f:
        json.dump(train_metrics, f, indent=2)
    with open(CONFIG["output"]["test_metrics"], 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\nTotal evaluation time: {datetime.now() - eval_start}")
    return metrics

def main():
    total_start = datetime.now()
    print("\nStarting Optimization Process")
    print(f"Target Metric: {CONFIG['optimization']['metric']}")
    print(f"Total Trials: {CONFIG['optimization']['n_trials']}")
    
    # Create all output directories
    output_dirs = set()
    for key, path in CONFIG["output"].items():
        if key == "study_db":
            # Handle database path
            storage_path = path.replace("sqlite:///", "")
            output_dirs.add(os.path.dirname(storage_path))
        else:
            # Handle file paths
            output_dirs.add(os.path.dirname(path))
    
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Existing database-loading logic unchanged...
    storage_path = CONFIG["output"]["study_db"].replace("sqlite:///", "")
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    
    try:
        study = optuna.load_study(
            study_name="RF_CC_Optimization",
            storage=CONFIG["output"]["study_db"],
            sampler=TPESampler(seed=CONFIG["optimization"]["random_state"])
        )
        print(f"Resuming existing study with {len(study.trials)} trials")
    except:
        study = optuna.create_study(
            study_name="RF_CC_Optimization",
            storage=CONFIG["output"]["study_db"],
            direction="maximize",
            sampler=TPESampler(seed=CONFIG["optimization"]["random_state"])
        )
        print("Initializing new study")


    # Compute remaining number of trials
    n_trials_done = len(study.trials)
    n_trials_total = CONFIG["optimization"]["n_trials"]
    remaining_trials = max(0, n_trials_total - n_trials_done)
    if remaining_trials > 0:
        print(f"\nRunning {remaining_trials} additional trials...")
        study.optimize(objective, n_trials=remaining_trials, catch=(Exception,))
    else:
        print(f"\nNo additional trials needed. Already completed {n_trials_done} trials.")
    
    best_params = {k: v for k, v in study.best_params.items() 
                  if not (k.startswith('threshold_') or k == 'permutation_index')}
    
    thresholds = [study.best_params[f'threshold_{i}'] 
                 for i in range(len(CONFIG["label_order"]))]
    
    # Get the best label order
    perm_idx = study.best_params['permutation_index']
    best_chain_order = list(ALL_PERMUTATIONS[perm_idx])
    
    # Save the best order
    with open(CONFIG["output"]["chain_order"], 'w') as f:
        json.dump(best_chain_order, f, indent=2)
    
    with open(CONFIG["output"]["model_params"], 'w') as f:
        json.dump(best_params, f, indent=2)
    with open(CONFIG["output"]["thresholds"], 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    metrics = evaluate_configuration(best_params, thresholds, best_chain_order)
    
    total_time = datetime.now() - total_start
    print("\nOptimization Complete")
    print(f"Total runtime: {total_time}")
    print(f"Results saved to: {os.path.dirname(CONFIG['output']['model_params'])}")
    print("\nPerformance Summary:")
    print(f"Training Weighted F1: {metrics['train']['weighted_f1']:.4f}")
    print(f"Test Weighted F1:     {metrics['test']['weighted_f1']:.4f}")
    print(f"\nAverage time per trial: {total_time / CONFIG['optimization']['n_trials']}")

if __name__ == "__main__":
    main()