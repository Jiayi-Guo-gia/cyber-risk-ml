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
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from optuna.samplers import TPESampler

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
        "model_params": "results/MOC/RF_best_model_params.json",
        "thresholds": "results/MOC/RF_best_thresholds.json",
        "train_metrics": "results/MOC/RF_train_metrics.json",
        "train_report": "results/MOC/RF_train_classification_report.txt",
        "train_predictions": "results/MOC/RF_train_predictions.csv",
        "test_metrics": "results/MOC/RF_test_metrics.json",
        "test_report": "results/MOC/RF_test_classification_report.txt",
        "test_predictions": "results/MOC/RF_test_predictions.csv",
        "study_db": "sqlite:///results/MOC/rf_optuna_study.db"
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

        model = RandomForestClassifier(**params)  # Directly use RF
        model.fit(X_train, y_train)
        
        # Get probabilities for each label
        val_probas = np.array([proba[:, 1] for proba in model.predict_proba(X_val)]).T
        val_preds = (val_probas >= thresholds).astype(int)
        
        fold_score = f1_score(y_val, val_preds, average=CONFIG["optimization"]["metric"])
        cv_scores.append(fold_score)
        print(f"Fold {fold+1} Score: {fold_score:.4f}")

    mean_score = np.mean(cv_scores)
    print(f"\nTrial {trial.number} Completed in {datetime.now() - start_time}")
    print(f"Average Score: {mean_score:.4f}")
    return mean_score

def evaluate_configuration(params, thresholds):
    print("\nStarting Final Evaluation")
    eval_start = datetime.now()
    data = load_data()
   
    os.makedirs(os.path.dirname(CONFIG["output"]["train_metrics"]), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["output"]["test_metrics"]), exist_ok=True)
    

    print("\nEvaluating on Training Set...")
    train_start = datetime.now()

    train_model = RandomForestClassifier(**params)
    train_model.fit(data["X_train"], data["Y_train"])
    

    train_probas = np.array([proba[:, 1] for proba in train_model.predict_proba(data["X_train"])]).T
    train_preds = (train_probas >= thresholds).astype(int)
    

    pd.DataFrame(train_preds, columns=CONFIG["label_order"]).to_csv(
        CONFIG["output"]["train_predictions"], index=False
    )
    

    train_report = classification_report(
        data["Y_train"], train_preds,
        target_names=CONFIG["label_order"],
        digits=4,
        output_dict=True,  
        zero_division=0    
    )

    with open(CONFIG["output"]["train_report"], 'w') as f:
        f.write("Training Set Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(classification_report(
            data["Y_train"], train_preds,
            target_names=CONFIG["label_order"],
            digits=4,
            zero_division=0
        ))
    
 
    label_metrics = {}
    for label in CONFIG["label_order"]:
        label_metrics[label] = {
            "precision": train_report[label]["precision"],
            "recall": train_report[label]["recall"],
            "f1-score": train_report[label]["f1-score"],
            "support": train_report[label]["support"]
        }
    

    train_metrics = {
        "label_metrics": label_metrics,
        "weighted_f1": train_report["weighted avg"]["f1-score"],
        "macro_f1": train_report["macro avg"]["f1-score"],
        "micro_f1": f1_score(data["Y_train"], train_preds, average='micro'),
        "sample_f1": f1_score(data["Y_train"], train_preds, average='samples'),
        "jaccard": jaccard_score(data["Y_train"], train_preds, average='samples'),
        "hamming_loss": hamming_loss(data["Y_train"], train_preds)
    }
    

    print("\nTraining Classification Report:")
    print(classification_report(
        data["Y_train"], train_preds,
        target_names=CONFIG["label_order"],
        digits=4,
        zero_division=0
    ))
    

    print("\nEvaluating on Test Set...")
    test_start = datetime.now()

    test_probas = np.array([proba[:, 1] for proba in train_model.predict_proba(data["X_test"])]).T
    test_preds = (test_probas >= thresholds).astype(int)

    pd.DataFrame(test_preds, columns=CONFIG["label_order"]).to_csv(
        CONFIG["output"]["test_predictions"], index=False
    )

    test_report = classification_report(
        data["Y_test"], test_preds,
        target_names=CONFIG["label_order"],
        digits=4,
        output_dict=True,  # return dict format
        zero_division=0    # handle undefined metrics
    )

    with open(CONFIG["output"]["test_report"], 'w') as f:
        f.write("Test Set Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(classification_report(
            data["Y_test"], test_preds,
            target_names=CONFIG["label_order"],
            digits=4,
            zero_division=0
        ))
    

    test_label_metrics = {}
    for label in CONFIG["label_order"]:
        test_label_metrics[label] = {
            "precision": test_report[label]["precision"],
            "recall": test_report[label]["recall"],
            "f1-score": test_report[label]["f1-score"],
            "support": test_report[label]["support"]
        }
    

    test_metrics = {
        "label_metrics": test_label_metrics,
        "weighted_f1": test_report["weighted avg"]["f1-score"],
        "macro_f1": test_report["macro avg"]["f1-score"],
        "micro_f1": f1_score(data["Y_test"], test_preds, average='micro'),
        "sample_f1": f1_score(data["Y_test"], test_preds, average='samples'),
        "jaccard": jaccard_score(data["Y_test"], test_preds, average='samples'),
        "hamming_loss": hamming_loss(data["Y_test"], test_preds)
    }
    

    metrics = {
        "train": train_metrics,
        "test": test_metrics
    }
    
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
    
    storage_path = CONFIG["output"]["study_db"].replace("sqlite:///", "")
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    
    try:
        print("\nAttempting to load existing study...")
        study = optuna.load_study(
            study_name="MLRF_Optimization",
            storage=CONFIG["output"]["study_db"],
            sampler=TPESampler(seed=CONFIG["optimization"]["random_state"])
        )
        print(f"Successfully loaded existing study with {len(study.trials)} trials.")
        
        # trials
        remaining_trials = max(0, CONFIG["optimization"]["n_trials"] - len(study.trials))
        if remaining_trials > 0:
            print(f"\nRunning {remaining_trials} additional trials...")
            study.optimize(objective, n_trials=remaining_trials, catch=(Exception,))
        else:
            print("\nNo additional trials needed.")

    except Exception as e:
        print(f"Study not found or failed to load: {e}")
        print("Creating a new study from scratch...")

        study = optuna.create_study(
            study_name="MLRF_Optimization",
            storage=CONFIG["output"]["study_db"],
            direction="maximize",
            sampler=TPESampler(seed=CONFIG["optimization"]["random_state"]),
        )
        print("New study created.")
        print(f"\nRunning all {CONFIG['optimization']['n_trials']} trials...")
        study.optimize(objective, n_trials=CONFIG["optimization"]["n_trials"], catch=(Exception,))
    
    best_params = {k: v for k, v in study.best_params.items() 
                  if not k.startswith('threshold_')}
    thresholds = [study.best_params[f'threshold_{i}'] 
                 for i in range(len(CONFIG["label_order"]))]
    
    with open(CONFIG["output"]["model_params"], 'w') as f:
        json.dump(best_params, f, indent=2)
    with open(CONFIG["output"]["thresholds"], 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    metrics = evaluate_configuration(best_params, thresholds)
    
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