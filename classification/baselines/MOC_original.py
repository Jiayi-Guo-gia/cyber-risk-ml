import numpy as np
import os 
import pandas as pd
import lightgbm as lgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    hamming_loss, 
    f1_score,
    jaccard_score
)

def main():
    # Load data
    X_train = pd.read_csv("data/X_train.csv")
    Y_train = pd.read_csv("data/Y_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    Y_test = pd.read_csv("data/Y_test.csv")

    # Define the base LightGBM classifier
    lgb_clf = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        metric='binary_logloss',
        num_leaves=64,
        max_depth=8,
        learning_rate=0.05,
        verbose=-1
    )
    
    # Wrap the base classifier in a MultiOutputClassifier
    multi_output_clf = MultiOutputClassifier(lgb_clf)
    multi_output_clf.fit(X_train, Y_train)

    # Predict probabilities
    Y_pred_prob = multi_output_clf.predict_proba(X_test)
    Y_pred_prob = np.array([prob[:, 1] for prob in Y_pred_prob])
    
    # Convert to binary predictions
    Y_pred_cls = (Y_pred_prob.T > 0.5).astype(int)
    Y_pred_df = pd.DataFrame(Y_pred_cls, columns=Y_test.columns)

    # Compute metrics
    metrics = {
        'jaccard': jaccard_score(Y_test, Y_pred_df, average='samples'),
        'hamming': hamming_loss(Y_test, Y_pred_df),
        'macro_f1': f1_score(Y_test, Y_pred_df, average='macro'),
        'micro_f1': f1_score(Y_test, Y_pred_df, average='micro'),
        'weighted_f1': f1_score(Y_test, Y_pred_df, average='weighted'),
        'sample_f1': f1_score(Y_test, Y_pred_df, average='samples')
    }

    # Print reports
    print("\n=== Detailed Classification Report ===")
    print(classification_report(
        Y_test.values,
        Y_pred_df.values,
        target_names=Y_test.columns,
        zero_division=0
    ))

    print("\n=== Final Evaluation Metrics ===")
    print(f"Jaccard Score:        {metrics['jaccard']:.4f}")
    print(f"Hamming Loss:         {metrics['hamming']:.4f}")
    print(f"Macro F1 Score:       {metrics['macro_f1']:.4f}") 
    print(f"Micro F1 Score:       {metrics['micro_f1']:.4f}")
    print(f"Weighted F1 Score:    {metrics['weighted_f1']:.4f}")
    print(f"Sample F1 Score:      {metrics['sample_f1']:.4f}")

    # Save results
    results_dir = "results/"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save predictions
    Y_pred_df.to_csv(os.path.join(results_dir, "MOC_original_predictions.csv"), index=False)
    
    # Save metrics
    pd.DataFrame({
        'Metric': ['Jaccard', 'Hamming Loss', 'Macro F1',
                  'Micro F1', 'Weighted F1', 'Sample F1'],
        'Value': [metrics[k] for k in ['jaccard', 'hamming', 'macro_f1',
                                      'micro_f1', 'weighted_f1', 'sample_f1']]
    }).to_csv(os.path.join(results_dir, "MOC_original_metrics.csv"), index=False)

if __name__ == "__main__":
    main()
