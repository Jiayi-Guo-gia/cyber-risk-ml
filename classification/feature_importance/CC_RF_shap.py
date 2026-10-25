"""
SHAP Feature Importance Analysis for Random Forest Classifier Chain
"""

import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.utils import resample

# Configuration parameters
CONFIG = {
    # Model configuration
    "model_params_path": "results/CC/RF_CC_best_model_params.json",
    
    # Data paths
    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/Y_train.csv"
    },
    
    # Label structure
    "label_order": [
        'Extortion/Fraud',
        'Privacy Violation',
        'IT Error',
        'Other',
        'Data Breach',
    ],
    
    # Output configuration
    "output": {
        "feature_importance": "results/feature/per_model/RF_CC_shap_importance.csv",
        "visualization": "results/feature/per_model/RF_CC_shap_top20.png"
    },
    
    # Computational parameters
    "shap_params": {
        "sampling": {
            "enable": True,
            "n_samples": 5000,
            "random_state": 42,
            "stratify": True  
        },
        "normalization": "max"  # Options: max, sum, none
    },
    
    "random_state": 42
}

def load_model_parameters():
    """Load trained model parameters from JSON file"""
    with open(CONFIG['model_params_path'], 'r') as f:
        return json.load(f)

def initialize_model(params):
    """Create and configure ClassifierChain model with parallel processing"""
    base_model = RandomForestClassifier(
        verbose=0,
        random_state=CONFIG['random_state'],
        n_jobs=-1  # Add parallel processing here
    )
    base_model.set_params(**params)
    return ClassifierChain(
        base_model,
        order=list(range(len(CONFIG["label_order"]))),
        random_state=CONFIG['random_state']
    )

def calculate_shap_importance(model, X, feature_names):
    """
    SHAP-based global feature importance for ClassifierChain(RandomForestClassifier).
    Tracks original features only, even as chain grows.
    """
    num_original = len(feature_names)
    total_importance = np.zeros(num_original)
    
    # Sampling
    if CONFIG['shap_params']['sampling']['enable']:
        X = resample(
            X,
            n_samples=min(CONFIG['shap_params']['sampling']['n_samples'], len(X)),
            random_state=CONFIG['shap_params']['sampling']['random_state']
        )

    X_res = X.copy()
    all_features = feature_names.copy()

    for i, estimator in enumerate(model.estimators_):
        # Step 1: Fit SHAP explainer on current feature set
        X_df = pd.DataFrame(X_res, columns=all_features)
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_df)

        # Step 2: Compute SHAP importance for this label model
        if isinstance(shap_values, list):
            if len(shap_values) == 2 and shap_values[1].ndim == 2:
                values = np.abs(shap_values[1])  # Use positive class
            else:
                values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        elif isinstance(shap_values, np.ndarray):
            values = np.abs(shap_values)
        else:
            raise ValueError(f"Unexpected SHAP structure for estimator {i}: {type(shap_values)}")

        # Step 3: Truncate to original features only
        values = values[:, :num_original]
        if values.ndim == 3:
            mean_values = np.mean(values, axis=(0, 2))  # shape: (n_features,)
        elif values.ndim == 2:
            mean_values = np.mean(values, axis=0)       # shape: (n_features,)
        else:
            raise ValueError(f"Unexpected shape of SHAP values: {values.shape}")
        total_importance += mean_values

        # Step 4: If not last estimator, add chain output
        if i < len(model.estimators_) - 1:
            chain_pred = estimator.predict_proba(X_res)[:, 1]
            X_res = np.column_stack([X_res, chain_pred])
            all_features.append(f'_chain_{i}')

    # Step 5: Normalize
    avg_importance = total_importance / len(model.estimators_)
    norm_type = CONFIG['shap_params']['normalization']
    if norm_type == 'max':
        normalized = avg_importance / (np.max(avg_importance) + 1e-10)
    elif norm_type == 'sum':
        normalized = avg_importance / (np.sum(avg_importance) + 1e-10)
    else:
        normalized = avg_importance

    return pd.DataFrame({
        'feature': feature_names,
        'shap_importance': normalized
    }).sort_values('shap_importance', ascending=False)


    
def generate_importance_plot(df, top_n=20):
    """Generate publication-quality feature importance visualization"""
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('shap_importance', ascending=True)
    
    if any(not isinstance(f, str) for f in plot_df['feature']):
        raise ValueError("All feature names must be strings.")
    
    colors = plt.cm.Greys(np.linspace(0.3, 0.9, len(plot_df)))
    bars = plt.barh(plot_df['feature'], plot_df['shap_importance'],
                    color=colors, height=0.7)
    
    plt.xlabel('Normalized SHAP Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Features - Classifier Chain (Random Forest)', fontsize=14)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', va='center', fontsize=9)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = CONFIG['output']['visualization']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main execution pipeline"""
    model_params = load_model_parameters()
    X_train = pd.read_csv(CONFIG['data_paths']['X_train'])
    y_train = pd.read_csv(CONFIG['data_paths']['y_train'])[CONFIG['label_order']]
    
    X_train.columns = [re.sub(r'[^\w]', '_', col) for col in X_train.columns]
    feature_names = X_train.columns.tolist()
    
    model = initialize_model(model_params)
    # Convert to numpy array to avoid feature names warning
    model.fit(X_train.values, y_train.values)
    
    importance_df = calculate_shap_importance(
        model, 
        X_train.values,  # Pass numpy array instead of DataFrame
        feature_names
    )
    
    os.makedirs(os.path.dirname(CONFIG['output']['feature_importance']), exist_ok=True)
    importance_df.to_csv(CONFIG['output']['feature_importance'], index=False)
    generate_importance_plot(importance_df)
    
    print("\nSHAP analysis completed successfully")
    print(f"-> Feature importance saved to: {CONFIG['output']['feature_importance']}")
    print(f"-> Visualization saved to: {CONFIG['output']['visualization']}")

if __name__ == "__main__":
    main()


