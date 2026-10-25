
"""
Multi-Label SHAP Analysis with Random Forest
"""

import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import resample

# Configuration parameters
CONFIG = {
    # Model configuration
    "model_params_path": "results/MOC/RF_MOC_best_model_params.json",
    
    # Data paths
    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/Y_train.csv"
    },
    
    # Label schema
    "labels": [
        'Data Breach',
        'Privacy Violation',
        'Extortion/Fraud',
        'IT Error',
        'Other'
    ],
    # Output configuration
    "output": {
        "feature_importance": "results/feature/RF_MOC_shap_importance.csv",
        "visualization": "results/feature/RF_MOC_shap_plot.png"
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
    """Create MultiOutputClassifier with Random Forest"""
    base_model = RandomForestClassifier(
        verbose=0,
        random_state=CONFIG['random_state']
    )
    base_model.set_params(**params)
    return MultiOutputClassifier(base_model)

def calculate_shap_importance(model, X, feature_names):
    """Compute global SHAP feature importance for MultiOutputClassifier"""
    if CONFIG['shap_params']['sampling']['enable']:
        X = resample(
            X,
            n_samples=min(CONFIG['shap_params']['sampling']['n_samples'], len(X)),
            random_state=CONFIG['shap_params']['sampling']['random_state']
        )

    X_df = pd.DataFrame(X, columns=feature_names)
    num_features = len(feature_names)
    total_importance = np.zeros(num_features)

    for i, estimator in enumerate(model.estimators_):
        explainer = shap.TreeExplainer(estimator, model_output="raw")
        shap_vals = explainer.shap_values(X_df)

        if isinstance(shap_vals, list):
            if len(shap_vals) == 2 and shap_vals[1].ndim == 2:
                values = np.abs(shap_vals[1])  # Positive class
            else:
                values = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        elif isinstance(shap_vals, np.ndarray):
            values = np.abs(shap_vals)
        else:
            raise ValueError(f"Unexpected SHAP value type for estimator {i}: {type(shap_vals)}")

        if values.ndim == 3:
            mean_values = np.mean(values, axis=(0, 2))  # shape: (n_features,)
        elif values.ndim == 2:
            mean_values = np.mean(values, axis=0)       # shape: (n_features,)
        else:
            raise ValueError(f"Unexpected shape of SHAP values: {values.shape}")
        total_importance += mean_values

    norm_type = CONFIG['shap_params']['normalization']
    if norm_type == 'max':
        importance_scores = total_importance / (np.max(total_importance) + 1e-10)
    elif norm_type == 'sum':
        importance_scores = total_importance / (np.sum(total_importance) + 1e-10)
    else:
        importance_scores = total_importance

    return pd.DataFrame({
        'feature': feature_names,
        'shap_importance': importance_scores
    }).sort_values('shap_importance', ascending=False)


def generate_importance_plot(df, top_n=20):
    """Generate visualization for feature importance"""
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('shap_importance', ascending=True)
    
    if any(not isinstance(f, str) for f in plot_df['feature']):
        raise ValueError("Invalid feature names detected")
    
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(plot_df)))
    bars = plt.barh(plot_df['feature'], plot_df['shap_importance'],
                    color=colors, height=0.7)
    
    plt.xlabel('Normalized SHAP Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Features - Multi-Output Classification (RF)', fontsize=14)
    
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
    # Load data and model
    model_params = load_model_parameters()
    X_train = pd.read_csv(CONFIG['data_paths']['X_train'])
    y_train = pd.read_csv(CONFIG['data_paths']['y_train'])[CONFIG['labels']]
    
    # Clean feature names
    X_train.columns = [re.sub(r'[^\w]', '_', col) for col in X_train.columns]
    feature_names = X_train.columns.tolist()
    
    # Train model
    model = initialize_model(model_params)
    model.fit(X_train.values, y_train.values)
    
    # Calculate and save importance
    importance_df = calculate_shap_importance(model, X_train.values, feature_names)
    
    os.makedirs(os.path.dirname(CONFIG['output']['feature_importance']), exist_ok=True)
    importance_df.to_csv(CONFIG['output']['feature_importance'], index=False)
    generate_importance_plot(importance_df)
    
    print("\nAnalysis completed:")
    print(f"  - Feature importance: {CONFIG['output']['feature_importance']}")
    print(f"  - Visualization: {CONFIG['output']['visualization']}")

if __name__ == "__main__":
    main()

