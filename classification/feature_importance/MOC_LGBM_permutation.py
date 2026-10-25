"""
Multi-Label Permutation Importance Analysis with LightGBM
"""

import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration parameters
CONFIG = {
    # Model configuration
    "model_params_path": "results/MOC/LGBM_MOC_best_model_params.json",
    
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
        "feature_importance": "results/feature/per_model/LGBM_MOC_permutation_importance.csv",
        "visualization": "results/feature/per_model/LGBM_MOC_permutation_top20.png"
    },
    
    # Computational parameters
    "permutation_params": {
        "sampling": {
            "enable": True,
            "n_samples": 5000,  # Reduced for faster computation
            "random_state": 42,
            "stratify": True 
        },
        "scoring": "roc_auc",
        "n_repeats": 10,
        "n_jobs": -1
    },
    
    "random_state": 42
}

def load_model_parameters():
    """Load trained model parameters from JSON file"""
    with open(CONFIG['model_params_path'], 'r') as f:
        return json.load(f)

def initialize_model(params):
    """Create MultiOutputClassifier with LightGBM"""
    base_model = LGBMClassifier(
        verbose=-1,
        random_state=CONFIG['random_state'],
        n_jobs=-1  # Enable parallel processing
    )
    base_model.set_params(**params)
    return MultiOutputClassifier(base_model)

def calculate_permutation_importance(model, X, y, feature_names):
    # Convert to DataFrame for feature name alignment
    X_df = pd.DataFrame(X, columns=feature_names)

    # Apply optional sampling
    if CONFIG['permutation_params']['sampling']['enable']:
        X_df, y = resample(
            X_df, y,
            n_samples=CONFIG['permutation_params']['sampling']['n_samples'],
            random_state=CONFIG['permutation_params']['sampling']['random_state'],
            stratify=y if CONFIG['permutation_params']['sampling']['stratify'] else None
        )

    # Initialize importance matrix
    importance_matrix = np.zeros((len(model.estimators_), len(feature_names)))
    
    for target_idx, estimator in enumerate(model.estimators_):
        result = permutation_importance(
            estimator,
            X_df,
            y[:, target_idx],
            scoring=CONFIG['permutation_params']['scoring'],
            n_repeats=CONFIG['permutation_params']['n_repeats'],
            n_jobs=CONFIG['permutation_params']['n_jobs'],
            random_state=CONFIG['random_state']
        )
        importance_matrix[target_idx] = np.abs(result.importances).mean(axis=1)  # Mean across repeats
    
    total_importance = importance_matrix.mean(axis=0)  # Average across targets
    normalized = total_importance / (total_importance.max() + 1e-10)

    
    # Create DataFrame with sorted results
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': normalized
    }).sort_values('importance', ascending=False)
    
    return importance_df

def generate_importance_plot(df, top_n=20):
    """Generate visualization for feature importance"""
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('importance', ascending=True)
    
    if any(not isinstance(f, str) for f in plot_df['feature']):
        raise ValueError("Invalid feature names detected")
    
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(plot_df)))
    bars = plt.barh(plot_df['feature'], plot_df['importance'],
                    color=colors, height=0.7)
    
    plt.xlabel('Normalized Permutation Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Features - Multi-Output Classification (LightGBM)', fontsize=14)
    
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
    model.fit(X_train, y_train)
    
    # Calculate and save importance
    importance_df = calculate_permutation_importance(model, X_train.values, y_train.values, feature_names)
    
    os.makedirs(os.path.dirname(CONFIG['output']['feature_importance']), exist_ok=True)
    importance_df.to_csv(CONFIG['output']['feature_importance'], index=False)
    generate_importance_plot(importance_df)
    
    print("\nPermutation importance analysis completed:")
    print(f"  - Feature importance: {CONFIG['output']['feature_importance']}")
    print(f"  - Visualization: {CONFIG['output']['visualization']}")

if __name__ == "__main__":
    main()