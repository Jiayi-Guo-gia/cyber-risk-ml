"""
Permutation Feature Importance Analysis for Random Forest Classifier Chain
"""
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=UserWarning)
import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

# Configuration parameters (only changed output paths)
CONFIG = {
    # Model configuration
    "model_params_path": "results/CC/RF_CC_best_model_params.json",
    
    # Data paths
    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/Y_train.csv"
    },
    
    "label_order": [
        'Extortion/Fraud',
        'Privacy Violation',
        'IT Error',
        'Other',
        'Data Breach',
    ],
    
    # Output configuration (changed paths)
    "output": {
        "feature_importance": "results/feature/RF_CC_permutation_importance.csv",
        "visualization": "results/feature/RF_CC_permutation_plot.png"
    },
    
    # Computational parameters (replaced shap_params with permutation_params)
    "permutation_params": {
        "sampling": {
            "enable": True,
            "n_samples": 2000,
            "random_state": 42,
            "stratify": True  
        },
        "scoring": "roc_auc",  # Options: roc_auc, accuracy, etc.
        "n_repeats": 10,
        "n_jobs": -1  # Use all available cores
    },
    
    "random_state": 42
}


def load_model_parameters():  # Changed from load_model_components to match call
    """Load trained model parameters from JSON file"""
    with open(CONFIG['model_params_path'], 'r') as f:
        return json.load(f)

def initialize_model(params):
    """Create and configure ClassifierChain model"""
    base_model = RandomForestClassifier(
        random_state=CONFIG['random_state'],
        n_jobs=-1,
        **params
    )
    return ClassifierChain(
        base_model,
        order=list(range(len(CONFIG["label_order"]))),
        random_state=CONFIG['random_state']
    )

def calculate_permutation_importance(model, X, y, feature_names):
    """Calculate permutation importance tracking all features, filtering originals at end"""
    num_original = len(feature_names)
    max_features = num_original + len(model.estimators_) - 1  # Original + chain features
    importance_matrix = np.zeros((len(model.estimators_), max_features))
    
    # Initialize working data
    X_work = X.copy()
    all_features = feature_names.copy()  # Will track all features
    
    # Sampling
    if CONFIG['permutation_params']['sampling']['enable']:
        n_samples = min(CONFIG['permutation_params']['sampling']['n_samples'], len(X))
        X_res, y_res = resample(
            X_work, y,
            n_samples=n_samples,
            random_state=CONFIG['permutation_params']['sampling']['random_state'],
            stratify=y if CONFIG['permutation_params']['sampling']['stratify'] else None
        )
    else:
        X_res, y_res = X_work, y

    for label_idx, estimator in enumerate(model.estimators_):
        # Current number of features (grows each iteration)
        num_current = len(all_features)
        
        # Calculate importance
        result = permutation_importance(
            estimator,
            X_res[:, :num_current],
            y_res[:, label_idx],
            n_repeats=CONFIG['permutation_params']['n_repeats'],
            scoring=CONFIG['permutation_params']['scoring'],
            n_jobs=CONFIG['permutation_params']['n_jobs'],
            random_state=CONFIG['random_state']
        )
        
        # Store ALL importance scores (original + chain)
        importance_matrix[label_idx, :num_current] = np.abs(result.importances).mean(axis=1)
        
        # Add chain feature if not last estimator
        if label_idx < len(model.estimators_) - 1:
            new_feature = estimator.predict_proba(X_res[:, :num_current])[:, 1]
            X_res = np.column_stack([X_res, new_feature])
            all_features.append(f'_chain_{label_idx}')

    # FILTER STEP: Get only original features' importance
    original_importance = importance_matrix[:, :num_original]
    
    # MEAN STEP: Average across labels
    avg_importance = original_importance.mean(axis=0)
    
    # Normalize
    normalized = avg_importance / (avg_importance.max() + 1e-10)
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': normalized
    }).sort_values('importance', ascending=False)

# Only change the axis label in generate_importance_plot()
def generate_importance_plot(df, top_n=20):
    """Generate publication-quality feature importance visualization"""
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('importance', ascending=True)
    
    if any(not isinstance(f, str) for f in plot_df['feature']):
        raise ValueError("All feature names must be strings.")
    
    colors = plt.cm.cividis(np.linspace(0.3, 0.9, len(plot_df)))
    bars = plt.barh(plot_df['feature'], plot_df['importance'],
                    color=colors, height=0.7)
    
    plt.xlabel('Normalized Permutation Importance Score', fontsize=12)  # Changed label
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

# Main function only needs to change the calculation function call
def main():
    """Main execution pipeline"""
    model_params = load_model_parameters()
    X_train = pd.read_csv(CONFIG['data_paths']['X_train'])
    y_train = pd.read_csv(CONFIG['data_paths']['y_train'])[CONFIG['label_order']]
    
    X_train.columns = [re.sub(r'[^\w]', '_', col) for col in X_train.columns]
    feature_names = X_train.columns.tolist()
    
    model = initialize_model(model_params)
    model.fit(X_train, y_train)
    
    importance_df = calculate_permutation_importance(  # Changed function call
        model, 
        X_train.values,
        y_train.values,
        feature_names
    )
    
    os.makedirs(os.path.dirname(CONFIG['output']['feature_importance']), exist_ok=True)
    importance_df.to_csv(CONFIG['output']['feature_importance'], index=False)
    generate_importance_plot(importance_df)
    
    print("\nPermutation importance analysis completed successfully")
    print(f"-> Feature importance saved to: {CONFIG['output']['feature_importance']}")
    print(f"-> Visualization saved to: {CONFIG['output']['visualization']}")

if __name__ == "__main__":
    main()


