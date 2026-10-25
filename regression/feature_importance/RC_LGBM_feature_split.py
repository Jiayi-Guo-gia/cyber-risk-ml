import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.multioutput import RegressorChain

# Configuration
CONFIG = {
    "model_params_path": "results/RC/LGBM_RC_best_params.json",
    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/Y_train.csv"
    },
    "output": {
        "feature_importance": "results/feature/per_model/LGBM_RC_split_importance.csv",
        "plot": "results/feature/per_model/LGBM_RC_split_top20.png"
    },
    "label_order": [1, 2, 4, 0, 3],
    "random_state": 42
}

def load_model_components():
    """Load and validate model parameters"""
    with open(CONFIG['model_params_path'], 'r') as f:
        params = json.load(f)
    assert 'importance_type' not in params, "Remove importance_type from params"
    return params

def initialize_model(params):
    """Create Regressor Chain with split-based importance"""
    base_model = LGBMRegressor(
        importance_type='split',
        verbosity=-1,
        random_state=CONFIG['random_state'],
        **params
    )
    return RegressorChain(base_model, order=CONFIG['label_order'])

def clean_feature_names(feature_names):
    """Standardize feature names and ensure uniqueness"""
    cleaned = []
    seen = set()
    for name in feature_names:
        # Replace special characters and trim underscores
        clean_name = re.sub(r'[^\w]', '_', name).strip('_')
        # Handle empty names and duplicates
        if not clean_name:
            clean_name = "empty_feature"
        orig_name = clean_name
        counter = 1
        while clean_name in seen:
            clean_name = f"{orig_name}_dup{counter}"
            counter += 1
        cleaned.append(clean_name)
        seen.add(clean_name)
    return cleaned

def calculate_feature_importance(model, original_features):
    """
    Calculate gain importance across chain models
    Filters out chain features added during training
    """
    num_targets = len(model.estimators_)
    num_original = len(original_features)
    importance_matrix = np.zeros((num_targets, num_original))
    
    # Track added chain features
    chain_features = set()
    
    for target_idx, estimator in enumerate(model.estimators_):
        # Get all features used in current estimator
        total_features = len(estimator.feature_importances_)
        
        # Calculate original features to keep
        valid_features = [f for f in original_features if f not in chain_features]
        valid_indices = [i for i, f in enumerate(original_features) if f in valid_features]
        
        # Extract importance for original features only
        imp_values = estimator.feature_importances_[:len(valid_features)]
        for i, idx in enumerate(valid_indices):
            importance_matrix[target_idx, idx] = imp_values[i]
        
        # Update chain features for next iterations
        if target_idx < num_targets - 1:
            chain_features.add(f'_chain_{target_idx}')
    
    # Average across targets
    global_importance = importance_matrix.mean(axis=0)
    
    return pd.DataFrame({
        'feature': original_features,
        'importance': global_importance / (global_importance.max() + 1e-10)
    }).sort_values('importance', ascending=False)

def visualize_top_features(df, top_n=20):
    """Generate horizontal bar plot of feature importance"""
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('importance', ascending=True)
    
    # Blue color gradient
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(plot_df)))
    
    bars = plt.barh(plot_df['feature'], plot_df['importance'], 
                   color=colors, height=0.7)
    
    plt.xlabel('Normalized Split Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Features - Regressor Chain (LightGBM)', fontsize=14)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', va='center', fontsize=9)
    
    # Style adjustments
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG['output']['plot'], dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load data and parameters
    params = load_model_components()
    X_train = pd.read_csv(CONFIG['data_paths']['X_train'])
    y_train = pd.read_csv(CONFIG['data_paths']['y_train'])
    
    # Clean and validate features
    original_features = X_train.columns.tolist()
    X_train.columns = clean_feature_names(original_features)
    feature_names = X_train.columns.tolist()
    
    # Train model
    model = initialize_model(params)
    model.fit(X_train, y_train)
    
    # Calculate and save importance
    importance_df = calculate_feature_importance(model, feature_names)
    importance_df.to_csv(CONFIG['output']['feature_importance'], index=False)
    
    # Generate visualization
    visualize_top_features(importance_df)
    
    # Validation output
    print("\nFeature Importance Analysis Completed Successfully")
    print(f"Top 5 Features:")
    print(importance_df.head().to_string(index=False))
    print(f"\nVisualization saved to: {CONFIG['output']['plot']}")

if __name__ == "__main__":
    main()
