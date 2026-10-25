import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

# Configuration
CONFIG = {
    "model_params_path": "results/MOR/LGBM_MOR_best_params.json",
    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/Y_train.csv"
    },
    "output": {
        "feature_importance": "results/feature/LGBM_MOR_top20_features_gain.csv",
        "plot": "results/feature/LGBM_MOR_feature_importance_plot_gain.png"
    },
    "random_state": 42
}

def load_model_components():
    """Load model parameters from JSON file"""
    with open(CONFIG['model_params_path'], 'r') as f:
        return json.load(f)

def initialize_model(params):
    """Create MultiOutputRegressor with LightGBM base model"""
    base_model = LGBMRegressor(
        verbosity=-1,
        random_state=CONFIG['random_state'],
        importance_type='gain',
        **params
    )
    return MultiOutputRegressor(base_model)

def calculate_feature_importance(model, X, feature_names):
    """
    Calculate normalized feature importance across all targets
    Returns: DataFrame sorted by importance score
    """
    num_labels = len(model.estimators_)
    importance_matrix = np.zeros((num_labels, len(feature_names)))
    
    for target_idx, estimator in enumerate(model.estimators_):
        importance_matrix[target_idx] = estimator.feature_importances_
    
    # Average importance across all targets
    global_importance = importance_matrix.mean(axis=0)
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': global_importance / global_importance.max()  # Normalized to [0,1]
    }).sort_values('importance', ascending=False)

def visualize_top_features(df, top_n=20):
    """Generate horizontal bar plot of feature importance"""
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('importance', ascending=True)
    
    # Blue color gradient
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(plot_df)))
    
    bars = plt.barh(plot_df['feature'], plot_df['importance'], 
                   color=colors, height=0.7)
    
    plt.xlabel('Normalized Gain Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Features - Multi-Output Regression (LightGBM)', fontsize=14)
    
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
    # Load model components
    model_params = load_model_components()
    
    # Load and prepare data
    X_train = pd.read_csv(CONFIG['data_paths']['X_train'])
    y_train = pd.read_csv(CONFIG['data_paths']['y_train'])
    
    # Clean feature names (replace special characters)
    X_train.columns = [re.sub(r'[^\w]', '_', col) for col in X_train.columns]
    feature_names = X_train.columns.tolist()
    
    # Train model
    model = initialize_model(model_params)
    model.fit(X_train, y_train)
    
    # Compute and save feature importance
    importance_df = calculate_feature_importance(model, X_train.values, feature_names)
    importance_df.to_csv(CONFIG['output']['feature_importance'], index=False)
    
    # Generate visualization
    visualize_top_features(importance_df)
    
    print("\nRegression Feature Importance Analysis Completed:")
    print(f"1. Feature importance saved to: {CONFIG['output']['feature_importance']}")
    print(f"2. Visualization saved to: {CONFIG['output']['plot']}")

if __name__ == "__main__":
    main()