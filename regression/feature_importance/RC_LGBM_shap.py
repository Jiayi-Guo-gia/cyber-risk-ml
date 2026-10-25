import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap  # Added for SHAP values
from lightgbm import LGBMRegressor
from sklearn.multioutput import RegressorChain

# Configuration (unchanged)
CONFIG = {
    "model_params_path": "results/RC/LGBM_RC_best_params.json",
    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/Y_train.csv"
    },
    "output": {
        "feature_importance": "results/feature/LGBM_RC_chain_top20_features_shap.csv",
        "plot": "results/feature/LGBM_RC_feature_importance_plot_shap.png"
    },
    "label_order": [1, 3, 2, 4, 0],
    "random_state": 42,
    "shap_params": {
        "n_samples": 5000  # Number of samples to use for SHAP calculation
    }
}

def load_model_components():
    with open(CONFIG['model_params_path'], 'r') as f:
        return json.load(f)

def initialize_model(params, label_order):
    base_model = LGBMRegressor(
        verbosity=-1,
        random_state=CONFIG['random_state'],
        n_jobs=-1,  # Added for parallel processing
        **params
    )
    return RegressorChain(base_model, order=label_order)

def calculate_feature_importance(model, X, original_features):
    """Calculate SHAP-based feature importance"""
    num_labels = len(model.estimators_)
    num_original_features = len(original_features)
    importance_matrix = np.zeros((num_labels, num_original_features))
    
    # Sample data for faster SHAP computation
    if CONFIG['shap_params']['n_samples'] < X.shape[0]:
        sample_idx = np.random.choice(X.shape[0], CONFIG['shap_params']['n_samples'], replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X
    
    X_work = X_sample.copy()
    
    for label_idx, estimator in enumerate(model.estimators_):
        # Calculate SHAP values
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_work)
        
        # Handle multi-output format
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary
        
        # Get absolute mean SHAP values for original features
        original_shap = np.abs(shap_values[:, :num_original_features]).mean(axis=0)
        importance_matrix[label_idx] = original_shap
        
        # Add predictions to features for next step
        if label_idx < num_labels - 1:
            y_pred = estimator.predict(X_work).reshape(-1, 1)
            X_work = np.hstack([X_work, y_pred])
    
    # Average SHAP importance across all labels
    avg_importance = importance_matrix.mean(axis=0)
    
    # Normalize
    importance_df = pd.DataFrame({
        'feature': original_features,
        'importance': avg_importance / (avg_importance.max() + 1e-10)
    }).sort_values('importance', ascending=False)
    
    return importance_df

def visualize_top_features(df, top_n=20):
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('importance', ascending=True)
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(plot_df)))

    bars = plt.barh(plot_df['feature'], plot_df['importance'], color=colors, height=0.7)
    plt.xlabel('Normalized Shap Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Features - Regressor Chain (LGBM)', fontsize=14)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.3f}', va='center', fontsize=9)

    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG['output']['plot'], dpi=300, bbox_inches='tight')
    plt.close()

def main():
    model_params = load_model_components()
    X_train = pd.read_csv(CONFIG['data_paths']['X_train'])
    y_train = pd.read_csv(CONFIG['data_paths']['y_train'])

    X_train.columns = [re.sub(r'[^\w]', '_', col) for col in X_train.columns]
    feature_names = X_train.columns.tolist()

    model = initialize_model(model_params, CONFIG['label_order'])
    model.fit(X_train.values, y_train.values)

    importance_df = calculate_feature_importance(model, X_train.values, feature_names)
    os.makedirs(os.path.dirname(CONFIG['output']['feature_importance']), exist_ok=True)
    importance_df.to_csv(CONFIG['output']['feature_importance'], index=False)

    visualize_top_features(importance_df)

    print("\n Regressor Chain Feature Importance Analysis Completed (LGBM):")
    print(f"1. CSV saved to: {CONFIG['output']['feature_importance']}")
    print(f"2. Plot saved to: {CONFIG['output']['plot']}")
    print(f"3. Modeling order used: {CONFIG['label_order']}")

if __name__ == "__main__":
    main()
