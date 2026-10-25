import json
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import resample


# Configuration
CONFIG = {
    "model_params_path": "results/MOR/LGBM_MOR_best_params.json",
    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/Y_train.csv"
    },
    "output": {
        "feature_importance": "results/feature/per_model/LGBM_MOR_shap_importance.csv",
        "plot": "results/feature/per_model/LGBM_MOR_shap_top20.png"
    },
    "shap_params": {
        "sampling": {
            "enable": True,
            "n_samples": 5000,
            "random_state": 42,
            "stratify": False  # only relevant for classification
        },
        "normalization": "max"  # Options: max, sum, none
    },
    "random_state": 42
}

def load_model_components():
    with open(CONFIG['model_params_path'], 'r') as f:
        return json.load(f)

def initialize_model(params):
    base_model = LGBMRegressor(
        verbosity=-1,
        random_state=CONFIG['random_state'],
        **params
    )
    return MultiOutputRegressor(base_model)

def calculate_shap_feature_importance(model, X_df):
    """
    Calculate SHAP-based feature importance across all targets
    with optional sampling and normalization.
    """
    # Optional sampling
    if CONFIG['shap_params']['sampling']['enable']:
        X_df = resample(
            X_df,
            n_samples=CONFIG['shap_params']['sampling']['n_samples'],
            random_state=CONFIG['shap_params']['sampling']['random_state']
        )

    num_targets = len(model.estimators_)
    num_features = X_df.shape[1]
    total_importance = np.zeros(num_features)

    for i, estimator in enumerate(model.estimators_):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_df)

        # For regression, shap_values is usually a list of 1 array
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Reshape or reduce if needed
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        elif shap_values.ndim == 3:
            shap_values = np.sum(np.abs(shap_values), axis=0)

        label_importance = np.abs(shap_values).mean(axis=0)
        total_importance += label_importance

    # Normalize
    norm = CONFIG['shap_params']['normalization']
    if norm == 'max':
        normalized = total_importance / (total_importance.max() + 1e-10)
    elif norm == 'sum':
        normalized = total_importance / (total_importance.sum() + 1e-10)
    else:
        normalized = total_importance

    return pd.DataFrame({
        'feature': X_df.columns.tolist(),
        'importance': normalized
    }).sort_values('importance', ascending=False)
    
def visualize_top_features(df, top_n=20):
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('importance', ascending=True)
    colors = plt.cm.Greys(np.linspace(0.3, 0.9, len(plot_df)))

    bars = plt.barh(plot_df['feature'], plot_df['importance'], color=colors, height=0.7)
    plt.xlabel('Normalized SHAP Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Features - Multi-Output Regression (LightGBM)', fontsize=14)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', fontsize=9)

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

    model = initialize_model(model_params)
    model.fit(X_train.values, y_train.values)

    importance_df = calculate_shap_feature_importance(model, X_train)
    os.makedirs(os.path.dirname(CONFIG['output']['feature_importance']), exist_ok=True)
    importance_df.to_csv(CONFIG['output']['feature_importance'], index=False)

    visualize_top_features(importance_df)

    print("\nSHAP Feature Importance Analysis Completed:")
    print(f"1. SHAP importance saved to: {CONFIG['output']['feature_importance']}")
    print(f"2. Visualization saved to: {CONFIG['output']['plot']}")

if __name__ == "__main__":
    main()
