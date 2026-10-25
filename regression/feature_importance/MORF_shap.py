import re
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample

# Configuration
CONFIG = {
    "model_params_path": "results/MOR/RF_best_params.json",  # Replace with your path
    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/Y_train.csv"
    },
    "output": {
        "feature_importance": "results/feature/per_model/MORF_shap_importance.csv",
        "plot": "results/feature/per_model/MORF_shap_top20.png"
    },
    "shap_params": {
        "sampling": {
            "enable": True,
            "n_samples": 5000,
            "random_state": 42
        },
        "normalization": "max"  # Options: max, sum, none
    },
    "random_state": 42
}

def load_model_components():
    with open(CONFIG['model_params_path'], 'r') as f:
        return json.load(f)

def initialize_model(params):
    model = RandomForestRegressor(
        random_state=CONFIG['random_state'],
        n_jobs=-1,
        **params
    )
    return model
def calculate_shap_feature_importance(model, X_df):
    if CONFIG['shap_params']['sampling']['enable']:
        X_df = resample(
            X_df,
            n_samples=min(CONFIG['shap_params']['sampling']['n_samples'], len(X_df)),
            random_state=CONFIG['shap_params']['sampling']['random_state']
        )

    X_values = X_df.values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_values)

    # shap_values shape: (n_samples, n_features, n_outputs)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        total_importance = np.abs(shap_values).mean(axis=(0, 2))  # mean over samples and outputs
    else:
        raise ValueError(f"Unexpected SHAP value shape: {np.shape(shap_values)}")

    # Normalization
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
    plt.title(f'Top {top_n} Features - RandomForest Regressor)', fontsize=14)

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

    model = initialize_model(model_params)
    model.fit(X_train.values, y_train.values)

    importance_df = calculate_shap_feature_importance(model, X_train)
    os.makedirs(os.path.dirname(CONFIG['output']['feature_importance']), exist_ok=True)
    importance_df.to_csv(CONFIG['output']['feature_importance'], index=False)

    visualize_top_features(importance_df)

    print("\nSHAP Feature Importance Analysis Completed:")
    print(f"1. CSV saved to: {CONFIG['output']['feature_importance']}")
    print(f"2. Plot saved to: {CONFIG['output']['plot']}")

if __name__ == "__main__":
    main()

