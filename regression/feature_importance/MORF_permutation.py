"""
Random Forest Permutation Importance Analysis (Single Output)
"""

import json
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.utils import resample

# Configuration
CONFIG = {
    "model_params_path": "results/MOR/RF_best_params.json",
    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/y_train.csv"  # Single output target
    },
    "output": {
        "feature_importance": "results/feature/per_model/MORF_permutation_importance.csv",
        "plot": "results/feature/per_model/MORF_permutation_top20.png"
    },
    "permutation_params": {
        "sampling": {
            "enable": True,
            "n_samples": 5000,
            "random_state": 42,
            "stratify": False
        },
        "scoring": "neg_mean_squared_error",
        "n_repeats": 10,
        "n_jobs": 1  # safer to avoid crash
    },
    "random_state": 42
}

def load_model_components():
    with open(CONFIG['model_params_path'], 'r') as f:
        return json.load(f)

def initialize_model(params):
    return RandomForestRegressor(
        random_state=CONFIG['random_state'],
        n_jobs=-1,
        **params
    )

def calculate_permutation_feature_importance(model, X_df, y):
    if CONFIG['permutation_params']['sampling']['enable']:
        n_samples = min(CONFIG['permutation_params']['sampling']['n_samples'], len(X_df))
        rng = np.random.RandomState(CONFIG['permutation_params']['sampling']['random_state'])
        sample_idx = rng.choice(len(X_df), size=n_samples, replace=False)
        X_sample = X_df.iloc[sample_idx]
        y_sample = y.iloc[sample_idx] if hasattr(y, 'iloc') else y[sample_idx]
    else:
        X_sample = X_df
        y_sample = y

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=CONFIG['permutation_params']['n_repeats'],
        scoring=CONFIG['permutation_params']['scoring'],
        n_jobs=CONFIG['permutation_params']['n_jobs'],
        random_state=CONFIG['random_state']
    )

    # Normalize importance scores
    # Take absolute importance and average across repeats
    abs_importance = np.abs(result.importances).mean(axis=1)  # shape: (n_features,)
    normalized = abs_importance / (abs_importance.max() + 1e-10)

    importance_df = pd.DataFrame({
        'feature': X_sample.columns.tolist(),
        'importance': normalized
    }).sort_values('importance', ascending=False)

    return importance_df

def visualize_top_features(df, top_n=20):
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('importance', ascending=True)
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(plot_df)))

    bars = plt.barh(plot_df['feature'], plot_df['importance'], color=colors, height=0.7)
    plt.xlabel('Normalized Permutation Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Features - Multi-target Regression Trees', fontsize=14)

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

    # Clean feature names
    X_train.columns = [re.sub(r'[^\w]', '_', col) for col in X_train.columns]

    model = initialize_model(model_params)
    model.fit(X_train, y_train)

    importance_df = calculate_permutation_feature_importance(model, X_train, y_train)
    
    os.makedirs(os.path.dirname(CONFIG['output']['feature_importance']), exist_ok=True)
    importance_df.to_csv(CONFIG['output']['feature_importance'], index=False)

    visualize_top_features(importance_df)

    print("\nPermutation Feature Importance Analysis Completed:")
    print(f"1. Feature importance saved to: {CONFIG['output']['feature_importance']}")
    print(f"2. Visualization saved to: {CONFIG['output']['plot']}")

if __name__ == "__main__":
    main()