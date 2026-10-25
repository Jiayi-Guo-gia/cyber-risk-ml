"""
Multi-Output Random Forest Permutation Importance Analysis
"""

import json
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import permutation_importance
from sklearn.utils import resample

# Configuration
CONFIG = {
    "model_params_path": "results/MOR/RF_MOR_best_params.json",
    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/Y_train.csv"
    },
    "output": {
        "feature_importance": "results/feature/RF_MOR_top20_features_permutation.csv",
        "plot": "results/feature/RF_MOR_feature_importance_plot_permutation.png"
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
    base_model = RandomForestRegressor(
        random_state=CONFIG['random_state'],
        n_jobs=-1,
        **params
    )
    return MultiOutputRegressor(base_model)

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

    num_targets = len(model.estimators_)
    num_features = X_sample.shape[1]
    total_importance = np.zeros(num_features)

    feature_names = X_sample.columns.tolist()
    # Initialize importance storage
    importance_matrix = np.zeros((len(model.estimators_), num_features))

    for i, estimator in enumerate(model.estimators_):
        X_values = X_sample.copy()
        y_values = y_sample.iloc[:, i] if hasattr(y_sample, 'iloc') else y_sample[:, i]

        result = permutation_importance(
            estimator,
            X_values,
            y_values,
            n_repeats=CONFIG['permutation_params']['n_repeats'],
            scoring=CONFIG['permutation_params']['scoring'],
            n_jobs=CONFIG['permutation_params']['n_jobs'],
            random_state=CONFIG['random_state']
        )
        # Store absolute mean importance for this target
        importance_matrix[i] = np.abs(result.importances).mean(axis=1)  # Mean across repeats
    
    total_importance = importance_matrix.mean(axis=0)  # Average across targets
    normalized = total_importance / (total_importance.max() + 1e-10)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': normalized
    }).sort_values('importance', ascending=False)

    return importance_df

def visualize_top_features(df, top_n=20):
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('importance', ascending=True)
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(plot_df)))

    bars = plt.barh(plot_df['feature'], plot_df['importance'], color=colors, height=0.7)
    plt.xlabel('Normalized Permutation Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Features - Multi-Output Regression (RF)', fontsize=14)

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
