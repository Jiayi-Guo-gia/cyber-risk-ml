import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain

# Configuration
CONFIG = {
    "model_params_path": "results/RC/RF_RC_best_params.json",
    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/Y_train.csv"
    },
    "output": {
        "feature_importance": "results/feature/per_model/RF_RC_impurity_importance.csv",
        "plot": "results/feature/per_model/RF_RC_impurity_top20.png"
    },
    "label_order": [1, 0, 3, 4, 2],  # Desired modeling order
    "random_state": 42
}

def load_model_components():
    with open(CONFIG['model_params_path'], 'r') as f:
        return json.load(f)

def initialize_model(params, label_order):
    # Remove unsupported parameter if present
    params.pop('permutation_index', None)
    base_model = RandomForestRegressor(
        random_state=CONFIG['random_state'],
        n_jobs=-1,
        **params
    )
    return RegressorChain(base_model, order=label_order)

def calculate_feature_importance(model, X_df, original_features):
    """
    Calculates impurity-based feature importance from RandomForestRegressor inside RegressorChain.
    Tracks only original features (ignores chain-added ones).
    """
    num_labels = len(model.estimators_)
    num_features = len(original_features)
    importance_matrix = np.zeros((num_labels, num_features))

    X_work = X_df.copy()

    for label_idx, estimator in enumerate(model.estimators_):
        importances = estimator.feature_importances_  # RF method

        # Only the first num_features columns correspond to original input
        importances = importances[:num_features]  # truncate if chain features exist
        importance_matrix[label_idx, :len(importances)] += importances

        # Add predictions for next step in the chain
        if label_idx < num_labels - 1:
            y_pred = estimator.predict(X_work)
            X_work[f'_chain_{label_idx}'] = y_pred

    # Average across all output dimensions
    avg_importance = importance_matrix.mean(axis=0)

    # Normalize and return
    importance_df = pd.DataFrame({
        'feature': original_features,
        'importance': avg_importance / (avg_importance.max() + 1e-10)
    }).sort_values('importance', ascending=False)

    return importance_df

def visualize_top_features(df, top_n=20):
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('importance', ascending=True)
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(plot_df)))

    bars = plt.barh(plot_df['feature'], plot_df['importance'], color=colors, height=0.7)
    plt.xlabel('Normalized Impurity Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Features - Multi-Output Regression (RF)', fontsize=14)

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

    importance_df = calculate_feature_importance(model, X_train, feature_names)
    importance_df.to_csv(CONFIG['output']['feature_importance'], index=False)

    visualize_top_features(importance_df)

    print("\n Regressor Chain Feature Importance Analysis Completed:")
    print(f"1. CSV saved to: {CONFIG['output']['feature_importance']}")
    print(f"2. Plot saved to: {CONFIG['output']['plot']}")
    print(f"3. Modeling order used: {CONFIG['label_order']}")

if __name__ == "__main__":
    main()
