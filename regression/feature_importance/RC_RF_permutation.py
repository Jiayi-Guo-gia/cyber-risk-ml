import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from sklearn.inspection import permutation_importance
from sklearn.utils import resample

# Configuration
CONFIG = {
    "model_params_path": "results/RC/RF_RC_best_params.json",
    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/Y_train.csv"
    },
    "output": {
        "feature_importance": "results/feature/per_model/RF_RC_permutation_importance.csv",
        "plot": "results/feature/per_model/RF_RC_permutation_top20.png"
    },
    "label_order": [1, 0, 3, 4, 2],
    "permutation_params": {
        "sampling": {
            "enable": True,
            "n_samples": 5000,
            "random_state": 42,
            "stratify": False
        },
        "scoring": "neg_mean_squared_error",
        "n_repeats": 10,
        "n_jobs": 1
    },
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
        n_jobs=1,
        **params
    )
    return RegressorChain(base_model, order=label_order)

def calculate_permutation_importance(model, X, y, original_features):
    """Calculate permutation importance for RegressorChain (track all features, filter originals at end)"""
    num_labels = len(model.estimators_)
    all_features = original_features.copy()
    
    # First pass: determine complete feature set through all chain steps
    X_sim = X.copy()
    for label_idx in range(num_labels - 1):
        X_sim[f'_chain_{label_idx}'] = 0  # Add placeholder for chain features
    complete_features = X_sim.columns.tolist()
    num_complete = len(complete_features)
    
    importance_matrix = np.zeros((num_labels, num_complete))
    
    # Apply sampling if enabled
    if CONFIG['permutation_params']['sampling']['enable']:
        sample_idx = resample(
            np.arange(len(X)),
            n_samples=min(CONFIG['permutation_params']['sampling']['n_samples'], len(X)),
            random_state=CONFIG['permutation_params']['sampling']['random_state'],
            stratify=None
        )
        X_work = X.iloc[sample_idx].copy()
        y_work = y.iloc[sample_idx].copy()
    else:
        X_work = X.copy()
        y_work = y.copy()

    # Track all features through the chain
    current_features = original_features.copy()
    
    for label_idx, estimator in enumerate(model.estimators_):
        # Calculate importance using current features
        result = permutation_importance(
            estimator,
            X_work[current_features],
            y_work.iloc[:, label_idx],
            n_repeats=CONFIG['permutation_params']['n_repeats'],
            random_state=CONFIG['random_state'],
            n_jobs=CONFIG['permutation_params']['n_jobs'],
            scoring=CONFIG['permutation_params']['scoring']
        )
        
        # ABS FIRST → THEN MEAN (across repeats)
        abs_mean_importance = np.abs(result.importances).mean(axis=1)  # Shape: (n_features,)
        
        # Store importance for current features
        for i, feat in enumerate(current_features):
            complete_idx = complete_features.index(feat)
            importance_matrix[label_idx, complete_idx] = abs_mean_importance[i]
        
        # Add predictions as new features for next step
        if label_idx < num_labels - 1:
            y_pred = estimator.predict(X_work[current_features]).reshape(-1, 1)
            new_feature = f'_chain_{label_idx}'
            X_work[new_feature] = y_pred
            current_features.append(new_feature)
    
    # Aggregate across labels (only for original features)
    original_indices = [complete_features.index(f) for f in original_features]
    avg_importance = np.abs(importance_matrix[:, original_indices]).mean(axis=0)
    
    # Normalize
    max_imp = avg_importance.max() if avg_importance.max() != 0 else 1
    normalized = avg_importance / max_imp
    
    return pd.DataFrame({
        'feature': original_features,
        'importance': normalized
    }).sort_values('importance', ascending=False)
    
def visualize_top_features(df, top_n=20):
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('importance', ascending=True)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(plot_df)))

    bars = plt.barh(plot_df['feature'], plot_df['importance'], color=colors, height=0.7)
    plt.xlabel('Normalized Permutation Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Features - Regressor Chain (RF)', fontsize=14)

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

    # Clean feature names
    X_train.columns = [re.sub(r'[^\w]', '_', col) for col in X_train.columns]
    feature_names = X_train.columns.tolist()

    model = initialize_model(model_params, CONFIG['label_order'])
    model.fit(X_train, y_train)

    importance_df = calculate_permutation_importance(
        model,
        X_train,
        y_train,
        feature_names
    )
    
    os.makedirs(os.path.dirname(CONFIG['output']['feature_importance']), exist_ok=True)
    importance_df.to_csv(CONFIG['output']['feature_importance'], index=False)

    visualize_top_features(importance_df)

    print("\nRegressor Chain Permutation Importance Analysis Completed (RF):")
    print(f"1. CSV saved to: {CONFIG['output']['feature_importance']}")
    print(f"2. Plot saved to: {CONFIG['output']['plot']}")
    print(f"3. Modeling order used: {CONFIG['label_order']}")

if __name__ == "__main__":
    main()