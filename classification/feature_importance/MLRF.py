"""
Multi-Label SHAP Analysis with Random Forest (Power Set Encoding)
"""

import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# Configuration
CONFIG = {
    "model_params_path": "results/MOC/RF_best_model_params.json",

    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/Y_train.csv"
    },

    "labels": [
        'Extortion/Fraud',
        'Privacy Violation',
        'IT Error',
        'Data Breach',
        'Other'
    ],

    "output": {
        "feature_importance": "results/feature/MLRF_shap_importance.csv",
        "visualization": "results/feature/MLRF_top20_shap_plot.png"
    },

    "shap_params": {
        "sampling": {
            "enable": True,
            "n_samples": 5000,
            "random_state": 42,
            "stratify": False
        },
        "normalization": "max"
    },

    "random_state": 42
}

def multilabel_to_class_id(y_multilabel):
    """Convert multi-label binary array to integer class ID (power set encoding)"""
    y_multilabel = np.asarray(y_multilabel)
    powers = 2 ** np.arange(y_multilabel.shape[1])[::-1]  # MSB first
    return y_multilabel.dot(powers)

def class_id_to_multilabel(class_ids, n_labels):
    """Decode integer class IDs back to binary multi-label format"""
    return ((class_ids[:, None] & (1 << np.arange(n_labels)[::-1])) > 0).astype(int)

def load_model_parameters():
    with open(CONFIG['model_params_path'], 'r') as f:
        return json.load(f)

def initialize_model(params):
    model = RandomForestClassifier(
        verbose=0,
        random_state=CONFIG['random_state']
    )
    model.set_params(**params)
    return model

def calculate_shap_importance(model, X, feature_names):
    if CONFIG['shap_params']['sampling']['enable']:
        X = resample(
            X,
            n_samples=CONFIG['shap_params']['sampling']['n_samples'],
            random_state=CONFIG['shap_params']['sampling']['random_state']
        )

    X_df = pd.DataFrame(X, columns=feature_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    if isinstance(shap_values, list):
        per_class_mean = np.array([np.mean(np.abs(sv), axis=0) for sv in shap_values])
        mean_importance = np.mean(per_class_mean, axis=0)
    else:
        mean_importance = np.mean(np.abs(shap_values), axis=0)

    norm_type = CONFIG['shap_params']['normalization']
    if norm_type == 'max':
        importance_scores = mean_importance / (np.max(mean_importance) + 1e-10)
    elif norm_type == 'sum':
        importance_scores = mean_importance / (np.sum(mean_importance) + 1e-10)
    else:
        importance_scores = mean_importance

    return pd.DataFrame({
        'feature': feature_names,
        'shap_importance': importance_scores
    }).sort_values('shap_importance', ascending=False)

def generate_importance_plot(df, top_n=20):
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('shap_importance', ascending=True)

    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(plot_df)))
    bars = plt.barh(plot_df['feature'], plot_df['shap_importance'],
                    color=colors, height=0.7)

    plt.xlabel('Normalized SHAP Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Features - RF Classifier', fontsize=14)

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

def main():
    model_params = load_model_parameters()
    X_train = pd.read_csv(CONFIG['data_paths']['X_train'])
    y_raw = pd.read_csv(CONFIG['data_paths']['y_train'])  # multi-hot matrix

    # Clean feature names
    X_train.columns = [re.sub(r'[^\w]', '_', col) for col in X_train.columns]
    feature_names = X_train.columns.tolist()

    # Convert multi-label to integer class
    y_train = multilabel_to_class_id(y_raw.values)

    # Train model
    model = initialize_model(model_params)
    model.fit(X_train, y_train)

    # SHAP Importance
    importance_df = calculate_shap_importance(model, X_train.values, feature_names)

    os.makedirs(os.path.dirname(CONFIG['output']['feature_importance']), exist_ok=True)
    importance_df.to_csv(CONFIG['output']['feature_importance'], index=False)
    generate_importance_plot(importance_df)

    print("\nSHAP analysis completed:")
    print(f"  - Feature importance: {CONFIG['output']['feature_importance']}")
    print(f"  - Visualization: {CONFIG['output']['visualization']}")

if __name__ == "__main__":
    main()
