import json
import re 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Configuration
CONFIG = {
    "model_params_path": "results/MOC/RF_MOC_best_model_params.json",
    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/Y_train.csv"
    },
    "output": {
        "feature_importance": "results/feature/per_model/RF_MOC_gini_importance.csv",
        "plot": "results/feature/per_model/RF_MOC_gini_top20.png"
    }
}

def load_model_components():
    """Load model parameters and thresholds"""
    with open(CONFIG['model_params_path'], 'r') as f:
        model_params = json.load(f)
    return model_params
    
def initialize_model(params):
    """Create MultiOutputClassifier with optimized parameters"""
    base_model = RandomForestClassifier(
        criterion='gini',
        random_state=42
    )
    base_model.set_params(**params)
    return MultiOutputClassifier(base_model)

def calculate_feature_importance(model, X, feature_names):
    """
    Calculate equally-weighted feature importance.
    """
    num_labels = len(model.estimators_)
    importance_matrix = np.zeros((num_labels, len(feature_names)))

    for label_idx, estimator in enumerate(model.estimators_):
        # Get raw feature importances
        importance_matrix[label_idx] = estimator.feature_importances_ 

    # Aggregate equally across all labels
    global_importance = importance_matrix.sum(axis=0)
    return pd.DataFrame({
        'feature': feature_names,
        'importance': global_importance / global_importance.max()  # Normalize
    }).sort_values('importance', ascending=False)

def visualize_top_features(df, top_n=20):
    """Generate horizontal bar plot of feature importance"""
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('importance', ascending=True)
    colors = plt.cm.Purples(np.linspace(0.3, 0.9, len(plot_df)))

    bars = plt.barh(plot_df['feature'], plot_df['importance'],
                    color=colors, height=0.7)
    plt.xlabel('Normalized Gini Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Features - Multi-Output Classification(RF)', fontsize=14)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', va='center', fontsize=9)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG['output']['plot'], dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load model components
    model_params = load_model_components()
    X_train = pd.read_csv(CONFIG['data_paths']['X_train'])
    y_train = pd.read_csv(CONFIG['data_paths']['y_train'])
    X_train.columns = [re.sub(r'[^\w]', '_', col) for col in X_train.columns]

    # Train model
    model = initialize_model(model_params)
    model.fit(X_train, y_train)

    # Compute feature importance
    importance_df = calculate_feature_importance(
        model, X_train.values, X_train.columns.tolist()
    )
    # Save and plot
    importance_df.head(20).to_csv(CONFIG['output']['feature_importance'], index=False)
    visualize_top_features(importance_df)

    print(f"Analysis completed successfully.\n"
          f"Feature importance saved to: {CONFIG['output']['feature_importance']}\n"
          f"Plot saved to: {CONFIG['output']['plot']}")

if __name__ == "__main__":
    main()
