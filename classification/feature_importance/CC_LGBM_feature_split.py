import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.multioutput import ClassifierChain

# Configuration
CONFIG = {
    "model_params_path": "results/CC/LGBM_CC_best_model_params.json",
    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/Y_train.csv"
    },
    "label_order": [
        'Data Breach',
        'IT Error',
        'Extortion/Fraud',
        'Other',
        'Privacy Violation',
    ],
    "output": {
        "feature_importance": "results/feature/LGBM_CC_top20_features_split.csv",
        "plot": "results/feature/LGBM_CC_feature_importance_plot_split.png"
    },
    "random_state": 42
}

def load_model_components():
    """Load model parameters and thresholds from JSON files"""
    with open(CONFIG['model_params_path'], 'r') as f:
        model_params = json.load(f)
    return model_params

def initialize_model(params):
    """Create ClassifierChain using LightGBM base model"""
    params['importance_type'] = 'split'  # Explicitly set split importance
    base_model = LGBMClassifier(
        verbosity=-1,
        random_state=CONFIG["random_state"]
    )
    base_model.set_params(**params)
    return ClassifierChain(
        base_model,
        order=list(range(len(CONFIG["label_order"]))),
        random_state=CONFIG["random_state"]
    )

def calculate_feature_importance(model, X, original_features):
    """Calculate importance of original features across label chain"""
    num_labels = len(model.estimators_)
    num_features = len(original_features)
    importance_matrix = np.zeros((num_labels, num_features))
    
    # Maintain chain state for feature augmentation
    X_augmented = X.copy()
    
    for label_idx, estimator in enumerate(model.estimators_):
        # Get all feature importances
        all_importances = estimator.feature_importances_
        
        # Extract original features (first N elements)
        original_importances = all_importances[:num_features]
        importance_matrix[label_idx] = original_importances
        
        # Add chain features for subsequent estimators
        if label_idx < num_labels - 1:
            probas = estimator.predict_proba(X_augmented)[:, 1]
            X_augmented = np.hstack([X_augmented, probas.reshape(-1, 1)])
    
    # Aggregate and normalize
    global_importance = importance_matrix.sum(axis=0)
    return pd.DataFrame({
        'feature': original_features,
        'importance': global_importance / global_importance.max()
    }).sort_values('importance', ascending=False)

def visualize_top_features(df, top_n=20):
    """Draw horizontal bar plot of top-N features"""
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('importance', ascending=True)
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(plot_df)))
    bars = plt.barh(plot_df['feature'], plot_df['importance'], 
                    color=colors, height=0.7)
    
    plt.xlabel('Normalized Split Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Features - Classifier Chain (LightGBM)', fontsize=14)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', va='center', fontsize=9)
    
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG['output']['plot'], dpi=300, bbox_inches='tight')
    plt.close()

def main():
    model_params=load_model_components()
    
    # Load and prepare data
    X_train = pd.read_csv(CONFIG['data_paths']['X_train'])
    y_train = pd.read_csv(CONFIG['data_paths']['y_train'])[CONFIG['label_order']]
    
    # Sanitize feature names
    X_train.columns = [re.sub(r'[^\w]', '_', col) for col in X_train.columns]
    original_features = X_train.columns.tolist()
    
    # Train model
    model = initialize_model(model_params)
    model.fit(X_train.values, y_train.values)
    
    # Compute and save feature importance
    importance_df = calculate_feature_importance(
        model, X_train.values, original_features
    )
    importance_df.to_csv(CONFIG['output']['feature_importance'], index=False)
    
    # Generate visualization
    visualize_top_features(importance_df)
    
    print(f"Analysis completed:\n"
          f"- Feature importance: {CONFIG['output']['feature_importance']}\n"
          f"- Visualization: {CONFIG['output']['plot']}")

if __name__ == "__main__":
    main()