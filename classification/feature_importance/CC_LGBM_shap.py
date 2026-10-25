
import json
import re
import os  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lightgbm import LGBMClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.utils import resample

# Configuration parameters
CONFIG = {
    # Model configuration
    "model_params_path": "results/CC/LGBM_CC_best_model_params.json",
    
    # Data paths
    "data_paths": {
        "X_train": "data/X_train.csv",
        "y_train": "data/Y_train.csv"
    },
    
    # Label structure
    "label_order": [
        'Data Breach',
        'IT Error',
        'Extortion/Fraud',
        'Other',
        'Privacy Violation',
    ],
    
    # Output configuration
    "output": {
        "feature_importance": "results/feature/per_model/LGBM_CC_shap_importance.csv",
        "visualization": "results/feature/per_model/LGBM_CC_shap_top20.png"
    },
    
    # Computational parameters
    "shap_params": {
        "sampling": {
            "enable": True,
            "n_samples": 5000,
            "random_state": 42,
            "stratify": True 
        },
        "normalization": "max"  # Options: max, sum, none
    },
    
    "random_state": 42
}


def load_model_parameters():
    """Load trained model parameters from JSON file"""
    with open(CONFIG['model_params_path'], 'r') as f:
        return json.load(f)


def initialize_model(params):
    """Create and configure ClassifierChain model"""
    base_model = LGBMClassifier(
        verbosity=-1,
        random_state=CONFIG['random_state']
    )
    base_model.set_params(**params)
    return ClassifierChain(
        base_model,
        order=list(range(len(CONFIG["label_order"]))),
        random_state=CONFIG['random_state']
    )


def calculate_shap_importance(model, X, feature_names):
    """
    Calculate normalized SHAP feature importance with efficient sampling
    Returns: DataFrame sorted by importance score
    """
    # Data sampling implementation
    if CONFIG['shap_params']['sampling']['enable']:
        X = resample(
            X,
            n_samples=CONFIG['shap_params']['sampling']['n_samples'],
            random_state=CONFIG['shap_params']['sampling']['random_state']
        )
    
    num_original = len(feature_names)
    total_importance = np.zeros(num_original)
    
    # Create working copy with original features
    X_work = pd.DataFrame(X, columns=feature_names.copy())
    
    for label_idx, estimator in enumerate(model.estimators_):
        # Use all current features including chain features
        X_temp = X_work.copy()
        
        # SHAP value calculation
        explainer = shap.TreeExplainer(estimator)
        shap_output = explainer.shap_values(X_temp)
        
        # Handle SHAP output format
        if isinstance(shap_output, list):
            # For binary classification: use positive class (index 1)
            shap_values = shap_output[1]
        else:
            # For regression/multiclass
            shap_values = shap_output
        
        # Ensure correct dimensions
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(-1, X_temp.shape[1])
        
        # Extract SHAP values for original features only
        original_shap = shap_values[:, :num_original]
        label_importance = np.abs(original_shap).mean(axis=0)
        total_importance += label_importance
        
        # Add chain feature to working copy
        X_work[f'_chain_{label_idx}'] = estimator.predict_proba(X_temp)[:, 1]
    
    # Normalization logic
    if CONFIG['shap_params']['normalization'] == 'max':
        importance_scores = total_importance / total_importance.max()
    elif CONFIG['shap_params']['normalization'] == 'sum':
        importance_scores = total_importance / total_importance.sum()
    else:
        importance_scores = total_importance
    
    return pd.DataFrame({
        'feature': feature_names,
        'shap_importance': importance_scores
    }).sort_values('shap_importance', ascending=False)


def generate_importance_plot(df, top_n=20):
    """Generate publication-quality feature importance visualization"""
    plt.figure(figsize=(12, 8))
    plot_df = df.head(top_n).sort_values('shap_importance', ascending=True)
    
    if any(not isinstance(f, str) for f in plot_df['feature']):
        raise ValueError("All feature names must be strings.")
    
    colors = plt.cm.Greys(np.linspace(0.3, 0.9, len(plot_df)))
    bars = plt.barh(plot_df['feature'], plot_df['shap_importance'],
                    color=colors, height=0.7)
    
    plt.xlabel('Normalized SHAP Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Features - Classifier Chain (LightGBM)', fontsize=14)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', va='center', fontsize=9)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Ensure directory exists before saving
    output_path = CONFIG['output']['visualization']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main execution pipeline"""
    # Load and prepare data
    model_params = load_model_parameters()
    X_train = pd.read_csv(CONFIG['data_paths']['X_train'])
    y_train = pd.read_csv(CONFIG['data_paths']['y_train'])[CONFIG['label_order']]
    
    # Clean feature names
    X_train.columns = [re.sub(r'[^\w]', '_', col) for col in X_train.columns]
    feature_names = X_train.columns.tolist()
    
    # Model training
    model = initialize_model(model_params)
    model.fit(X_train.values, y_train.values)
    
    # SHAP analysis
    importance_df = calculate_shap_importance(
        model, 
        X_train.values, 
        feature_names
    )
    
    # Save results
    os.makedirs(os.path.dirname(CONFIG['output']['feature_importance']), exist_ok=True)
    importance_df.to_csv(CONFIG['output']['feature_importance'], index=False)
    generate_importance_plot(importance_df)
    
    # Console output
    print("\nSHAP analysis completed successfully")
    print(f"-> Feature importance saved to: {CONFIG['output']['feature_importance']}")
    print(f"-> Visualization saved to: {CONFIG['output']['visualization']}")


if __name__ == "__main__":
    main()

