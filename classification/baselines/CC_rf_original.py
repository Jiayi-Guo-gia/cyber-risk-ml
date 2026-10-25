import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    hamming_loss,
    f1_score,
    jaccard_score
)
from itertools import permutations
from tqdm import tqdm

def evaluate_chain_order(X_train, Y_train, X_test, Y_test, chain_order):
    """Evaluate a specific chain order and return metrics."""
    # Define the base LightGBM classifier
    rf_clf = RandomForestClassifier(
        n_estimators=100,        
        max_depth=8,           
        random_state=42,         
        n_jobs=-1,               
        class_weight='balanced'  
    )
    
    # Convert chain order to column indices
    label_to_index = {label: idx for idx, label in enumerate(Y_train.columns)}
    chain_order_indices = [label_to_index[label] for label in chain_order]
    
    # Wrap the base classifier in a ClassifierChain
    chain_clf = ClassifierChain(rf_clf, order=chain_order_indices)
    chain_clf.fit(X_train, Y_train)
    
    # Predict probabilities
    Y_pred_prob = chain_clf.predict_proba(X_test)
    Y_pred_cls = (Y_pred_prob > 0.5).astype(int)
    
    # Compute metrics
    metrics = {
        'jaccard': jaccard_score(Y_test, Y_pred_cls, average='samples'),
        'hamming': hamming_loss(Y_test, Y_pred_cls),
        'macro_f1': f1_score(Y_test, Y_pred_cls, average='macro'),
        'micro_f1': f1_score(Y_test, Y_pred_cls, average='micro'),
        'weighted_f1': f1_score(Y_test, Y_pred_cls, average='weighted'),
        'sample_f1': f1_score(Y_test, Y_pred_cls, average='samples')
    }
    
    return metrics

def get_best_order(results_df):
    """Get the chain order with the best Weighted F1 score."""
    best_order = results_df.sort_values(by='weighted_f1', ascending=False).iloc[0]
    return {
        'label_order': best_order['order'],
        'weighted_f1': best_order['weighted_f1'],
        'sample_f1': best_order['sample_f1'],
        'macro_f1': best_order['macro_f1'],
        'micro_f1': best_order['micro_f1'],
        'jaccard': best_order['jaccard'],
        'hamming': best_order['hamming']
    }

def get_top_orders(results_df, metric, ascending=False):
    """Get the top 5 chain orders for a specific metric."""
    return results_df.sort_values(by=metric, ascending=ascending).head(5)[['order', metric]]

def main():
    # Load data
    X_train = pd.read_csv("data/X_train.csv")
    Y_train = pd.read_csv("data/Y_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    Y_test = pd.read_csv("data/Y_test.csv")
    
    # Get all possible chain orders
    labels = Y_train.columns.tolist()
    print("Y_train columns:", labels)
    all_orders = list(permutations(labels))
    
    # Evaluate each chain order
    results = []
    for idx, order in enumerate(tqdm(all_orders, desc="Evaluating Chain Orders")):
        metrics = evaluate_chain_order(X_train, Y_train, X_test, Y_test, order)
        results.append({
            'order': order,
            **metrics,
            'order_index': idx + 1
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save all results
    results_dir = "results/rf/"
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, "chain_order_results.csv"), index=False)
    
    
    # Find and display best order
    best_order = get_best_order(results_df)
    print("\n=== Optimal Chain Order ===")
    print("Label Sequence:", " → ".join(best_order['label_order']))
    print("Performance Metrics:")
    print(f"- Weighted F1: {best_order['weighted_f1']:.4f}")
    print(f"- Sample F1: {best_order['sample_f1']:.4f}")
    print(f"- Macro F1: {best_order['macro_f1']:.4f}")
    print(f"- Micro F1: {best_order['micro_f1']:.4f}")
    print(f"- Jaccard: {best_order['jaccard']:.4f}")
    print(f"- Hamming Loss: {best_order['hamming']:.4f}")
    
    # Save top 5 orders for each metric
    metrics_config = {
        'jaccard': False,
        'hamming': True,   # Lower is better
        'macro_f1': False,
        'micro_f1': False,
        'weighted_f1': False,
        'sample_f1': False
    }
    
    all_top_orders = []  
    top_orders_data = []

    for metric, ascending in metrics_config.items():
        top_df = get_top_orders(results_df, metric, ascending=ascending)
        top_df.to_csv(os.path.join(results_dir, f"top_5_{metric}.csv"), index=False)
        
    # Collect all orders (preserving duplicates)
        for _, row in top_df.iterrows():
            order = list(row['order'])
            metric_value = row[metric]
            
            all_top_orders.append(order)
            top_orders_data.append({
                'order': order,
                'source_metric': metric,
                'metric_value': metric_value
            })

    # Save consolidated list with duplicates
    consolidated_df = pd.DataFrame(top_orders_data)
    consolidated_df.to_csv(os.path.join(results_dir, "all_top_orders.csv"), index=False)
    
    # Save the best order
    with open(os.path.join(results_dir, "best_chain_order.txt"), "w") as f:
        f.write(f"Best Chain Order (Weighted F1): {best_order['label_order']}\n")
        f.write("\nPerformance Metrics:\n")
        f.write(f"- Weighted F1: {best_order['weighted_f1']:.4f}\n")
        f.write(f"- Sample F1: {best_order['sample_f1']:.4f}\n")
        f.write(f"- Macro F1: {best_order['macro_f1']:.4f}\n")
        f.write(f"- Micro F1: {best_order['micro_f1']:.4f}\n")
        f.write(f"- Jaccard: {best_order['jaccard']:.4f}\n")
        f.write(f"- Hamming Loss: {best_order['hamming']:.4f}\n")
    
    # Plot performance metrics
    plot_performance(results_df, results_dir)

def plot_performance(results_df, results_dir):
    """Plot performance metrics for all chain orders."""
    plt.figure(figsize=(16, 12))
    
    metrics_config = [
        ('jaccard', 'Jaccard Score', 'b'),
        ('hamming', 'Hamming Loss', 'r'),
        ('macro_f1', 'Macro F1', 'g'),
        ('micro_f1', 'Micro F1', 'm'),
        ('weighted_f1', 'Weighted F1', 'c'),
        ('sample_f1', 'Sample F1', 'orange')
    ]
    
    for idx, (metric, title, color) in enumerate(metrics_config, 1):
        plt.subplot(2, 3, idx)
        plt.plot(results_df['order_index'], results_df[metric], 
                marker='o', linestyle='-', color=color)
        plt.title(title)
        plt.xlabel("Chain Order Index")
        plt.ylabel(title)
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "chain_performance_plot.png"))
    plt.close()

if __name__ == "__main__":
    main()