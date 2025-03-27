import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os

def load_data(ground_truth_path, prediction_paths):
    """
    Load ground truth and prediction CSV files
    
    Args:
        ground_truth_path (str): Path to ground truth CSV file
        prediction_paths (list): List of paths to prediction CSV files
    
    Returns:
        tuple: Ground truth DataFrame and list of prediction DataFrames
    """
    # Load ground truth
    gt_df = pd.read_csv(ground_truth_path)
    
    # Load predictions
    pred_dfs = []
    model_names = []
    for path in prediction_paths:
        pred_dfs.append(pd.read_csv(path))
        model_names.append(os.path.basename(path).split('.')[0])
    
    return gt_df, pred_dfs, model_names

def analyze_ground_truth(gt_df):
    """
    Analyze ground truth data distribution
    
    Args:
        gt_df (DataFrame): Ground truth DataFrame
    
    Returns:
        dict: Distribution statistics for each hand
    """
    stats = {}
    
    for hand in ['left_hand_gesture', 'right_hand_gesture']:
        total = len(gt_df)
        value_counts = gt_df[hand].value_counts(dropna=False)
        
        # Calculate percentages
        stats[hand] = {
            'total': total,
            'distribution': {
                label: {
                    'count': count,
                    'percentage': count / total * 100
                } for label, count in value_counts.items()
            }
        }
    
    return stats

def calculate_metrics(gt_df, pred_df):
    """
    Calculate classification metrics for predictions
    
    Args:
        gt_df (DataFrame): Ground truth DataFrame
        pred_df (DataFrame): Prediction DataFrame
    
    Returns:
        dict: Metrics for each hand
    """
    metrics = {}
    
    for hand in ['left_hand_gesture', 'right_hand_gesture']:
        y_true = gt_df[hand].fillna(-1)  # Replace None with -1 for comparison
        y_pred = pred_df[hand].fillna(-1)
        
        # Get unique classes (excluding None/-1)
        classes = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
        if -1 in classes:
            classes.remove(-1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes + [-1])
        
        # For binary classification, get TP, FP, TN, FN for each class
        class_metrics = {}
        
        for idx, cls in enumerate(classes):
            # Create binary classification (class vs rest)
            y_true_bin = (y_true == cls).astype(int)
            y_pred_bin = (y_pred == cls).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
            
            accuracy = accuracy_score(y_true_bin, y_pred_bin)
            precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            
            class_metrics[cls] = {
                'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        # Overall metrics (multiclass)
        overall_accuracy = accuracy_score(y_true, y_pred)
        overall_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        overall_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        overall_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics[hand] = {
            'per_class': class_metrics,
            'overall': {
                'accuracy': overall_accuracy,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1_score': overall_f1
            },
            'confusion_matrix': cm,
            'labels': classes + [-1]
        }
    
    return metrics

def create_summary_df(gt_stats, all_metrics, model_names):
    """
    Create a summary DataFrame of all metrics
    
    Args:
        gt_stats (dict): Ground truth statistics
        all_metrics (list): List of metrics for each model
        model_names (list): List of model names
    
    Returns:
        DataFrame: Summary DataFrame
    """
    rows = []
    
    # For each model
    for i, metrics in enumerate(all_metrics):
        row = {'model': model_names[i]}
        
        # For each hand
        for hand in ['left_hand_gesture', 'right_hand_gesture']:
            # Overall metrics
            for metric, value in metrics[hand]['overall'].items():
                row[f'{hand}_{metric}'] = value
            
            # Per-class metrics
            for cls, class_metrics in metrics[hand]['per_class'].items():
                for metric, value in class_metrics.items():
                    row[f'{hand}_class{cls}_{metric}'] = value
        
        rows.append(row)
    
    # Create DataFrame
    summary_df = pd.DataFrame(rows)
    return summary_df

def plot_confusion_matrices(all_metrics, model_names):
    """
    Plot confusion matrices for all models
    
    Args:
        all_metrics (list): List of metrics for each model
        model_names (list): List of model names
    """
    num_models = len(model_names)
    fig, axes = plt.subplots(2, num_models, figsize=(5*num_models, 10))
    
    for i, metrics in enumerate(all_metrics):
        for j, hand in enumerate(['left_hand_gesture', 'right_hand_gesture']):
            cm = metrics[hand]['confusion_matrix']
            labels = metrics[hand]['labels']
            
            if num_models == 1:
                ax = axes[j]
            else:
                ax = axes[j, i]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_title(f"{model_names[i]} - {hand.split('_')[0].capitalize()} Hand")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
    
    plt.tight_layout()
    return fig

def main(ground_truth_path, prediction_paths, output_path):
    """
    Main function to run the analysis
    
    Args:
        ground_truth_path (str): Path to ground truth CSV file
        prediction_paths (list): List of paths to prediction CSV files
        output_path (str): Path to save output CSV file
    """
    # Load data
    gt_df, pred_dfs, model_names = load_data(ground_truth_path, prediction_paths)
    
    # Analyze ground truth
    gt_stats = analyze_ground_truth(gt_df)
    
    # Calculate metrics for each model
    all_metrics = []
    for pred_df in pred_dfs:
        metrics = calculate_metrics(gt_df, pred_df)
        all_metrics.append(metrics)
    
    # Create summary DataFrame
    summary_df = create_summary_df(gt_stats, all_metrics, model_names)
    
    # Save to CSV
    summary_df.to_csv(output_path, index=False)
    
    # Create confusion matrix plots
    fig = plot_confusion_matrices(all_metrics, model_names)
    
    # Save figure
    fig.savefig(output_path.replace('.csv', '_confusion_matrices.png'))
    
    # Print ground truth stats
    print("Ground Truth Statistics:")
    for hand, stats in gt_stats.items():
        print(f"\n{hand.split('_')[0].capitalize()} Hand:")
        for label, info in stats['distribution'].items():
            label_str = "None" if pd.isna(label) else label
            print(f"  {label_str}: {info['count']} samples ({info['percentage']:.2f}%)")
    
    return summary_df, gt_stats, all_metrics

if __name__ == "__main__":
    # File paths (to be specified by the user)
    ground_truth_path = "kk_eval_video/groundtruth.csv"
    prediction_paths = [
        "kk_eval_video/keypoint6mar_dt_eval_flip.csv",
        "kk_eval_video/keypoint6mar_knn_eval_flip.csv",
        "kk_eval_video/keypoint6mar_lrc_eval_flip.csv",
        "kk_eval_video/keypoint6mar_nn_eval_flip.csv",
    ]
    output_path = "kk_eval_video/summary.csv"
    
    # Run the analysis
    summary_df, gt_stats, all_metrics = main(ground_truth_path, prediction_paths, output_path)
    
    # Additional visualization: Plot accuracy, precision, recall as bar charts
    plt.figure(figsize=(12, 8))
    
    model_names = ['dt', 'knn', 'lrc', 'nn']
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    hands = ['left_hand_gesture', 'right_hand_gesture']
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        
        # For each hand
        for j, hand in enumerate(hands):
            values = [metrics[hand]['overall'][metric] for metrics in all_metrics]
            x = np.arange(len(model_names)) + (j * 0.4 - 0.2)
            plt.bar(x, values, width=0.4, label=f"{hand.split('_')[0].capitalize()} Hand")
        
        plt.xticks(np.arange(len(model_names)), model_names, rotation=45, ha='right')
        plt.title(f"Overall {metric.capitalize()}")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.csv', '_metrics_comparison.png'))