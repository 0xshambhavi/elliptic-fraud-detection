import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from data_loader import load_elliptic_data
from model import GraphSAGE

def evaluate_pipeline(args):
    # 1. Load data
    print(f"Loading data from {args.data_dir}...")
    data, _ = load_elliptic_data(args.data_dir, scale_features=not args.no_scale)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move data to device
    data = data.to(device)
    
    # 2. Instantiate model and load best weights
    in_channels = data.x.size(1)
    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=args.hidden_dim,
        out_channels=1,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weight file {args.model_path} not found. Did you run train.py first?")
        
    print(f"Loading best model weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # 3. Model evaluation
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index).squeeze(-1)
        
        # Select test mask entries
        test_logits = logits[data.test_mask]
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds = (test_logits > 0).float().cpu().numpy()
        test_labels = data.y[data.test_mask].cpu().numpy()
        
    # 4. Compute metrics
    acc = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, zero_division=0)
    recall = recall_score(test_labels, test_preds, zero_division=0)
    f1 = f1_score(test_labels, test_preds, zero_division=0)
    auc_roc = roc_auc_score(test_labels, test_probs)
    
    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": float(auc_roc)
    }
    
    # Print metrics
    print("\n" + "="*40)
    print("           EVALUATION METRICS")
    print("="*40)
    print(f"  Accuracy:    {acc:.4%}")
    print(f"  Precision:   {precision:.4%}")
    print(f"  Recall:      {recall:.4%}")
    print(f"  F1-Score:    {f1:.4%}")
    print(f"  AUC-ROC:     {auc_roc:.4%}")
    print("="*40)
    
    # 5. Save metrics to metrics.json
    metrics_json_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_json_path}")
    
    # 6. Generate and save confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    
    # Create premium plot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    # Custom color map (vibrant/premium theme)
    cmap = plt.cm.Blues
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Licit (0)', 'Illicit (1)'])
    disp.plot(ax=ax, cmap=cmap, colorbar=False, values_format='d')
    
    # Visual improvements
    ax.set_title("Confusion Matrix: Fraud Detection (GraphSAGE)", fontsize=12, pad=15, fontweight='bold')
    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=10)
    ax.set_ylabel("True Label", fontsize=10, labelpad=10)
    
    # Annotate cells with percentages too
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm_normalized[i, j] > 0.5 else "black"
            ax.text(j, i + 0.15, f"({cm_normalized[i, j]:.1%})",
                    ha="center", va="center", color=text_color, fontsize=8)
            
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate GraphSAGE on Elliptic Bitcoin Dataset")
    parser.add_argument('--data_dir', type=str, default='./elliptic_bitcoin_dataset',
                        help='Directory containing the dataset CSV files')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GraphSAGE layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--model_path', type=str, default='best_model.pt',
                        help='Path to the trained model weights')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory to save metrics and confusion matrix')
    parser.add_argument('--no_scale', action='store_true',
                        help='Disable feature standardization')
                        
    args = parser.parse_args()
    evaluate_pipeline(args)
