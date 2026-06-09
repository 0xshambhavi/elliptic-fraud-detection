import os
import argparse
import torch
import numpy as np
from sklearn.metrics import f1_score
from data_loader import load_elliptic_data
from model import GraphSAGE

def train_pipeline(args):
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 1. Load data
    print(f"Loading data from {args.data_dir}...")
    data, pos_weight = load_elliptic_data(args.data_dir, scale_features=not args.no_scale)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move data and weights to device
    data = data.to(device)
    pos_weight = pos_weight.to(device)
    
    # 2. Instantiate model
    in_channels = data.x.size(1)
    print(f"Initializing GraphSAGE model (layers={args.num_layers}, hidden_dim={args.hidden_dim}, dropout={args.dropout})...")
    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=args.hidden_dim,
        out_channels=1,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # 3. Setup Optimizer and Loss Criterion
    # BCEWithLogitsLoss combines Sigmoid and BCE in one stable layer. pos_weight balances the loss.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_f1 = -1.0
    best_epoch = -1
    
    print("\nStarting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        # Training step
        model.train()
        optimizer.zero_grad()
        
        logits = model(data.x, data.edge_index).squeeze(-1)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index).squeeze(-1)
            val_loss = criterion(val_logits[data.val_mask], data.y[data.val_mask])
            
            # Predict binary class using 0.5 probability threshold (logit > 0)
            val_preds = (val_logits[data.val_mask] > 0).float().cpu().numpy()
            val_labels = data.y[data.val_mask].cpu().numpy()
            
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
            
        # Log training progress every few epochs or every epoch if small
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val F1-score: {val_f1:.4f}")
            
        # Save best model based on validation F1 score (standard for fraud detection with high imbalance)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), args.model_path)
            
    print(f"\nTraining completed! Best Validation F1-score: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Best model state saved to {args.model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train GraphSAGE on Elliptic Bitcoin Dataset")
    parser.add_argument('--data_dir', type=str, default='./elliptic_bitcoin_dataset',
                        help='Directory containing the dataset CSV files')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GraphSAGE layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='L2 regularization/weight decay')
    parser.add_argument('--model_path', type=str, default='best_model.pt',
                        help='Path to save the best model weights')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_scale', action='store_true',
                        help='Disable feature standardization')
                        
    args = parser.parse_args()
    train_pipeline(args)
