import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_elliptic_data(data_dir, scale_features=True):
    """
    Loads the Elliptic Bitcoin Dataset, constructs a PyTorch Geometric Data graph,
    creates stratified train/val/test splits on labeled nodes, scales features,
    and calculates class weights.
    
    Args:
        data_dir (str): Directory containing the Elliptic CSV files.
        scale_features (bool): If True, normalizes features using StandardScaler.
        
    Returns:
        data (torch_geometric.data.Data): Graph data object.
        pos_weight (torch.Tensor): Class weight for BCE loss.
    """
    classes_path = os.path.join(data_dir, 'elliptic_txs_classes.csv')
    edgelist_path = os.path.join(data_dir, 'elliptic_txs_edgelist.csv')
    features_path = os.path.join(data_dir, 'elliptic_txs_features.csv')
    
    if not all(os.path.exists(p) for p in [classes_path, edgelist_path, features_path]):
        raise FileNotFoundError(
            f"Could not find all required dataset files in {data_dir}. "
            "Please ensure elliptic_txs_classes.csv, elliptic_txs_edgelist.csv, "
            "and elliptic_txs_features.csv are present."
        )
        
    print("Loading classes...")
    df_classes = pd.read_csv(classes_path)
    
    print("Loading features...")
    # Raw features file does not have a header row
    df_features = pd.read_csv(features_path, header=None)
    # Give column names: Column 0 is txId, Column 1 is time_step, rest are features
    df_features.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(df_features.shape[1] - 2)]
    
    print("Loading edgelist...")
    df_edgelist = pd.read_csv(edgelist_path)
    
    # 1. Map transaction IDs to contiguous 0-based indices
    tx_ids = df_features['txId'].unique()
    tx_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}
    
    # 2. Merge features and classes to ensure consistent alignment
    df_merged = pd.merge(df_features, df_classes, on='txId', how='left')
    
    # Map labels: '1' -> illicit (1.0), '2' -> licit (0.0), 'unknown' -> -1
    label_map = {'1': 1, '2': 0, 'unknown': -1}
    df_merged['label'] = df_merged['class'].map(label_map).fillna(-1).astype(int)
    
    # Extracted feature columns (all columns except txId and class)
    # This includes the time_step column as the first feature, followed by the 165 features (total 166)
    feature_cols = [col for col in df_merged.columns if col not in ['txId', 'class', 'label']]
    X_data = df_merged[feature_cols].values.astype(np.float32)
    y_data = df_merged['label'].values
    
    # 3. Map edgelist using node mapping
    # Filter edgelist to ensure nodes exist in feature mapping
    valid_edges = df_edgelist[df_edgelist['txId1'].isin(tx_to_idx) & df_edgelist['txId2'].isin(tx_to_idx)]
    
    edge_index_src = valid_edges['txId1'].map(tx_to_idx).values
    edge_index_dst = valid_edges['txId2'].map(tx_to_idx).values
    
    # PyG expects long tensors of shape (2, num_edges)
    edge_index = np.vstack([edge_index_src, edge_index_dst])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # 4. Stratified Split (60% Train / 20% Val / 20% Test) on labeled nodes only
    labeled_indices = np.where(y_data != -1)[0]
    labeled_labels = y_data[labeled_indices]
    
    # Split labeled nodes into 60% train and 40% temp
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        labeled_indices,
        labeled_labels,
        test_size=0.40,
        random_state=42,
        stratify=labeled_labels
    )
    
    # Split temp into 50% val and 50% test (each representing 20% of the total)
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )
    
    # 5. Standardize Features to prevent leakage and speed up training
    if scale_features:
        scaler = StandardScaler()
        # Fit on training features only
        scaler.fit(X_data[train_idx])
        X_data = scaler.transform(X_data)
        
    x = torch.tensor(X_data, dtype=torch.float)
    y = torch.tensor(y_data, dtype=torch.float)  # BCE loss requires target to be float
    
    # 6. Initialize masks
    num_nodes = len(tx_ids)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # 7. Compute class weights to counter imbalance
    num_licit_train = np.sum(y_train == 0)
    num_illicit_train = np.sum(y_train == 1)
    
    # pos_weight = negative_samples / positive_samples
    pos_weight_val = num_licit_train / max(1, num_illicit_train)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    print(f"Data loading completed:")
    print(f"  Graph Structure: {num_nodes} nodes, {edge_index.size(1)} edges")
    print(f"  Labeled nodes: {len(labeled_indices)} (Licit [0]: {np.sum(labeled_labels == 0)}, Illicit [1]: {np.sum(labeled_labels == 1)})")
    print(f"  Train nodes: {train_mask.sum().item()} (Licit: {np.sum(y_train == 0)}, Illicit: {np.sum(y_train == 1)})")
    print(f"  Val nodes: {val_mask.sum().item()} (Licit: {np.sum(y_val == 0)}, Illicit: {np.sum(y_val == 1)})")
    print(f"  Test nodes: {test_mask.sum().item()} (Licit: {np.sum(y_test == 0)}, Illicit: {np.sum(y_test == 1)})")
    print(f"  Class Imbalance Ratio (Licit/Illicit): {pos_weight_val:.4f}")
    
    return data, pos_weight

if __name__ == '__main__':
    # Simple check if run directly
    import sys
    if len(sys.argv) > 1:
        load_elliptic_data(sys.argv[1])
    else:
        print("Please run this script passing the dataset path as parameter.")
