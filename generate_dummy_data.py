import os
import argparse
import pandas as pd
import numpy as np

def generate_dummy_data(output_dir, num_nodes=1000, num_edges=1500, num_features=165):
    """
    Generates a synthetic dataset mimicking the structure of the Elliptic Bitcoin Dataset.
    
    Args:
        output_dir (str): Directory where the CSV files should be saved.
        num_nodes (int): Number of unique transaction nodes.
        num_edges (int): Number of transaction edges.
        num_features (int): Number of feature dimensions (excluding txId and time_step).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating dummy Elliptic dataset inside '{output_dir}'...")
    
    # 1. Generate unique transaction IDs (txIds)
    # Generate distinct, high-value transaction IDs similar to raw data
    np.random.seed(42)
    tx_ids = np.random.choice(np.arange(10000000, 99999999), size=num_nodes, replace=False)
    
    # Assign time steps (1 to 10)
    time_steps = np.random.randint(1, 11, size=num_nodes)
    
    # 2. Generate classes: '1' (illicit), '2' (licit), 'unknown'
    # Real dataset has ~10% illicit, ~40% licit, ~50% unknown among transactions
    class_probs = [0.10, 0.40, 0.50]
    class_choices = ['1', '2', 'unknown']
    classes = np.random.choice(class_choices, size=num_nodes, p=class_probs)
    
    # Save classes
    df_classes = pd.DataFrame({
        'txId': tx_ids,
        'class': classes
    })
    classes_path = os.path.join(output_dir, 'elliptic_txs_classes.csv')
    df_classes.to_csv(classes_path, index=False)
    print(f"  Saved classes to {classes_path}")
    
    # 3. Generate Features
    # Column 0: txId, Column 1: time_step, Columns 2-166: 165 features
    features_matrix = np.random.normal(loc=0.0, scale=1.0, size=(num_nodes, num_features))
    df_features = pd.DataFrame(features_matrix)
    # Insert txId and time_step at the beginning
    df_features.insert(0, 'time_step', time_steps)
    df_features.insert(0, 'txId', tx_ids)
    
    features_path = os.path.join(output_dir, 'elliptic_txs_features.csv')
    # Save WITHOUT headers, matching the raw dataset format
    df_features.to_csv(features_path, header=False, index=False)
    print(f"  Saved features to {features_path}")
    
    # 4. Generate Edgelist
    # Select source and target indices randomly from existing nodes
    txId1_indices = np.random.randint(0, num_nodes, size=num_edges)
    txId2_indices = np.random.randint(0, num_nodes, size=num_edges)
    
    # In transaction graphs, edges represent BTC flow and usually connect
    # nodes within the same time step or close time steps. For synthetic data,
    # we can connect nodes at random, but we will ensure source node is mapped.
    txId1 = tx_ids[txId1_indices]
    txId2 = tx_ids[txId2_indices]
    
    df_edgelist = pd.DataFrame({
        'txId1': txId1,
        'txId2': txId2
    })
    
    # Remove self-loops for cleanliness
    df_edgelist = df_edgelist[df_edgelist['txId1'] != df_edgelist['txId2']]
    
    edgelist_path = os.path.join(output_dir, 'elliptic_txs_edgelist.csv')
    df_edgelist.to_csv(edgelist_path, index=False)
    print(f"  Saved edgelist to {edgelist_path}")
    print("Dummy dataset generation complete!\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic Elliptic Bitcoin Dataset")
    parser.add_argument('--output_dir', type=str, default='./dummy_elliptic_dataset',
                        help='Directory to save synthetic files')
    parser.add_argument('--nodes', type=int, default=1000,
                        help='Number of transaction nodes to generate')
    parser.add_argument('--edges', type=int, default=1500,
                        help='Number of edges to generate')
    args = parser.parse_args()
    generate_dummy_data(args.output_dir, num_nodes=args.nodes, num_edges=args.edges)
