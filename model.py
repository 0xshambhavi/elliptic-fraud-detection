import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE model for node classification.
    Supports customizable input channels, hidden dimension, output channels,
    number of layers, and dropout rate.
    """
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=3, dropout=0.3):
        super(GraphSAGE, self).__init__()
        
        if num_layers < 2:
            raise ValueError("Number of layers must be at least 2.")
            
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        
        # Input SAGEConv layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        # Hidden SAGEConv layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        # Output SAGEConv layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x (Tensor): Node feature matrix of shape (num_nodes, in_channels)
            edge_index (LongTensor): Graph connectivity matrix of shape (2, num_edges)
            
        Returns:
            x (Tensor): Unnormalized logit values of shape (num_nodes, out_channels)
        """
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Last layer outputs logits (no activation function since BCEWithLogitsLoss is used)
        x = self.convs[-1](x, edge_index)
        return x
