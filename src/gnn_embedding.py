import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
class GNNModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNModule, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, input_dim)

        # self.conv1 = GATConv(input_dim, hidden_dim)
        # self.conv2 = GATConv(hidden_dim, input_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return self.conv1(x, edge_index)


def build_graph_from_coordinates(coords_df, k=5):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(coords_df)
    edges = knn.kneighbors(return_distance=False)

    source_nodes = torch.tensor([i for i in range(len(coords_df)) for _ in range(k)], dtype=torch.long)
    target_nodes = torch.tensor(edges.flatten(), dtype=torch.long)

    edge_index = torch.stack([source_nodes, target_nodes], dim=0)

    return edge_index
