import torch.nn as nn
import torch.nn.functional as F
from graphs.models.custom_layers.graph_convolution import GraphConvolution
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, ChebConv, GraphConv, MessagePassing


class GCN(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dropout=0.5, **kwargs):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(c_in, 64)
        self.conv2 = GraphConv(64, 16)
        # self.gat = GATConv(16, c_out, heads=2)

    def forward(self, x, edge_index):
        # x is features and edge_index is two lists [0,1,2,3] --> [4,5,6,7]
        # x, edge_index = data.x, data.edge_index

        # run through first conv layer

        x = self.conv1(x, edge_index)
        # apply activation
        x = F.relu(x)
        # apply dropout regularization
        x = F.dropout(x, p=0.5, training=self.training)
        # run through second conv layer
        x = self.conv2(x, edge_index)

        # x = F.dropout(x, p=0.5, training=self.training)
        #
        # x = self.gat(x, edge_index)

        # softmax probabilities
        return F.log_softmax(x, dim=1)