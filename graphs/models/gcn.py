import torch.nn as nn
import torch.nn.functional as F
from graphs.models.custom_layers.graph_convolution import GraphConvolution
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, ChebConv, GraphConv, MessagePassing


class GCN(nn.Module):
    # def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dropout=0.5, **kwargs):
    #     super(GCN, self).__init__()
    #
    #     layers = []
    #     in_channels, out_channels = c_in, c_hidden
    #     for l_idx in range(num_layers - 1):
    #         layers += [
    #             GCNConv(in_channels=in_channels,
    #                       out_channels=out_channels,
    #                       **kwargs),
    #             nn.ReLU(inplace=True),
    #             nn.Dropout(dropout)
    #         ]
    #         in_channels = c_hidden
    #     layers += [GCNConv(in_channels=in_channels,
    #                          out_channels=c_out,
    #                          **kwargs)]
    #     self.layers = nn.ModuleList(layers)
    #
    #     # self.gc1 = GraphConv(nfeat, nhid)
    #     # self.gc2 = GraphConv(nhid, nclass)
    #     # self.dropout = dropout
    #
    # def forward(self, x, edge_index):
    #     for l in self.layers:
    #         # For graph layers, we need to add the "edge_index" tensor as additional input
    #         # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
    #         # we can simply check the class type.
    #         if isinstance(l, MessagePassing):
    #             x = l(x, edge_index)
    #         else:
    #             x = l(x)
    #     return x
    #     # x = F.relu(self.gc1(x, adj))
    #     # x = F.dropout(x, self.dropout, training=self.training)
    #     # x = self.gc2(x, adj)
    #     # return F.log_softmax(x, dim=1)

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dropout=0.5, **kwargs):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(c_in, c_hidden)
        self.conv2 = GraphConv(c_hidden, c_out)

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

        # softmax probabilities
        return F.log_softmax(x, dim=1)