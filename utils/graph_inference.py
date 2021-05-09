import numpy as np
import random
import torch



def sample_graph_copying(seed, node_neighbors_dict, labels, epsilon=0.01, set_seed=False):
    # set seed for random choices
    if set_seed:
        np.random.seed(seed)  # decide the seed for graph inference
        random.seed(seed)

    labels = labels.cpu().numpy()
    # N is number of nodes
    N = len(labels)

    # K is number of classes
    K = np.max(labels) + 1
    # print('K', K)
    # build the labels_dict such that labels_dict[0] is a list of all node indices
    # which are in class 0.
    labels_dict = {}
    for k in range(K):
        labels_dict[k] = np.where(labels == k)[0]

    # print('labels_dict', labels_dict)

    # for each of the nodes we have
    for i in range(N):
        if random.uniform(0, 1) < 1 - epsilon:
            sampled_node = np.random.choice(labels_dict[labels[i]], 1)[0]
        else:
            # if roll fails, just use node i
            sampled_node = i





        # sampled_node = i
        #
        # for t in range(5):
        #     deg_node_i = len(node_neighbors_dict[i])
        #     # deg_node_i = len(labels_dict[labels[i]])
        #     sample_node = np.random.choice(labels_dict[labels[i]], 1)[0]
        #     # sample_node = np.random.choice(labels_dict[labels[i]], 1)[0]
        #     deg_node_w = len(node_neighbors_dict[sample_node])
        #     # deg_node_w = len(labels_dict[labels[sample_node]])
        #
        #     # pick uniformly in (0,1) if less than 1-e, pick a node
        #     if random.uniform(0, 1) <= min(1, float(deg_node_i) / float(deg_node_w)):
        #     # if random.uniform(0, 1) < 1 - epsilon:
        #         # if roll succeeds, pick 1 node at random with the same class that
        #         # node i has.
        #         sampled_node = sample_node
        #         # sampled_node = np.random.choice(labels_dict[labels[i]], 1)[0]
        #
        #     else:
        #         # if roll fails, just use node i
        #         sampled_node = i








        # print('i', i, 'sampled_node', sampled_node)
        # get row of adj matrix for the node (sampled_node) that we just picked out
        row_index_i = node_neighbors_dict[sampled_node]
        # print('row_index_i', row_index_i)
        # construct a row of adj mat rix containing i for each element
        col_index_i = i * np.ones(len(row_index_i))
        col_index_i = col_index_i.astype(int)
        # print('col_index_i', col_index_i)
        # build the Adj matrix by concatenating row and col indices.
        if i == 0:
            row_index = row_index_i
            col_index = col_index_i
        else:
            row_index = np.concatenate((row_index, row_index_i), axis=0)
            col_index = np.concatenate((col_index, col_index_i), axis=0)

    # print('row_index', row_index)
    # print('col_index', col_index)
    link_index_row = np.concatenate((row_index, col_index), axis=0)
    link_index_col = np.concatenate((col_index, row_index), axis=0)
    # data length is 2N
    data = np.ones(len(link_index_row))
    # print('link_index_row', link_index_row)
    # print('link_index_col', link_index_col)
    # print('data', data)
    # sampled_graph_dense = np.zeros((N, N))
    # sampled_graph_dense[link_index_row, link_index_col] = 1
    # plt.imshow(sampled_graph_dense[np.ix_(order, order)])
    # plt.colorbar()
    # plt.show()
    # plt.show(block=False)
    # time.sleep(5)
    # plt.close('all')
    new_edge_index = torch.vstack((torch.tensor(link_index_row), torch.tensor(link_index_col)))
    return new_edge_index