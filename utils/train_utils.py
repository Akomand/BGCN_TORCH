import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import math
import numpy as np

"""
Learning rate adjustment used for CondenseNet model training
"""
def adjust_learning_rate(optimizer, epoch, config, batch=None, nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = config.max_epoch * nBatch
        T_cur = (epoch % config.max_epoch) * nBatch + batch
        lr = 0.5 * config.learning_rate * (1 + math.cos(math.pi * T_cur / T_total))
    else:
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = config.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def build_node_neighbor_dict(data):
  edge_from, edge_to = data.edge_index.to('cpu')

  # Number of nodes
  N = data.x.shape[0]
  true_vec = torch.ones(edge_from.shape)
  false_vec = true_vec - 1
  node_neighbor_dict = {}

  for i in range(N):
    mask = torch.where(edge_from == i, true_vec, false_vec).bool()
    node_neighbor_dict[i] = edge_to[mask]

  return node_neighbor_dict



def data_partition_random(data, label_n_per_class):
    test_set_n = 1000
    val_set_n = 500

    # THE WHOLE SET
    # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, one_hot_labels, order = load_data(dataset_name)

    N = len(data.y)  # the sample number (LENGTH OF TARGET VECTOR)
    # print(N)
    K = data.y.max() + 1  # the class number
    # print(K)

    labels = data.y.cpu()  # labels vector

    train_index_new = np.zeros(K * label_n_per_class).astype(int)  # total number of training samples (1D vector)
    train_mask_new = np.zeros(N).astype(bool)  # N-dimensional training mask
    val_mask_new = np.zeros(N).astype(bool)  # N-dimensional validation mask
    test_mask_new = np.zeros(N).astype(bool)  # N-dimensional testing mask

    y_train_new = np.zeros((N, K))  # N TARGET VALUES EACH K DIMENSIONAL (NUM CLASSES)
    y_val_new = np.zeros((N, K))  #
    y_test_new = np.zeros((N, K))

    # BUILDING TRAIN, TEST, AND VALIDATION SETS

    # CONSISTS OF A CLASS MAPPED TO ARRAY OF INDICES THAT HAVE SAME CLASS (NODES THAT HAVE SAME CLASS)
    class_index_dict = {}

    for i in range(K):
        # RETURNING THE ARRAY (EACH TIME CALLED) OF INDICES HAVING CLASS i
        # if array([1, 1, 2, 3, 4, 5])
        # return np.where(labels==1)[0] --> array([0, 1])
        class_index_dict[i] = np.where(labels == i)[
            0]  # where() means if condition true pick x value else pick y value. same dim as arr.

    for i in range(K):
        class_index = class_index_dict[i]
        # RANDOMLY SAMPLE 5 (label n per class) INDICES from class i
        train_index_one_class = np.random.choice(class_index, label_n_per_class, replace=False)
        print("The training set index for class {} is {}".format(i, train_index_one_class))

        # SLICES OF 5 (TAKES RANDOMLY SAMPLED INDICES (FROM CLASS i) AND PLACES THEM INTO total training samples vector)
        train_index_new[i * label_n_per_class:(i + 1) * label_n_per_class] = train_index_one_class

    train_index_new = list(train_index_new)
    # print(train_index_new)

    # SET DIFFERENCE (REMOVE THE 5-SAMPLED ARRAY INDICES WE JUST CREATED)
    test_val_potential_index = list(set([i for i in range(N)]) - set(train_index_new))
    # print(len(test_val_potential_index))
    # FROM TEST SET ABOVE, RANDOMY SAMPLE 1000 INDICES
    test_index_new = np.random.choice(test_val_potential_index, test_set_n, replace=False)

    # SET DIFFERENCE (REMOVE THE 1000-SAMPLED TEST ARRAY INDICES WE JUST CREATED)
    potential_val_index = list(set(test_val_potential_index) - set(test_index_new))
    # print(len(potential_val_index))
    # FROM TEST SET ABOVE, RANDOMY SAMPLE 500 INDICES
    val_index_new = np.random.choice(potential_val_index, val_set_n, replace=False)

    # INDICATE WHICH INDICES ARE TRAIN SAMPLES
    train_mask_new[train_index_new] = True
    # INDICATE WHICH INDICES ARE VAL SAMPLES
    val_mask_new[val_index_new] = True
    # INDICATE WHICH INDICES ARE TEST SAMPLES
    test_mask_new[test_index_new] = True

    print(len(train_index_new))
    # Making new one-hot vectors (partition)
    for i in train_index_new:
        y_train_new[i][labels[i]] = 1

    for i in val_index_new:
        y_val_new[i][labels[i]] = 1

    for i in test_index_new:
        y_test_new[i][labels[i]] = 1

    return train_mask_new, val_mask_new, test_mask_new




