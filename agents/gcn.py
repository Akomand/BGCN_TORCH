"""
BGCN Main agent
"""
import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn

from torch_geometric.data import DataLoader, Batch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import TUDataset

from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from agents.base import BaseAgent

from graphs.models.gcn import GCN
from datasets.planetoid import PlanetoidDataLoader

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics
from utils.train_utils import data_partition_random
from utils.graph_inference import sample_graph_copying
from utils.train_utils import build_node_neighbor_dict

cudnn.benchmark = True


class BGCNAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define dataset
        self.data = Planetoid(root='/tmp/Cora', name="Cora", split='full', transform=NormalizeFeatures())
        self.dataset = self.data[0]

        # define models
        self.model = GCN(self.data.num_node_features, self.config.hidden_dim, self.data.num_classes)

        # define loss
        self.loss = nn.NLLLoss()

        # define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = None

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        # self.model.load_state_dict(torch.load(file_name))
        # self.validate()
        pass

    def save_checkpoint(self, file_name, is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        # torch.save(self.model.state_dict(), file_name)
        pass

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
            # self.save_checkpoint(self.config.checkpoint_file)

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        # Semi-supervision (sample labels_per_class)
        train_m, val_m, test_m = data_partition_random(self.dataset, self.config.labels_per_class)
        self.dataset.train_mask = train_m
        self.dataset.val_mask = val_m
        self.dataset.test_mask = test_m


        node_neighbor_dict = build_node_neighbor_dict(self.dataset)
        self.dataset.original_graph_edge_index = self.dataset.edge_index

        for epoch in range(1, self.config.max_epoch + 1):
            if epoch == self.config.pre_train_epochs:
                # time to sample a graph...
                self.dataset.edge_index = sample_graph_copying(12345, node_neighbor_dict, self.dataset.y, set_seed=True)
            elif epoch > self.config.pre_train_epochs:
                self.dataset.edge_index = sample_graph_copying(12345, node_neighbor_dict, self.dataset.y, set_seed=False)
                # if epoch > self.config.weight_collection_epoch:
                #     # Get acc_OG_graph
                #
                #     # Get acc_sample_graph

            self.train_one_epoch()
            # self.validate()

            self.current_epoch += 1

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        val_losses = []
        # for batch_idx, (data, target) in enumerate(self.train_loader):
        data, target = self.dataset.x, self.dataset.y
        # print(self.dataset.x.shape)
        self.optimizer.zero_grad()
        output = self.model(data, self.dataset.edge_index)
        loss = F.nll_loss(output[self.dataset.train_mask], target[self.dataset.train_mask])
        loss.backward()
        self.optimizer.step()
        _, pred = output.max(dim=1)

        train_correct = int(pred[self.dataset.train_mask].eq(target[self.dataset.train_mask]).sum().item())
        val_loss = F.nll_loss(output[self.dataset.val_mask], target[self.dataset.val_mask])
        val_losses.append(val_loss)
        val_correct = int(pred[self.dataset.val_mask].eq(target[self.dataset.val_mask]).sum().item())
        if self.current_epoch % 10 == 9:
            print('Epoch', self.current_epoch + 1, ': train_loss', loss.item(), 'train_acc',
                  train_correct / int(self.dataset.train_mask.sum()),
                  'val loss', val_loss.item(), 'val_accuracy', val_correct / int(self.dataset.val_mask.sum()))


    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            # for data, target in self.test_loader:
            data, target = self.dataset.x, self.dataset.y
            # print(data.shape)
            # print(target.shape)
            output = self.model(data, self.dataset.edge_index)
            # print(output.shape)
            test_loss += F.nll_loss(output[self.dataset.test_mask], target[self.dataset.test_mask], size_average=False)  # sum up batch loss

            _, pred = output.max(dim=1)
            correct = int(pred[self.dataset.test_mask].eq(target[self.dataset.test_mask]).sum().item())
            acc = correct / int(self.dataset.test_mask.sum())

            print('Sample Graph Test Accuracy: {:.4f}'.format(acc))


            self.dataset = self.swap_graphs(self.dataset)

            data, target = self.dataset.x, self.dataset.y
            # print(data.shape)
            # print(target.shape)
            output = self.model(data, self.dataset.edge_index)
            # print(output.shape)
            test_loss += F.nll_loss(output[self.dataset.test_mask], target[self.dataset.test_mask],
                                    size_average=False)  # sum up batch loss

            _, pred = output.max(dim=1)
            correct = int(pred[self.dataset.test_mask].eq(target[self.dataset.test_mask]).sum().item())
            acc = correct / int(self.dataset.test_mask.sum())

            print('Original Graph Test Accuracy: {:.4f}'.format(acc))


    def swap_graphs(self, data):
        edge_index = data.edge_index
        data.edge_index = data.original_graph_edge_index
        data.original_graph_edge_index = edge_index
        return data

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
