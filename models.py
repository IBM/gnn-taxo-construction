# Model
from torch.nn import functional as F
from global_config import Config
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch


# Graph Neural Networks
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num_entities, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        self.num_entities = num_entities
        self.weight_adj = Parameter(torch.FloatTensor(num_entities, num_entities))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_adj.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# Graph2Taxo
class GRAPH2TAXO(torch.nn.Module):
    """
    Graph2Taxo model
    """
    def __init__(self, num_entities, num_relations):
        super(GRAPH2TAXO, self).__init__()
        self.num_entities = num_entities
        torch.manual_seed(Config.random_seed)
        self.emb_e = torch.nn.Embedding(num_entities, Config.init_emb_size, padding_idx=0)

        self.gc0 = GraphConvolution(Config.init_emb_size, Config.init_emb_size, num_entities)
        self.gc1 = GraphConvolution(Config.init_emb_size, Config.gc1_emb_size, num_entities)
        self.gc2 = GraphConvolution(Config.gc1_emb_size, Config.gc2_emb_size, num_entities)
        self.gc3 = GraphConvolution(Config.gc1_emb_size, Config.embedding_dim, num_entities)

        self.conv1 = nn.Conv1d(1, Config.channels, Config.kernel_size, stride=1, padding=int(
            math.floor(Config.kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)

        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout_rate)
        self.feature_map_drop = torch.nn.Dropout(Config.dropout_rate)

        self.fc = torch.nn.Linear((Config.embedding_dim+8) * Config.channels, Config.embedding_dim)
        self.fc2 = torch.nn.Linear((Config.embedding_dim + Config.gc1_emb_size)*2 , Config.embedding_dim)
        self.fc3 = torch.nn.Linear(Config.embedding_dim, 1)
        self.fc4 = torch.nn.Linear(300 * 2, Config.embedding_dim)
        self.fc_gcn = torch.nn.Linear(Config.embedding_dim, 1)
        self.fc_com = torch.nn.Linear(2, 1)
        self.fc_dag = torch.nn.Linear(Config.embedding_dim + Config.gc1_emb_size, 1)

        self.bn_init = torch.nn.BatchNorm1d(Config.init_emb_size)
        self.bn0 = torch.nn.BatchNorm1d(Config.embedding_dim + 8)
        self.bn1 = torch.nn.BatchNorm1d(Config.channels)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn5 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn6 = torch.nn.BatchNorm1d(Config.embedding_dim + Config.gc1_emb_size)
        self.bn7 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn8 = torch.nn.BatchNorm1d(16)
        self.bn9 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_word = torch.nn.BatchNorm1d(300)
        self.bn_w = torch.nn.BatchNorm1d(num_entities)
        self.bn_edge = torch.nn.BatchNorm1d(3)

        self.loss = torch.nn.BCELoss()


    def forward(self, e1, e2, rel, X, adjs, terms, e1_index, e2_index, word_embs, fre, degree, substr):

        # Use the random initialized embeddings
        emb_initial = self.emb_e(X)
        emb_initial = F.relu(emb_initial)  # option
        emb_initial = self.inp_drop(emb_initial)  # option

        x = self.gc0(emb_initial, adjs)
        x = self.bn_init(x)
        x = F.relu(x)
        x = self.inp_drop(x)  # option

        x = self.gc1(x, adjs)
        x = self.bn3(x)
        x = F.relu(x)

        s = self.gc2(x, adjs)
        s = F.softmax(s, dim=1)
        out = torch.mm(s.transpose(0, 1), x)
        out = F.relu(out)
        out = self.inp_drop(out)

        out_adj = torch.matmul(torch.matmul(s.transpose(0, 1), adjs), s)
        out = self.gc3(out, out_adj)
        out = F.relu(out)
        out = self.inp_drop(out)

        emb_dp = torch.matmul(s, out)
        emb_dp = F.relu(emb_dp)
        emb_dp = self.inp_drop(emb_dp)

        x = torch.cat([x, emb_dp], 1)
        x = self.bn6(x)

        e1_embedded = x[e1]
        e2_embedded = x[e2]
        x = torch.cat([e1_embedded, e2_embedded], 1)
        x = self.fc2(x)
        x = self.bn7(x)

        feas = torch.cat([fre, substr], 1)
        x = torch.cat([x, feas], 1)
        x = self.bn0(x)
        x = x.view(x.size()[0], 1, x.size()[1])
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = self.bn2(x)
        #x = F.relu(x)
        #x = self.feature_map_drop(x)
        x = self.fc3(x)
        pred = torch.sigmoid(x)

        return pred