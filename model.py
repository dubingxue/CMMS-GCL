
from torch_geometric.nn import GCNConv,GATConv,GINConv,GINEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.utils import subgraph, k_hop_subgraph
import copy


# device = 'cuda'
def Subgraph(data, aug_ratio):
    data = copy.deepcopy(data)

    # return data
    x = data.x
    edge_index = data.edge_index

    sub_num = int(data.num_nodes * aug_ratio)
    idx_sub = torch.randint(0, data.num_nodes, (1,)).to(edge_index.device)
    last_idx = idx_sub
    diff = None

    for k in range(1, sub_num):
        keep_idx, _, _, _ = k_hop_subgraph(last_idx, 1, edge_index)
        # print("subgraph: {}, keep_idx size: {}".format(k, keep_idx.shape[0]) )
        if keep_idx.shape[0] == last_idx.shape[0] or keep_idx.shape[0] >= sub_num or k == sub_num - 1:
            combined = torch.cat((last_idx, keep_idx)).to(edge_index.device)
            uniques, counts = combined.unique(return_counts=True)
            diff = uniques[counts == 1]
            break

        last_idx = keep_idx

    diff_keep_num = min(sub_num - last_idx.shape[0], diff.shape[0])
    diff_keep_idx = torch.randperm(diff.shape[0])[:diff_keep_num].to(edge_index.device)
    final_keep_idx = torch.cat((last_idx, diff_keep_idx))

    drop_idx = torch.ones(x.shape[0], dtype=bool)
    drop_idx[final_keep_idx] = False
    x[drop_idx] = 0

    # final_keep_idx = final_keep_idx.to(device)
    final_keep_idx = final_keep_idx

    edge_index, _ = subgraph(final_keep_idx, edge_index)

    data.x = x
    data.edge_index = edge_index
    return data


class CMMS_GCL(nn.Module):
    def __init__(self, num_features_xd=84,dropout=0.2,aug_ratio=0.4):
        super(CMMS_GCL, self).__init__()

        self.W_rnn = nn.GRU(bidirectional=True, num_layers=1, input_size=100, hidden_size=100)
        self.f1 = torch.nn.Linear(200, 512)
        self.f2 = torch.nn.Linear(512, 256)
        self.f1_1 = torch.nn.Linear(200, 512)
        self.f2_1 = torch.nn.Linear(512, 256)
        self.W_att = nn.Linear(num_features_xd * 10, num_features_xd * 10)

        self.conv1 = GINConv(nn.Linear(num_features_xd, num_features_xd))
        self.conv2 = GINConv(nn.Linear(num_features_xd, num_features_xd * 10))
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1024)
        self.fc_g2 = torch.nn.Linear(1024, 512)

        self.fc_g3 = torch.nn.Linear(768, 256)
        self.fc_g4 = torch.nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.r = 10
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.aug_ratio = aug_ratio

    def softmax(self,input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d,dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)

    def forward(self, data,x,edge_index,batch,smi_em):

        # Sequence Encoder
        smi_em = smi_em.view(-1, 100, 100)
        smi_em, h = self.W_rnn(smi_em)
        smi_em = torch.relu(smi_em)
        sentence_att = torch.tanh(self.f1(smi_em))
        sentence_att = self.f2(sentence_att)
        sentence_att = self.softmax(sentence_att, 1)
        sentence_att = sentence_att.transpose(1, 2)
        sentence_embed = sentence_att @ smi_em
        smi_em = torch.sum(sentence_embed, 1) / self.r  # multi head
        smi_em = self.f1_1(smi_em)
        smi_em = self.f2_1(smi_em)

        # Graph Encoder
        x_1 = self.conv1(x, edge_index)
        x_1 = self.relu(x_1)
        x_2 = self.conv2(x_1, edge_index)
        x_2 = self.relu(x_2)
        x_g = torch.cat([gmp(x_2, batch), gap(x_2, batch)], dim=1)
        x_g = self.relu(self.fc_g1(x_g))
        x_g = self.dropout(x_g)
        x_g = self.fc_g2(x_g)

        # Sub-structure Sampling
        data_aug1 = Subgraph(data, self.aug_ratio)
        y, y_edge_index, y_batch = data_aug1.x, data_aug1.edge_index, data_aug1.batch

        y_1 = self.conv1(y, y_edge_index)
        y_1 = self.relu(y_1)
        y_2 = self.conv2(y_1, y_edge_index)
        y_2 = self.relu(y_2)
        y_g = torch.cat([gmp(y_2, batch), gap(y_2, batch)], dim=1)
        y_g = self.relu(self.fc_g1(y_g))
        y_g = self.dropout(y_g)
        y_g = self.fc_g2(y_g)

        # Stability predictor
        z = torch.cat((x_g,smi_em), dim=1)
        z = self.relu(self.fc_g3(z))
        z = self.dropout(z)
        z = self.fc_g4(z)

        return z,x_g,y_g





