
import torch
import torch.nn.functional as F
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

###EdgeSHAPer as a function###
def edgeshaper(model, data, x, E,batch, smi_em,M=100, target_class=0, P=None, log_odds=True, seed=42, device="cpu"):

    rng = default_rng(seed=seed)
    model.eval()
    phi_edges = []

    num_nodes = x.shape[0]
    num_edges = E.shape[1]

    if P == None:
        max_num_edges = num_nodes * (num_nodes - 1)
        graph_density = num_edges / max_num_edges
        P = graph_density

    for j in tqdm(range(num_edges)):
        marginal_contrib = 0
        for i in range(M):
            E_z_mask = rng.binomial(1, P, num_edges)
            E_mask = torch.ones(num_edges)
            pi = torch.randperm(num_edges)

            E_j_plus_index = torch.ones(num_edges, dtype=torch.int)
            E_j_minus_index = torch.ones(num_edges, dtype=torch.int)
            selected_edge_index = np.where(pi == j)[0].item()
            for k in range(num_edges):
                if k <= selected_edge_index:
                    E_j_plus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

            for k in range(num_edges):
                if k < selected_edge_index:
                    E_j_minus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_minus_index[pi[k]] = E_z_mask[pi[k]]

            # with edge j
            retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(device).squeeze()
            E_j_plus = torch.index_select(E, dim=1, index=retained_indices_plus)

            out = model(data,x, E_j_plus, batch=batch,smi_em=smi_em)

            if not log_odds:
                out_prob = F.softmax(out, dim=1)
            else:
                out_prob = out  # out prob variable now containts log_odds

            V_j_plus = out_prob[0][target_class].item()

            # without edge j
            retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(device).squeeze()
            E_j_minus = torch.index_select(E, dim=1, index=retained_indices_minus)

            out = model(data, x, E_j_minus, batch=batch,smi_em=smi_em)

            if not log_odds:
                out_prob = F.softmax(out, dim=1)

            else:
                out_prob = out

            V_j_minus = out_prob[0][target_class].item()

            marginal_contrib += (V_j_plus - V_j_minus)

        phi_edges.append(marginal_contrib / M)
    return phi_edges
