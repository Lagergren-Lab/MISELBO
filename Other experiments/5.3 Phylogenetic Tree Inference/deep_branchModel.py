import torch
import torch.nn as nn
import torch.nn.functional as F
from invariant_branchModel import BatchIndexedMLP
from base_branchModel import BaseModel
import math

import pdb


class Encoder(nn.Module):
    def __init__(self, ntips, rootsplit_embedding_map, subsplit_embedding_map, psp=True, feature_dim=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.mean_std_model = BaseModel(ntips, rootsplit_embedding_map, subsplit_embedding_map, psp=psp, feature_dim=feature_dim)
    
    @property
    def embedding_dim(self):
        return self.mean_std_model.embedding_dim
        
    @property
    def padding_dim(self):
        return self.mean_std_model.padding_dim
            
    def forward(self, tree_list, return_details=False):
        self.mean_std_model.pad_feature()
        mean, std, neigh_ss_idxes = zip(*map(lambda x: self.mean_std_model.mean_std(x, return_adj_matrix=True), tree_list))
        mean, std, neigh_ss_idxes = torch.stack(mean, dim=0), torch.stack(std, dim=0), torch.stack(neigh_ss_idxes, dim=0)
        
        samp_log_branch, logq_branch = self.mean_std_model.sample_branch_base(len(tree_list))
        samp_log_branch, logq_branch = samp_log_branch * std.exp() + mean, logq_branch - torch.sum(std, -1)

        if not return_details:
            return samp_log_branch, logq_branch, neigh_ss_idxes
        else:
            return samp_log_branch, logq_branch, neigh_ss_idxes, mean, std
        

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, samp_log_branch, log_branch, *args):
        return samp_log_branch - 2.0, log_branch
        

class PlanarNF(nn.Module):
    """
    Permutation equvariant Planar flows for phylogenentic models.
    
    """
    def __init__(self, ntips, embedding_dim, *args, num_of_layers_nf=16, **kwargs):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_of_layers = num_of_layers_nf
        self.planar_weight = nn.Parameter(torch.randn(self.embedding_dim, self.num_of_layers)*0.01, requires_grad=True)
        self.planar_gamma = nn.Parameter(torch.randn(self.embedding_dim, self.num_of_layers)*0.01, requires_grad=True)
        self.planar_bias = nn.Parameter(torch.zeros(self.num_of_layers), requires_grad=True)
        
    def forward(self, samp_log_branch, logq_branch, neigh_ss_idxes):        
        planar_weight_padded, planar_gamma_padded = torch.cat((self.planar_weight, torch.zeros(1, self.num_of_layers)), dim=0), torch.cat((self.planar_gamma, torch.zeros(1, self.num_of_layers)), dim=0)
        planar_weight, planar_gamma, planar_bias = planar_weight_padded[neigh_ss_idxes].sum(-2), planar_gamma_padded[neigh_ss_idxes].sum(-2), self.planar_bias
        
        inner_prod = torch.sum(planar_gamma * planar_weight, dim=1, keepdim=True)
        gamma_hat = planar_gamma + (-1. + torch.log1p(inner_prod.exp()) - inner_prod) / torch.sum(planar_weight**2, dim=1, keepdim=True) * planar_weight
        regularized_inner_prod = torch.sum(gamma_hat * planar_weight, dim=1)
        for i in range(self.num_of_layers):
            planar_func = torch.tanh(torch.sum(planar_weight[:,:,i] * samp_log_branch, dim=-1, keepdim=True) + planar_bias[i])
            samp_log_branch = samp_log_branch + gamma_hat[:,:,i] * planar_func
            logq_branch -= torch.log1p((1.0-planar_func.squeeze()**2) * regularized_inner_prod[:,i])
        
        return samp_log_branch - 2.0, logq_branch


class RealNVP(nn.Module):
    """
    Permutation equivariant RealNVP for phylogenetic models.
    
    """
    def __init__(self, ntips, embedding_dim, padding_dim, hidden_sizes=[64], num_of_layers_nf=10, **kwargs):
        super().__init__()
                                        
        self.ntips, self.embedding_dim, self.padding_dim = ntips, embedding_dim, padding_dim

        self.num_of_layers = num_of_layers_nf
        self.realnvp = nn.ModuleList([BatchIndexedMLP(self.embedding_dim, hidden_sizes, 2*self.embedding_dim) for _ in range(self.num_of_layers)])

    def forward(self, samp_log_branch, logq_branch, neigh_ss_idxes):        
        samp_log_branch_a, samp_log_branch_b = samp_log_branch[:, :self.ntips], samp_log_branch[:, self.ntips:]
        neigh_ss_idxes_a, neigh_ss_idxes_b = neigh_ss_idxes[:, :self.ntips], neigh_ss_idxes[:, self.ntips:]
                    
        for i, mlp in enumerate(self.realnvp):
            # neigh_ss_idxes_a_updated = torch.where(neigh_ss_idxes_a==self.padding_dim, torch.LongTensor([2*self.embedding_dim]), neigh_ss_idxes_a)
            neigh_ss_idxes_a_shifted = torch.where(neigh_ss_idxes_a==self.padding_dim, torch.LongTensor([-1]), neigh_ss_idxes_a+self.embedding_dim)
            # s, t = torch.chunk(mlp(samp_log_branch_b, batch_in_index=neigh_ss_idxes_b, batch_out_index=torch.cat((neigh_ss_idxes_a, neigh_ss_idxes_a+self.embedding_dim), 1)), 2 , dim=-1)
            s, t = torch.chunk(mlp(samp_log_branch_b, batch_in_index=neigh_ss_idxes_b, batch_out_index=torch.cat((neigh_ss_idxes_a, neigh_ss_idxes_a_shifted), 1)), 2 , dim=-1)
            s = torch.sigmoid(s+2.)

            samp_log_branch_a = s * samp_log_branch_a + t
            logq_branch -= torch.sum(s.log(), dim=-1)
            samp_log_branch_a, samp_log_branch_b = samp_log_branch_b, samp_log_branch_a
            neigh_ss_idxes_a, neigh_ss_idxes_b = neigh_ss_idxes_b, neigh_ss_idxes_a

        if i%2 == 0:
            samp_log_branch = torch.cat([samp_log_branch_b, samp_log_branch_a], -1)
        else:
            samp_log_branch = torch.cat([samp_log_branch_a, samp_log_branch_b], -1)
        
        return samp_log_branch - 2.0, logq_branch


class DeepModel(nn.Module):
    FlowModel = {'planar': PlanarNF, 'realnvp': RealNVP, 'identity':Identity}
                    
    def __init__(self, ntips, rootsplit_embedding_map, subsplit_embedding_map, psp=True, feature_dim=50, hidden_sizes=[64], num_of_layers_nf=16,
                 flow_type='planar', bias=True, **kwargs):
        super().__init__()
        self.mean_std_encoder = Encoder(ntips, rootsplit_embedding_map, subsplit_embedding_map, psp=psp, feature_dim=feature_dim)
        
        self.embedding_dim, self.padding_dim = self.mean_std_encoder.embedding_dim, self.mean_std_encoder.padding_dim
        self.invariant_flow = self.FlowModel[flow_type](ntips, self.embedding_dim, self.padding_dim, hidden_sizes=hidden_sizes, num_of_layers_nf=num_of_layers_nf)
    
    def forward(self, tree_list):
        samp_log_branch, logq_branch, neigh_ss_idxes = self.mean_std_encoder(tree_list)
        
        return self.invariant_flow(samp_log_branch, logq_branch, neigh_ss_idxes)
        
        