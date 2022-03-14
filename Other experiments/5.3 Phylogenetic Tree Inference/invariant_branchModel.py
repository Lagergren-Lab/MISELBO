import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import pdb


class IndexedLinear(nn.Linear):
    """
    Implementation of permutation equivariant linear layers.
    
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.in_features, self.out_features = in_features, out_features
    
    def forward(self, x, in_index=None, out_index=None):
        indexed_weight, indexed_bias = self.weight, self.bias
        if in_index is not None:
            padded_weight = torch.cat((self.weight.t(), torch.zeros(1, self.out_features)), dim=0)
            indexed_weight = padded_weight[in_index].sum(1).t() 
                
        if out_index is not None:            
            padded_weight = torch.cat((self.weight, torch.zeros(1, self.in_features)), dim=0)
            indexed_weight = padded_weight[out_index].sum(1)
            if self.bias is not None:
                padded_bias = torch.cat((self.bias, torch.zeros(1)), dim=0)
                indexed_bias = padded_bias[out_index].sum(1)

        return F.linear(x, indexed_weight, indexed_bias)
        

class BatchIndexedLinear(nn.Linear):
    """
    Batched permutation equivariant linear layers
    
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.in_features, self.out_features = in_features, out_features
        
    def forward(self, x, batch_in_index=None, batch_out_index=None):
        batch_weight, batch_bias = self.weight.t(), self.bias
        if batch_in_index is not None:
            padded_weight = torch.cat((self.weight.t(), torch.zeros(1, self.out_features)), dim=0)
            batch_weight = padded_weight[batch_in_index].sum(-2)
        
        if batch_out_index is not None:
            padded_weight = torch.cat((self.weight, torch.zeros(1, self.in_features)), dim=0)
            batch_weight = torch.transpose(padded_weight[batch_out_index].sum(-2), 1, 2)
            if self.bias is not None:
                padded_bias = torch.cat((self.bias, torch.zeros(1)), dim=0)
                batch_bias = padded_bias[batch_out_index].sum(-1)
        
        if batch_in_index is not None or batch_out_index is not None:       
            x = x.unsqueeze(dim=1)
            output = x.bmm(batch_weight).squeeze(dim=1)
        else:
            output = x.mm(batch_weight)
        if self.bias is not None:
            output += batch_bias
            
        return output

        
class IndexedMLP(nn.Module):
    """
    Implementation of permutation equivariant MLP.
    
    """
    def __init__(self, nin, hidden_sizes, nout, bias=True):
        super().__init__()
        self.nin, self.nout = nin, nout
        self.mlp = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.mlp.extend([
                IndexedLinear(h0, h1, bias),
                nn.ELU()
            ])
        self.mlp.pop()
        self.num_of_layers = len(self.mlp) - 1
        self.mlp = nn.Sequential(*self.mlp)        
    
    def forward(self, x, in_index=None, out_index=None):
        for layer_idx, module in self.mlp._modules.items():
            if layer_idx == '0':
                x = module(x, in_index=in_index)
            elif layer_idx == str(self.num_of_layers):
                x = module(x, out_index=out_index)
            else:
                x = module(x)
        return x
        

class BatchIndexedMLP(nn.Module):
    """
    Batched permutation equivariant MLP.
    
    """
    def __init__(self, nin, hidden_sizes, nout, bias=True):
        super().__init__()
        self.nin, self.nout = nin, nout
        self.mlp = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.mlp.extend([
                BatchIndexedLinear(h0, h1, bias),
                nn.ELU()
            ])
        self.mlp.pop()
        self.num_of_layers = len(self.mlp) - 1
        self.mlp = nn.Sequential(*self.mlp)        
    
    def forward(self, x, batch_in_index=None, batch_out_index=None):
        for layer_idx, module in self.mlp._modules.items():
            if layer_idx == '0':
                x = module(x, batch_in_index=batch_in_index)
            elif layer_idx == str(self.num_of_layers):
                x = module(x, batch_out_index=batch_out_index)
            else:
                x = module(x)
        return x           