import time
import math

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor


def create_orthonormal_matrix(A):
    # returns an orthonormal matrix (square) of size (min(A.shape), min(A.shape))
    Q, R = torch.qr(A)
    return Q

def replace_svft_with_fused_linear(model, target_modules_list):
    print("Replacing SVFT layers with new Linear layers")

    # filter out svft layer
    target_modules_list = [l for l in target_modules_list if "svft_layer" not in l]

    for target_path in tqdm(reversed(target_modules_list), total=len(target_modules_list)):
        parent_path = target_path[: target_path.rfind(".")] if "." in target_path else ""
        target_name = target_path.split(".")[-1]
        parent = model.get_submodule(parent_path) if parent_path else model
        target = model.get_submodule(target_path)
        in_dim = target.svft_layer.v.shape[1]
        out_dim = target.svft_layer.u.shape[0]
        if target.bias is None:
            lin = torch.nn.Linear(in_dim, out_dim, bias=False)
        else:
            lin = torch.nn.Linear(in_dim, out_dim, bias=True)
            lin.bias.data = target.bias.data
        lin.weight.data = target.merge_and_unload()
        parent.__setattr__(target_name, lin)

class SVFTLayer(nn.Module):
    def __init__(self, u, s, v, off_diag, pattern="banded", rank=None, fill_orthonormal=False, transporter=False, eval=False, svft_ckpt=False):

        """
        @inputs:
            u: torch.Tensor. Left singular vectors of pre-trained weight matrix
            s: torch.Tensor. Singular values of pre-trained weight matrix
            v: torch.Tensor. Right singular vectors of pre-trained weight matrix
            off_diag: int. Total off-diagonals to be used to populate matrix M (as referred in main paper)
            pattern: str. Choices: "banded", "random", "top_k". Using "banded" with off_diag=1 simulates SVFT-plain
            rank: int. Constraints how many singular vectors and values to use.
            fill_orthonormal: bool. To determine if random orthonormal basis should be used
        """

        super().__init__()

        self.off_diag = off_diag
        rank = s.shape[0] if rank is None else min(s.shape[0], rank)
        self.n = rank
        diff_rank = s.shape[0] - rank
        self.transporter = transporter
        self.eval_mode = eval
        self.svft_ckpt = svft_ckpt

        if fill_orthonormal:
            Q_u = torch.randn_like(u).to(s.device)
            torch.nn.init.orthogonal_(Q_u)
            Q_v = torch.randn_like(v).to(s.device)
            torch.nn.init.orthogonal_(Q_v)

            u = torch.cat([u[:, :rank], Q_u[:, :diff_rank]], dim=1)
            v = torch.cat([v[:rank, :], Q_v[:diff_rank, :]], dim=0)
            s = torch.cat([s[:rank], torch.zeros(diff_rank).to(s.device)], dim=0)
            self.n = s.shape[0]

        else:
            s = s[:rank]
            u = u[:, :rank]
            v = v[:rank, :]

        self.u = nn.Parameter(u.clone().detach().contiguous(), requires_grad=False)

        s_pre = s.cpu().detach().clone().contiguous()
        self.s_pre_edge_index = torch.sparse.spdiags(s_pre, torch.LongTensor([0]), (self.n, self.n)).coalesce().indices()
        self.s_pre = nn.Parameter(s_pre, requires_grad=False)
        
        if pattern=="banded":  
            diags = 2*self.off_diag + 1
            offsets_positive = torch.arange(0, self.off_diag+1)
            offsets_negative = torch.arange(-1, -self.off_diag-1, -1)
            self.offsets  = torch.cat([offsets_positive, offsets_negative])
            self.s_edge_index = torch.sparse.spdiags(torch.randn([diags, self.n]), self.offsets, (self.n, self.n)).coalesce().indices()
            if not transporter and not self.eval_mode and not self.svft_ckpt:
                self.s = torch.nn.Parameter(torch.zeros(self.s_edge_index.shape[1]), requires_grad=True)
            elif transporter or self.eval_mode or self.svft_ckpt:
                s_temp = SparseTensor(row=self.s_edge_index[0], col=self.s_edge_index[1], value=torch.randn(self.s_edge_index.shape[1])).to_dense()
                self.s = torch.zeros_like(s_temp).to(self.u.device)
                self.s = torch.nn.Parameter(self.s, requires_grad=True)
                
                s_pre_temp = SparseTensor(row=self.s_pre_edge_index[0], col=self.s_pre_edge_index[1], value=self.s_pre).to_dense()
                del self.s_pre
                self.s_pre = s_pre_temp.clone().to(self.u.device)
                self.s_pre = torch.nn.Parameter(self.s_pre, requires_grad=False)

        elif pattern=="random":
            print("Random pattern")
            k = self.n*(2*self.off_diag+1) - self.off_diag*(self.off_diag+1)
            rows = torch.randint(0, self.n, (k,))
            cols = torch.randint(0, self.n, (k,))
            self.s_edge_index = torch.stack([rows, cols])
            self.s = torch.nn.Parameter(torch.zeros(k), requires_grad=True)

        elif pattern=="top_k":

            if u.shape == v.shape:
                coeffs = u@v.T
            else:
                coeffs = u if u.shape[0]==u.shape[1] else v

            k = self.n*(2*self.off_diag+1) - self.off_diag*(self.off_diag+1)
            # Flatten the tensor to 1D
            flattened_tensor = coeffs.contiguous().view(-1)
            _, top_indices_flat = torch.topk(flattened_tensor, k)
            num_rows, num_cols = coeffs.size()
            rows = top_indices_flat // num_cols
            cols = top_indices_flat % num_cols
            self.s_edge_index = torch.stack([rows, cols])
            self.s = torch.nn.Parameter(torch.zeros(k), requires_grad=True)
       
        torch.nn.init.kaiming_normal_(self.s[None, :])
        self.s.squeeze()

        self.register_buffer('s_pre_row', self.s_pre_edge_index[0])
        self.register_buffer('s_pre_col', self.s_pre_edge_index[1])
        self.register_buffer('s_row', self.s_edge_index[0])
        self.register_buffer('s_col', self.s_edge_index[1])

        self.gate = nn.Parameter(torch.tensor([0.], dtype=torch.float32), requires_grad=True)

        self.v = nn.Parameter(v.clone().detach().contiguous(), requires_grad=False)


    def forward(self, x):
        x  = x @ self.get_weights()
        return x


    def get_weights(self):
        if not self.transporter and not self.eval_mode and not self.svft_ckpt:
            s = SparseTensor(row=self.s_row, col=self.s_col, value=self.s*F.sigmoid(self.gate)).to(self.u.device)
            s_pre = SparseTensor(row=self.s_pre_row, col=self.s_pre_col, value=self.s_pre).to(self.u.device)
            del_s = s_pre + s
            weight = (del_s @ self.v).T
            weight = weight @ self.u.T
        else:
            # In the transporter case, s is already dense and the gate is already applied
            del_s = self.s_pre + self.s
            weight = (del_s @ self.v).T
            weight = weight @ self.u.T
            # This version under here is just for numerical stability purposes
            # t_s_pre = (self.s_pre @ self.v).T
            # t_s_pre = t_s_pre @ self.u.T
            # t_s = (self.s @ self.v).T
            # t_s = t_s @ self.u.T
            # weight = t_s_pre + t_s
        return weight
    
    def get_s_pre_dense(self):
        if self.svft_ckpt or self.eval_mode or self.transporter:
            return self.s_pre
        s_pre = SparseTensor(row=self.s_pre_row, col=self.s_pre_col, value=self.s_pre).to(self.u.device)
        return s_pre.to_dense()
    
    def get_s_dense(self):
        if self.svft_ckpt or self.eval_mode or self.transporter:
            return self.s
        s = SparseTensor(row=self.s_row, col=self.s_col, value=self.s*F.sigmoid(self.gate)).to(self.u.device)
        return s.to_dense()
    
    def get_u(self):
        return self.u
    
    def get_v(self):
        return self.v

    def merge_and_unload(self):
        return self.get_weights().T.contiguous()

   
class LinearWithSVFT(nn.Module):

    def __init__(self, linear, off_diag, pattern="banded", rank=None, fill_orthonormal=False, transporter=False, eval=False, svft_ckpt=False):
        """
        @inputs:
                linear: torch.Tensor. Linear Layer that has to adapted
                off_diag: int. total number off diagonals to be used if pattern is 'banded' 
                          for remaining patterns, equivalent number of learnable parameters are learnt
                rank: SVD rank 
                fill_orthonormal: bool. To determine if random orthonormal basis should be used
        """
        
        super().__init__()

        self.bias = linear.bias

        # since linear.weight is on GPU, computing SVD will be significantly faster
        svd = torch.linalg.svd(linear.weight, full_matrices=False)

        self.svft_layer = SVFTLayer(svd[0], 
                                    svd[1], 
                                    svd[2], 
                                    off_diag=off_diag, 
                                    pattern=pattern, 
                                    rank=rank, 
                                    fill_orthonormal=fill_orthonormal,
                                    transporter=transporter,
                                    eval=eval,
                                    svft_ckpt=svft_ckpt)

    def forward(self, x):
        if self.bias is not None:
            x = self.svft_layer(x) + self.bias
            return x

        else:
            x = self.svft_layer(x)
            return x

    def merge_and_unload(self):
        return self.svft_layer.merge_and_unload()

    @property
    def weight(self):
        """Return the current merged weight tensor with autograd tracking.

        This follows the behaviour callers expect from an nn.Linear-like
        ``weight`` attribute: when used in a loss, the returned tensor will be
        part of the autograd graph so gradients flow back into the SVFT
        parameters. If you need a read-only copy that does not build a graph,
        use ``weight_no_grad`` instead.
        """
        # compute merged weight normally so it remains connected to the SVFT params
        w = self.svft_layer.get_weights().T.contiguous()
        return w

    @property
    def get_u(self):
        return self.svft_layer.get_u()

    @property
    def get_v(self):
        return self.svft_layer.get_v()

    @property
    def s_pre(self):
        return self.svft_layer.get_s_pre_dense()

    @property
    def s(self):
        return self.svft_layer.get_s_dense()


def freeze_model(model, exclude_list = None):
    ''' Freeze all parameters of the model '''
    if exclude_list is None:
        exclude_list = []

    for n, p in model.named_parameters():
        if not any(e in n for e in exclude_list):
            p.requires_grad = False