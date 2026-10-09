from copy import deepcopy
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import torch
import re
import math
import wandb

from utils.transporter_functions import (
    find_unique_layers,
    merge_svft_layers,
    load_domain,
    create_task_vector,
    clean_pretrain_names,
    replace_rotated_residual
)

class Transporter:
    """
    Base class for all the transporters.
    A transporter is a module that takes in input a source and a target, and transports the
    data from the source to the target.

    Parameters:
    w_source: state dict of model trained with SVFT on the source class starting from source domain
    w_target: state dict initialized with SVFT trained on the target domain without knowledge of the target class
    """
    def __init__(self, w_source=None, w_target=None):
        self.w_source = w_source
        self.w_target = w_target

    def transport(self):
        pass
class SimpleTransporter(Transporter):
    """
    This version of the transporter, takes in as input a source and a target, and transports the data from the source to the target.
    It is performed in a simple way by just replacing the residual of the source (known) with the residual of the target (unknown).
    """
    def __init__(self, w_source=None, w_target=None):
        super().__init__(w_source=w_source, w_target=w_target)
        # Regex pattern to match 's' parameter (avoid matching 's_pre'); use raw string and anchors
        self.s_param_name = r'^transformer\.encoder\..*\.svft_layer\.s$'

    def transport(self):
        """
        Safely copy matching parameters named by `self.s_param_name` from source -> target.
        Assumes keys between the two state dicts match exactly (as you indicated).
        """
        src = self.w_source
    
        tgt = {}
        for k, p in src.items():
            if re.match(self.s_param_name, k):
                tgt[k] = p.detach().clone()
                
        return tgt

class RotationTransporter(Transporter):
    """
    This version of the transporter, takes in as input a source and a target, and transports the data from the source to the target.
    It is performed by obtaining a rototranslation matrix between the source and the target, keeping s_pre and s separate, rotating only s.
    """
    def __init__(self, w_source=None, w_target=None, mask=False, alpha_estimate=False):
        super().__init__(w_source=w_source, w_target=w_target)
        self.mask = mask
        self.alpha_estimate = alpha_estimate

    def apply_rototranslation(self, u_source, s_source, v_source_t, u_target, v_target_t):
        v_source = v_source_t.T
        v_target = v_target_t.T

        l_matrix = u_target.T @ u_source
        r_matrix = v_target.T @ v_source

        # Initial teleportation of the unseen class residual
        s_target_hat = l_matrix @ s_source @ r_matrix.T

        if self.mask == 'banded':
            mask = s_source != 0
            s_target = s_target_hat * mask
        elif self.mask == 'plain':
            mask = torch.eye(s_source.shape[0], s_source.shape[1]).to(self.device)
            s_target = s_target_hat * mask
        else:
            s_target = s_target_hat

        return s_target

    def compute_norm_preserving_alpha(self, sigma_s: torch.Tensor, sigma_t: torch.Tensor, eps: float = 1e-8) -> float:
        """
        Computes the scaling factor alpha based on the Frobenius norms of the singular values.
        
        Args:
            sigma_s: 1D tensor of source singular values for layer l.
            sigma_t: 1D tensor of target singular values for layer l.
            eps: Small epsilon to prevent division by zero.
            
        Returns:
            float: The computed scaling factor alpha_l.
        """
        # Calculate the L2 norm of the 1D vectors, which is equivalent 
        # to the Frobenius norm of their respective diagonal matrices.
        norm_s = torch.linalg.norm(sigma_s)
        norm_t = torch.linalg.norm(sigma_t)
        
        # Compute the ratio with numerical stability
        alpha = norm_t / (norm_s + eps)
        
        return alpha.item()

    def transport(self):
        tgt = self.w_target
        self.unique_layers = find_unique_layers(self.w_source)

        self.device = tgt[next(iter(tgt))].device

        transported_dict = {}
        for layer in self.unique_layers:
            u_source = self.w_source[f'{layer}.svft_layer.u'].detach().clone().to(self.device)
            s_source = self.w_source[f'{layer}.svft_layer.s'].detach().clone().to(self.device)
            v_source_t = self.w_source[f'{layer}.svft_layer.v'].detach().clone().to(self.device)

            u_target = tgt[f'{layer}.svft_layer.u'].detach().clone().to(self.device)
            v_target_t = tgt[f'{layer}.svft_layer.v'].detach().clone().to(self.device)

            s_target = self.apply_rototranslation(u_source, s_source, v_source_t, u_target, v_target_t)

            if self.alpha_estimate:
                s_pre_source = self.w_source[f'{layer}.svft_layer.s_pre'].detach().clone().to(self.device)
                s_pre_target = tgt[f'{layer}.svft_layer.s_pre'].detach().clone().to(self.device)
                alpha = self.compute_norm_preserving_alpha(torch.diag(s_pre_source), torch.diag(s_pre_target))
                s_target = alpha * s_target

            transported_dict[f'{layer}.svft_layer.s'] = s_target

        return transported_dict
    
    
class Analogy:
    """
    """
    def __init__(self, source_domain=None, target_domain=None, source_class=None, pretrain=None, beta=1.0):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.source_class = source_class
        self.pretrain = pretrain
        self.beta = beta

    def compute_analogy_transportation(self, source):
        source_keys = list(source.keys())

        tv_a = create_task_vector(self.pretrain, self.source_domain, self.device, source_keys)
        tv_b = create_task_vector(self.pretrain, self.target_domain, self.device, source_keys)
        tv_c = create_task_vector(self.pretrain, source, self.device, source_keys)

        transported_analogy = {}
        for k in tv_c:
            # if k not in tv_a.keys():
            #     tv_a[k] = 0.0
            # if k not in tv_b.keys():
            #     tv_b[k] = 0.0
            transported_analogy[k] = self.pretrain[k] + self.beta * (tv_c[k] - tv_a[k] + tv_b[k])

        return transported_analogy


    def transport(self):
        self.device = self.source_class[next(iter(self.source_class))].device
        self.unique_layers = find_unique_layers(self.source_class)
        self.pretrain = clean_pretrain_names(self.pretrain, self.device)
        
        self.source_domain = load_domain(self.source_domain) # Arriva nella forma .weights
        self.target_domain = load_domain(self.target_domain) # Arriva nella forma .weights

        recomposed_source = merge_svft_layers(self.source_class, self.unique_layers, self.device)
        transported_analogy = self.compute_analogy_transportation(recomposed_source)

        return transported_analogy

class MergedTransporter(Transporter):
    """
    This class implements a merged transporter that combines the outputs of the RotationTransporter and the Analogy transporter
    and combines them with a parameter alpha that weights the contribution of the two transporters.
    """
    def __init__(self, w_source=None, w_target=None, source_domain=None, target_domain=None, pretrain=None, alpha=0.5, mask=False, alpha_estimate=False):
        super().__init__(w_source=w_source, w_target=w_target)
        self.w_source = w_source # SVFT source class
        self.w_target = w_target # SVFTed target domain
        self.source_domain = source_domain # Non-SVFTed source domain
        self.target_domain = target_domain # Non-SVFTed target domain
        self.pretrain = pretrain # Theta_0
        self.alpha = alpha
        self.mask = mask
        self.alpha_estimate = alpha_estimate

    def transport(self):
        self.device = self.w_target[next(iter(self.w_target))].device
        self.unique_layers = find_unique_layers(self.w_source)
        
        # Get outputs from both transporters
        rotation_transporter = RotationTransporter(w_source=self.w_source, w_target=self.w_target, mask=self.mask, alpha_estimate=self.alpha_estimate)
        analogy_transporter = Analogy(source_domain=self.source_domain, target_domain=self.target_domain, source_class=self.w_source, pretrain=self.pretrain)

        rotated = rotation_transporter.transport()
        analogied = analogy_transporter.transport()

        self.w_target = replace_rotated_residual(self.w_target, rotated, self.device)
        merged_abra = merge_svft_layers(self.w_target, self.unique_layers, self.device)

        # Merge outputs with weighting
        merged = {}
        for k in analogied.keys():
            if k in merged_abra:
                merged[k] = self.alpha * merged_abra[k] + (1 - self.alpha) * analogied[k]
            else:
                merged[k] = analogied[k]
        return merged