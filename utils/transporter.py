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
    merge_state_dicts,
    prepare_for_analogy,
    replace_rotated_residual
)
from utils.matrix_heatmap import save_matrix_heatmap

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
    def __init__(self, w_source=None, w_target=None, mask=False, weighted=False, calibration_classes=None, calib_exp=None, alpha_estimate=False):
        super().__init__(w_source=w_source, w_target=w_target)
        self.mask = mask
        self.weighted = weighted
        self.calibration_classes = calibration_classes
        self.calib_exp = calib_exp
        self.alpha_estimate = alpha_estimate

        if calibration_classes is not None:
            self.calibrate()
        
    def calibrate(self):
        source_base = f"/work/horizon_store/abraout/experts/cityscapes/normal/{self.calib_exp}_"
        target_base = f"/work/horizon_store/abraout/experts/cityscapes/fog/{self.calib_exp}_"
        
        self.calibration_data = {}
        for cls in self.calibration_classes:
            source_path = Path(source_base + cls + "/best.pth")
            target_path = Path(target_base + cls + "/best.pth")
            
            # Load calibration pairs
            source_calib = torch.load(source_path)
            target_calib = torch.load(target_path)
            
            self.calibration_data[cls] = {
                'source': source_calib,
                'target': target_calib
            }

    def apply_rototranslation(self, u_source, s_source, v_source_t, u_target, v_target_t, layer=None):
        v_source = v_source_t.T
        v_target = v_target_t.T

        if self.weighted:
            # Weighted Procrustes (focuses alignment on large residual values)
            weights = torch.abs(s_source)
            
            m_l = u_target.T @ u_source @ weights @ weights.T
            u_l, _, v_l_t = torch.linalg.svd(m_l)
            l_matrix = u_l @ v_l_t
            
            m_r = v_target.T @ v_source @ weights.T @ weights
            u_r, _, v_r_t = torch.linalg.svd(m_r)
            r_matrix = u_r @ v_r_t
        else:
            # Standard Unweighted Procrustes (original ABRA)
            l_matrix = u_target.T @ u_source
            r_matrix = v_target.T @ v_source

        # Initial teleportation of the unseen class residual
        s_target_hat = l_matrix @ s_source @ r_matrix.T

        # save_matrix_heatmap(l_matrix, f"l_matrix.png", title=f"L Matrix Heatmap")
        # save_matrix_heatmap(r_matrix, f"r_matrix.png", title=f"R Matrix Heatmap")
        # exit()

        if self.calibration_classes is not None:
            # Initialize numerators and denominators for the element-wise scale
            num = torch.zeros_like(s_target_hat)
            den = torch.zeros_like(s_target_hat)
            
            # (Make sure self.calibration_data matches your class attribute)
            for _, calib_data in getattr(self, 'calibration_data', self.calibration_classes).items():
                s_source_calib = calib_data['source'][f"{layer}.svft_layer.s"].to(self.device)
                s_target_calib = calib_data['target'][f"{layer}.svft_layer.s"].to(self.device)
                
                s_target_calib_hat = l_matrix @ s_source_calib @ r_matrix.T
                
                if self.mask == 'banded':
                    s_target_calib_hat = s_target_calib_hat * (s_source_calib != 0)
                elif self.mask == 'plain':
                    s_target_calib_hat = s_target_calib_hat * torch.eye(s_source_calib.shape[0], s_source_calib.shape[1], device=self.device)
                
                # Element-wise accumulation
                num += s_target_calib_hat * s_target_calib
                den += s_target_calib_hat * s_target_calib_hat
                
            # 3. Apply Bayesian Prior (Smoothing)
            # lam pulls the scale toward 1.0 for elements where the calibration classes have 0 activation.
            # W = (0 + lam) / (0 + lam) = 1.0 (No scaling applied to unseen features)
            lam = 1e-4 
            W = (num + lam) / (den + lam)
            
            # Clamp W to avoid extreme spikes from numerical instability
            W = torch.clamp(W, min=0.1, max=10.0)
            
            # Apply the element-wise scale to the unseen class
            s_target_hat = s_target_hat * W
            
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

            s_target = self.apply_rototranslation(u_source, s_source, v_source_t, u_target, v_target_t, layer)

            if self.alpha_estimate:
                s_pre_source = self.w_source[f'{layer}.svft_layer.s_pre'].detach().clone().to(self.device)
                s_pre_target = tgt[f'{layer}.svft_layer.s_pre'].detach().clone().to(self.device)
                alpha = self.compute_norm_preserving_alpha(torch.diag(s_pre_source), torch.diag(s_pre_target))
                s_target = alpha * s_target

            transported_dict[f'{layer}.svft_layer.s'] = s_target

        return transported_dict
    
    
class RotationTransporterGemini(RotationTransporter):
    def __init__(self, w_source=None, w_target=None, mask=False):
        super().__init__(w_source, w_target, mask)
        
    def apply_rototranslation(self, u_source, s_source, v_source_t, u_target, v_target_t):
        """
        Computes orthogonal transformations L and R to align source and target 
        domains using standard Orthogonal Procrustes (no regularization).
        """
        v_source = v_source_t.T
        v_target = v_target_t.T

        # Calculate cross-covariance matrices
        base_M_L = u_target.T @ u_source
        base_M_R = v_target.T @ v_source

        # Apply SVD to project onto the orthogonal manifold (L.T @ L = I)
        U_L, _, Vh_L = torch.linalg.svd(base_M_L, full_matrices=False)
        l_matrix = U_L @ Vh_L

        # Apply SVD for R (R.T @ R = I)
        U_R, _, Vh_R = torch.linalg.svd(base_M_R, full_matrices=False)
        r_matrix = U_R @ Vh_R
        
        s_target = l_matrix @ s_source @ r_matrix
        
        return s_target
    
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
        # a = prepare_for_analogy(self.source_domain, self.pretrain)
        # b = prepare_for_analogy(self.target_domain, self.pretrain)
        # c = prepare_for_analogy(source, self.pretrain)
        
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

class AnalogyFull:
    """
    """
    def __init__(self, source_domain=None, target_domain=None, source_class=None, pretrain=None, beta=1.0):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.source_class = source_class
        self.pretrain = pretrain
        self.beta = beta

    def compute_analogy_transportation(self, source):
        a = prepare_for_analogy(self.source_domain, self.pretrain)
        b = prepare_for_analogy(self.target_domain, self.pretrain)
        c = prepare_for_analogy(source, self.pretrain)

        tv_a = create_task_vector(self.pretrain, a, self.device)
        tv_b = create_task_vector(self.pretrain, b, self.device)
        tv_c = create_task_vector(self.pretrain, c, self.device)
        
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
        self.pretrain = clean_pretrain_names(self.pretrain, self.device)
        
        self.source_domain = load_domain(self.source_domain) # Arriva nella forma .weights
        self.target_domain = load_domain(self.target_domain) # Arriva nella forma .weights

        transported_analogy = self.compute_analogy_transportation(self.source_class)

        return transported_analogy

class MergedTransporter(Transporter):
    """
    This class implements a merged transporter that combines the outputs of the RotationTransporter and the Analogy transporter
    and combines them with a parameter alpha that weights the contribution of the two transporters.
    """
    def __init__(self, w_source=None, w_target=None, source_domain=None, target_domain=None, pretrain=None, alpha=0.5, mask=False, weighted=False, calibration_classes=None, calib_exp=None, alpha_estimate=False):
        super().__init__(w_source=w_source, w_target=w_target)
        self.w_source = w_source # SVFT source class
        self.w_target = w_target # SVFTed target domain
        self.source_domain = source_domain # Non-SVFTed source domain
        self.target_domain = target_domain # Non-SVFTed target domain
        self.pretrain = pretrain # Theta_0
        self.alpha = alpha
        self.mask = mask
        self.weighted = weighted
        self.calibration_classes = calibration_classes
        self.calib_exp = calib_exp
        self.alpha_estimate = alpha_estimate

    def transport(self):
        self.device = self.w_target[next(iter(self.w_target))].device
        self.unique_layers = find_unique_layers(self.w_source)
        
        # Get outputs from both transporters
        rotation_transporter = RotationTransporter(w_source=self.w_source, w_target=self.w_target, mask=self.mask, weighted=self.weighted, calibration_classes=self.calibration_classes, calib_exp=self.calib_exp, alpha_estimate=self.alpha_estimate)
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