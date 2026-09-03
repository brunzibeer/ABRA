import torch
from utils.transporter_functions import find_unique_layers

def compute_mse(transported_model, finetuned_model):
    layers_t = find_unique_layers(transported_model.state_dict())
    layers_gt = find_unique_layers(finetuned_model.state_dict())
    assert set(layers_t) == set(layers_gt), "Error! Mismatch in layers between transported and finetuned models"
    mse_dict_sep = {}
    mse_dict_sep_banded = {}
    mse_dict_sep_plain = {}
    mse_dict_merged = {}
    for lt, lgt in zip(layers_t, layers_gt):
        assert lt == lgt, f"Error! Mismatch in layer names: {lt} vs {lgt}"

        param_t = transported_model.state_dict()[f'{lt}.svft_layer.s']
        param_gt = finetuned_model.state_dict()[f'{lgt}.svft_layer.s']
        mse = torch.mean((param_t - param_gt) ** 2).item()
        mse_dict_sep[lt] = mse

        param_t_banded = param_t.clone()
        band_mask = param_gt == 0
        param_t_banded[band_mask] = 0
        mse_banded = torch.mean((param_t_banded - param_gt) ** 2).item()
        mse_dict_sep_banded[lt] = mse_banded

        param_t_plain = param_t.clone()
        plain_mask = torch.eye(param_gt.size(0), device=param_gt.device).bool()
        param_t_plain[~plain_mask] = 0
        mse_plain = torch.mean((param_t_plain - param_gt) ** 2).item()
        mse_dict_sep_plain[lt] = mse_plain

        param_spre_t = transported_model.state_dict()[f'{lt}.svft_layer.s_pre']
        param_u_t = transported_model.state_dict()[f'{lt}.svft_layer.u']
        param_v_t = transported_model.state_dict()[f'{lt}.svft_layer.v']

        param_spre_gt = finetuned_model.state_dict()[f'{lgt}.svft_layer.s_pre']
        param_u_gt = finetuned_model.state_dict()[f'{lgt}.svft_layer.u']
        param_v_gt = finetuned_model.state_dict()[f'{lgt}.svft_layer.v']

        # u @ (s_pre + s) @ v
        recomoposed_t = param_u_t @ (param_spre_t + param_t) @ param_v_t
        recomoposed_gt = param_u_gt @ (param_spre_gt + param_gt) @ param_v_gt
        mse_merged = torch.mean((recomoposed_t - recomoposed_gt) ** 2).item()
        mse_dict_merged[lt] = mse_merged

    # Compute the average MSE across all layers
    avg_mse_sep = sum(mse_dict_sep.values()) / len(mse_dict_sep)
    avg_mse_sep_banded = sum(mse_dict_sep_banded.values()) / len(mse_dict_sep_banded)
    avg_mse_sep_plain = sum(mse_dict_sep_plain.values()) / len(mse_dict_sep_plain)
    avg_mse_merged = sum(mse_dict_merged.values()) / len(mse_dict_merged)

    # Pretty print the average MSE for separate and merged cases
    print(f"Average MSE (separate): {avg_mse_sep:.6f}")
    print(f"Average MSE (separate banded): {avg_mse_sep_banded:.6f}")
    print(f"Average MSE (separate plain): {avg_mse_sep_plain:.6f}")
    print(f"Average MSE (merged): {avg_mse_merged:.6f}")                                                        
