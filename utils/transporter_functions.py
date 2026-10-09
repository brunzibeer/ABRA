from copy import deepcopy
import torch

def find_unique_layers(state_dict):
    unique_layers = []
    for n, p in state_dict.items():
        if 'svft' in n:
            name = n.rsplit('.', 2)[0]
            unique_layers.append(name)
    unique_layers = list(set(unique_layers))
    return unique_layers

def replace_rotated_residual(state_dict, rotated, device):
    for k in rotated.keys():
        state_dict[k] = rotated[k].to(device)
    return state_dict

def merge_svft_layers(state_dict, unique_layers, device):
    merged_dict = {}
    for layer in unique_layers:
        u = state_dict[f'{layer}.svft_layer.u'].to(device)
        s_pre = state_dict[f'{layer}.svft_layer.s_pre'].to(device)
        s = state_dict[f'{layer}.svft_layer.s'].to(device)
        vt = state_dict[f'{layer}.svft_layer.v'].to(device)
        
        merged_dict[f'{layer}.weight'] = u @ (s_pre + s) @ vt
    return merged_dict

def load_domain(path):
    return torch.load(path)

def create_task_vector(start_dict, finish_dict, device, filter=[]):
    tv = {}
    for k in finish_dict.keys():
        if k in start_dict.keys():
            if len(filter) == 0 or any(f in k for f in filter):
                tv[k] = finish_dict[k].to(device) - start_dict[k].to(device)

    return tv

def clean_pretrain_names(pretrain, device):
    return {k.replace("module.", "", 1): v.to(device) for k, v in pretrain.items()}
