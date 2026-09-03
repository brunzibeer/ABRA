from pathlib import Path

import torch
from groundingdino.util.misc import save_on_master

def save_checkpoint(args, model, svft: bool = False, pth_name: str = 'best', keys = [], merge: bool = False):
    state_dict = model.state_dict()

    cleaned_state_dict = None
    if svft:
        cleaned_state_dict = {}
        processed_names = []
        svft_state_dict = {k: v for k, v in state_dict.items() if 'svft' in k}
        for n, _ in svft_state_dict.items():
            name = n.rsplit('.', 2)[0]
            if name in processed_names:
                continue
            processed_names.append(name)
            module = dict(model.named_modules()).get(name, None)
            
            if not merge:
                cleaned_state_dict[f'{name}.svft_layer.u'] = module.get_u
                cleaned_state_dict[f'{name}.svft_layer.s_pre'] = module.s_pre
                cleaned_state_dict[f'{name}.svft_layer.s'] = module.s
                cleaned_state_dict[f'{name}.svft_layer.v'] = module.get_v
            else:
                # Merge the svft parameters into the original weight
                weight_key = name + '.weight'
                
                u = module.get_u
                s_pre = module.s_pre
                s = module.s
                v = module.get_v
                merged_svft = u @ (s_pre + s) @ v
                
                cleaned_state_dict[weight_key] = merged_svft
    else:
        cleaned_state_dict = {k: v for k, v in state_dict.items() if k in keys}

    assert cleaned_state_dict is not None, "Something is wrong in saving"
    out = Path(args.exp_out) / f"{pth_name}.pth"
    save_on_master(cleaned_state_dict, out)