import re
from tqdm import tqdm
from torch.nn import Linear

def match_name_keywords(n: str, name_keywords: list):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out

def get_trainable_parameters(args, model, type='default'):
    trainable = []
    if type == 'all':
        trainable = [
            {
                'params': [p for _, p in model.named_parameters() if p.requires_grad],
                'lr': args.lr
            }
        ]
    elif type == 'default':
        trainable = [
            {
                'params': [p for n, p in model.named_parameters() if p.requires_grad and not match_name_keywords(n, args.lp_names)],
                'lr': args.lr
            },
            {
                'params': [p for n, p in model.named_parameters() if p.requires_grad and match_name_keywords(n, args.lp_names)],
                'lr': args.lp_lr
            }
        ]
    elif type == 'adapt':
        trainable = [
            {
                'params': [p for n, p in model.named_parameters() if p.requires_grad and not 
                           match_name_keywords(n, args.lp_names) and 
                           match_name_keywords(n, 'backbone') and not
                           match_name_keywords(n, 'transformer')],
                'lr': args.backbone_lr
            },
            {
                'params': [p for n, p in model.named_parameters() if p.requires_grad and not 
                           match_name_keywords(n, args.lp_names) and 
                           match_name_keywords(n, 'transformer') and not
                           match_name_keywords(n, 'backbone')],
                'lr': args.lr
            },
            {
                'params': [p for n, p in model.named_parameters() if p.requires_grad and match_name_keywords(n, args.lp_names)],
                'lr': args.lp_lr
            }
        ]

    return trainable

def get_trainable_keys(model):
    trainable = []
    for k, p in model.named_parameters():
        if p.requires_grad:
            trainable.append(k)

    return trainable

def get_target_modules_list(model, trainable, svft=False):
    regular_expression = None
    if trainable == 'encoder':
        regular_expression = r"\b[\w\.]*transformer\.encoder[\w\.]*\b"
    elif trainable == 'encoder_attn':
        regular_expression = r"^(?=.*transformer\.encoder)(?=.*attn)[\w\.]+$"
    elif trainable == 'decoder':
        regular_expression = r"^[\w\.]*transformer\.decoder[\w\.]*$"
    elif trainable == 'decoder_attn':
        regular_expression = r"^(?=.*transformer\.decoder)(?=.*attn)[\w\.]+$"
    elif trainable == 'attn':
        # Train all the attention layer in the model
        regular_expression = r"^(?=.*attn)[\w\.]+$"
    elif trainable == 'all':
        regular_expression = r".+"
    elif trainable == 'frcnn':
        regular_expression = r"^(rpn|roi_heads)[\w\.]*$"

    target_modules = []
    for n, m in model.named_modules():
        if re.fullmatch(regular_expression, n):
            target_modules.append(n)
            if not svft:
                for param in m.parameters():
                    param.requires_grad = True

    return target_modules

def create_and_replace_modules(model, target_modules_list, create_fn):
    print("Replacing Linear layers with SVFT layers")

    for target_path in tqdm(reversed(target_modules_list), total=len(target_modules_list)):
        parent_path = target_path[: target_path.rfind(".")] if "." in target_path else ""
        target_name = target_path.split(".")[-1]
        parent = model.get_submodule(parent_path) if parent_path else model
        target = model.get_submodule(target_path)
        if isinstance(target, Linear):
            parent.__setattr__(target_name, create_fn(target))