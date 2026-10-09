# Standard python libraries
import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path
import uuid

# AI Specific libraries
import torch
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler, DataLoader
import wandb

# Custom pieces of codes and functions
import groundingdino.util.misc as utils
from utils import binary_to_boolean_type
from utils.builders import *
from utils.parameters import get_trainable_parameters, get_trainable_keys, get_target_modules_list, create_and_replace_modules
from utils.checkpoints import save_checkpoint
from utils.fda_engine import run_fda
from utils.alpha_search import run_alpha_search
from utils.compute_mse import compute_mse
from groundingdino.util.slconfig import SLConfig
from groundingdino.datasets import get_coco_api_from_dataset
from groundingdino.models.svft import LinearWithSVFT
from tools.wandb_utils import initialize_wandb
from engine import train_one_epoch, evaluate

def get_parser():
    parser = argparse.ArgumentParser(description="OpenGD: Open Vocabulary Object Detection")
    
    # Input and output paths
    parser.add_argument('--output_root', type=Path, help='path to output root')
    parser.add_argument('--root', type=Path, help='path to dataset root')
    parser.add_argument('--train_annotations', type=Path, help='path to training annotations')
    parser.add_argument('--test_annotations', type=Path, help='path to validation annotations')
    parser.add_argument('--label_map', type=Path)
    parser.add_argument('--fix_size', type=binary_to_boolean_type, default=False)
    parser.add_argument('--objectify', type=binary_to_boolean_type, default=False, help='Witch labels to "object" for training')
    parser.add_argument('--add_split', type=binary_to_boolean_type, default=True, help='add split to image folder')
    
    # Model related args
    parser.add_argument('--model_cfg', type=Path, required=True, help='path to model config file')
    parser.add_argument('--weights', type=Path, default=None, required=False, help='path to model weights')
    parser.add_argument('--finetuning', type=Path, default=None, help='path to the finetuned weights of the model')
    parser.add_argument('--trainable', type=str, default='encoder_attn', choices=['encoder', 'decoder', 'encoder_attn', 'decoder_attn', 'attn', 'all'], help='parameters to train')
    parser.add_argument('--param_type', type=str, default='default', choices=['all', 'default', 'adapt'])

    # Experiments args
    parser.add_argument('--device', type=str, default='cuda', help='device to use for experiment')
    parser.add_argument('--exp_name', type=str, required=True, help='name of the experiment')
    parser.add_argument('--test', type=binary_to_boolean_type, default=False, help='if True, perform evaluation only')
    parser.add_argument('--test_period', type=int, default=1, help='evaluation period (in epochs)')
    parser.add_argument('--class_loss_w', type=float, default=1.0, help='Weight for loss on classes')
    parser.add_argument('--object_loss_w', type=float, default=1.0, help='Weight for loss on object')
    parser.add_argument('--save_coco_format', type=Path, default=None, help='If provided, save zero shot prediction as coco format')
    parser.add_argument('--zs_th', type=float, default=0.3, help='Threshold for saving zero-shot prediction')
    
    # Optimization Args
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loading')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='number of gradient accumulation steps')
    parser.add_argument('--optim', type=str, choices=['adamw', 'sgd'], default='sgd')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lp_names', type=str, nargs='+', default=['ref_point_head', 'sampling_offsets'])
    parser.add_argument('--lp_lr', type=float, default=1e-5)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum for SGD')
    parser.add_argument('--nesterov', type=binary_to_boolean_type, default=False, help='Use nesterov with SGD')
    parser.add_argument('--scheduler', type=str, default=None, choices=['step', 'multi', 'cosine', 'sequential'], help='which LR scheduler to use')
    parser.add_argument('--step_size', type=int, default=None, help='step size for StepLR scheduler')
    parser.add_argument('--step_milestones', type=int, default=[8, 11], nargs='+')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='Warmup ratio for cosine scheduler')
    
    # PEFT
    parser.add_argument('--svft', type=binary_to_boolean_type, default=False, help='use svft as PEFT method')
    parser.add_argument('--svft_pattern', type=str, default='banded', choices=['banded'], help='SVFT sparsity pattern')
    parser.add_argument('--svft_diag', type=int, default=2, help='parameter to enlarge or reduce the banded pattern')
    parser.add_argument('--svft_ckpt', type=binary_to_boolean_type, default=False, help='define if the finetuning checkpoint has to be loaded after svft')

    # ABRA
    parser.add_argument('--transporter', type=str, default=None, choices=['simple', 'abra', 'analogy', 'merged'], help='transport method to use')
    parser.add_argument('--abra_mask', type=str, default=None, choices=['plain', 'banded'], help='mask the rotation output for numerical stability')
    parser.add_argument('--source_class', type=Path, default=None, help='path to the class checkpoint') # Also for param delta
    parser.add_argument('--source_domain', type=Path, default=None, help='source domain to use')
    parser.add_argument('--target_domain', type=Path, default=None, help='target domain to use')
    parser.add_argument('--merged_alpha', type=float, default=0.5, help='alpha for merging the weights in the "merged" transporter')
    parser.add_argument('--mse', type=binary_to_boolean_type, default=0, help='Compute MSE between transported and finetuned weights (only for analysis)')
    parser.add_argument('--mse_gt', type=Path, default=None, help='path to the finetuned checkpoint to use as GT for MSE computation (only for analysis)')

    # Alpha search
    parser.add_argument('--alpha_search', type=binary_to_boolean_type, default=False, help='Whether to perform alpha search for the merged transporter')
    parser.add_argument('--min_alpha', type=float, default=0.1, help='Minimum alpha for alpha search')
    parser.add_argument('--max_alpha', type=float, default=5.0, help='Maximum alpha for alpha search')
    parser.add_argument('--alpha_step', type=float, default=0.1, help='Alpha step for alpha search')
    parser.add_argument('--alpha_patience', type=int, default=5, help='Stop alpha search early if AP does not improve after this many steps')
    parser.add_argument('--alpha_estimate', type=binary_to_boolean_type, default=False, help='Whether to estimate a good alpha from the eigenvalues of the transported rotation matrix instead of performing a search')
    
    # UDA
    parser.add_argument('--uda', type=str, default=None, choices=['fda'], help='apply UDA baseline')
    parser.add_argument('--source_root', type=Path, default=None, help='path to source dataset root for UDA')
    parser.add_argument('--target_root', type=Path, default=None, help='path to target dataset root for UDA')
    parser.add_argument('--source_train_annotations', type=Path, default=None, help='path to source training annotations for UDA')
    parser.add_argument('--target_train_annotations', type=Path, default=None, help='path to target training annotations for UDA')
    parser.add_argument('--target_test_annotations', type=Path, default=None, help='path to target validation annotations for UDA')
    parser.add_argument('--fda_ent', type=float, default=0.0, help='Apply entropy loss on target domain for FDA')
    parser.add_argument('--fda_lb', type=float, default=0.01, help='L parameter for FDA')
    parser.add_argument('--fda_ita', type=float, default=2.0, help='ITA parameter for FDA')
    
    
    # Debugging related args
    parser.add_argument('--debug', type=binary_to_boolean_type, default=False, help='if True, run in debug mode (smaller dataset, more logging, etc.)')
    parser.add_argument('--visualize_predictions', type=binary_to_boolean_type, default=False, help='if True, visualize predictions')
    parser.add_argument('--visualize_targets', type=binary_to_boolean_type, default=False, help='if True, visualize targets')
    parser.add_argument('--save_abra', type=binary_to_boolean_type, default=False, help='if True, save the transported weights of ABRA')
    
    # Wandb related args
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity name')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb project name')


    return parser


def main(args):
    if not torch.cuda.is_available() and 'cuda' in args.device:
        exit("CUDA is not available, please check your device argument or your CUDA installation.")
    device = torch.device(args.device)

    # 1 Load Dataset
    ####### TRAIN DS #######
    args.add_split = any([n in str(args.root) for n in ['cityscapes', 'AODRaw', 'bdd']])
    args.coco_val_path = args.test_annotations if not args.uda else args.target_test_annotations
    if not args.test and not args.uda:
        train_dataset = build_dataset(
            args=args,
            img_set='train',
            type='odvg',
            add_split=args.add_split
        )
        train_sampler = RandomSampler(train_dataset)
        train_sampler_batch = BatchSampler(
            sampler=train_sampler,
            batch_size=args.batch_size,
            drop_last=True
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_sampler_batch,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers
        )

    if not args.uda:
        ####### TEST DS #######
        test_dataset = build_dataset(
            args=args,
            img_set='val',
            type='coco',
            add_split=args.add_split
        )
        test_sampler = SequentialSampler(test_dataset)
        test_loader = DataLoader(
            dataset=test_dataset,
            sampler=test_sampler,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers
        )

        # Extract base dataset from CocoDataset for evaluation
        base_dataset = get_coco_api_from_dataset(test_dataset)
    
    # 2 Load Model and Loss
    model, post_proc = build_model(args)
    model.to(device=device)
    criterion = build_loss(args)
    
    # 3 Load weights
    pretrain = torch.load(args.weights, map_location='cpu')['model']
    pretrain = utils.clean_state_dict(pretrain) # Used to remove 'module' from name
    load_output = model.load_state_dict(pretrain, strict=False)
    print(load_output)
    
    # 4 Load finetuning
    if args.finetuning is not None and not args.svft_ckpt:
        finetuning = torch.load(args.finetuning, map_location='cpu')
        finetuning = utils.clean_state_dict(finetuning)
        load_output = model.load_state_dict(finetuning, strict=False)
        print(load_output)

    # 5 Freeze parameters
    for _, p in model.named_parameters():
        p.requires_grad = False
    
    modules = get_target_modules_list(model=model, trainable=args.trainable, svft=args.svft)
    
    # 6 Apply stuff
    original_model = None
    if args.svft:
        if args.transporter == 'merged':
            original_model = deepcopy(model)
        create_and_replace_modules(
            model=model,
            target_modules_list=modules,
            create_fn=partial(
                LinearWithSVFT,
                off_diag=args.svft_diag,
                pattern=args.svft_pattern,
                transporter=args.transporter is not None,
                eval=args.test,
                svft_ckpt=args.svft_ckpt,
            )
        )

    svft_model_mse = None
    if args.mse:
        assert args.svft, "MSE computation is only implemented for SVFT models"
        svft_model_mse = deepcopy(model)
        gt_ckpt = torch.load(args.mse_gt, map_location='cpu')
        gt_ckpt = utils.clean_state_dict(gt_ckpt)

        svft_model_mse.load_state_dict(gt_ckpt, strict=False)
    
    if args.finetuning is not None and args.svft_ckpt:
        finetuning = torch.load(args.finetuning, map_location='cpu')
        finetuning = utils.clean_state_dict(finetuning)
        load_output = model.load_state_dict(finetuning, strict=False)
        print(load_output)

        if args.mse:
            torch.load(args.mse_gt, map_location='cpu')
            gt_ckpt = utils.clean_state_dict(torch.load(args.mse_gt, map_location='cpu'))
            svft_model_mse.load_state_dict(gt_ckpt, strict=False)

            return compute_mse(model, svft_model_mse)

    # 7 Optimizer and Scheduler
    trainable_params = get_trainable_parameters(args=args, model=model, type=args.param_type)
    trainable_keys = get_trainable_keys(model=model)
    args.trainable_keys = trainable_keys
    optimizer = build_optimizer(args, params=trainable_params)

    scheduler = None
    if args.scheduler is not None and not args.test and not args.uda:
        scheduler, optimizer = build_scheduler(args=args, optimizer=optimizer, train_dataset=train_dataset)
    
    # 8 Train
    if args.uda and not args.test:
        if args.uda == 'fda':
            return run_fda(
                args=args,
                model=model,
                optimizer=optimizer,
                pp=post_proc,
                device=device,
                criterion=criterion
            )
    elif not args.uda and not args.test:
        best_ap = -99999
        for ep in range(args.epochs):
            if ep % args.test_period == 0:
                eval_stats = evaluate(
                    args=args,
                    model=model,
                    postprocessors=post_proc,
                    loader=test_loader,
                    base_ds=base_dataset,
                    epoch=ep,
                    max_epoch=args.epochs,
                    device=device,
                    output_root=args.exp_out
                )
                ap = eval_stats['coco_eval_bbox'][0]
                
            if ep != 0 and ap > best_ap:
                best_ap = ap
                save_checkpoint(args=args, model=model, svft=args.svft, pth_name='best', keys=trainable_keys)
            save_checkpoint(args=args, model=model, svft=args.svft, pth_name='last', keys=trainable_keys)

            # Train one epoch
            train_one_epoch(
                args=args,
                model=model,
                criterion=criterion,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=ep,
                max_epochs=args.epochs,
                max_norm=args.clip_max_norm,
                scheduler=scheduler
            )

            if scheduler is not None and args.scheduler != 'onecylcer':
                scheduler.step()
                
            if wandb.run is not None:
                wandb.log({'TRAIN/LR': optimizer.param_groups[0]['lr']})
     
    # 9 Eval
    if args.transporter is not None and args.test:
        w_source = torch.load(args.source_class)
        
        transporter = build_transporter(
            args=args,
            transporter_type=args.transporter,
            w_source=w_source,
            w_target=deepcopy(model.state_dict()),
            mask=args.abra_mask,
            analogy_source=args.source_domain,
            analogy_target=args.target_domain,
            pretrain=pretrain
        )
        transported = transporter.transport()
        if args.transporter == 'merged':
            model = original_model
        load_output = model.load_state_dict(transported, strict=False)

        if args.mse:
            return compute_mse(model, svft_model_mse)
        
        if args.save_abra:
            return save_checkpoint(args=args, model=model, svft=True, pth_name='merged', keys=trainable_keys, merge=True)
    
    if args.test:
        args.epochs = 1

    if args.alpha_search:
        assert not args.alpha_estimate, "Alpha estimation and alpha search cannot be used together"
        best_alpha = run_alpha_search(
            args=args,
            pretrain=pretrain,
            transported=transported,
            model=model,
            post_proc=post_proc,
            test_loader=test_loader,
            base_dataset=base_dataset,
            device=device,
            evaluate_fn=evaluate
        )
        print(f"Alpha search complete. Running final evaluation with best alpha={best_alpha:.3f}")
        
    evaluate(
        args=args,
        model=model,
        postprocessors=post_proc,
        loader=test_loader,
        base_ds=base_dataset,
        epoch=args.epochs,
        max_epoch=args.epochs,
        device=device,
        output_root=args.exp_out
    )
    if not args.test:
        save_checkpoint(args=args, model=model, svft=args.svft, pth_name='last', keys=trainable_keys)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    args.exp_out = args.output_root
    args.exp_out.mkdir(parents=True, exist_ok=True)
    
    cfg_path = args.model_cfg
    model_cfg = SLConfig.fromfile(cfg_path)
    model_cfg = model_cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in model_cfg.items():
        if k not in args_vars:
            setattr(args, k, v)

    if args.wandb_entity is not None and args.wandb_project is not None:
        args.wandb_name = args.exp_name
        args.jobnum = str(uuid.uuid4())
        initialize_wandb(args)
    
    main(args)
