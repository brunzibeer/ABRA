import wandb

def initialize_wandb(args) -> None:
    """
    Initializes wandb, if installed.

    Args:
        args: the arguments of the current execution
    """
    assert wandb is not None, "Wandb not installed, please install it or run without wandb"
    assert args.wandb_name is not None, "Please provide a wandb run name"
    run_name = args.wandb_name

    run_id = args.jobnum.split('-')[0]
    name = f'{run_name}_{run_id}'
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=name)
    args.wandb_url = wandb.run.get_url()
