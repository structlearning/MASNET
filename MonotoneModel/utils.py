import pathlib
import torch
import yaml
from easydict import EasyDict
import numpy as np
from torch.utils.data import DataLoader
from SetModel.set_monotonicy_data import SimpleDataset, SubsetDataset, FacilityLocation
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
import random                                                                                                                                                                
from lightning.pytorch.callbacks import ModelCheckpoint
import os
from lightning.pytorch import seed_everything
from SetModel.monotne_models import MonotneodeSetModel, InvariantDeepSets, SetTransformer

class StopAtValAccCallback(Callback):
    """
    Callback for early stopping when validation accuracy reaches a target value.
    
    Args:
        target_acc (float): Accuracy threshold for stopping training early.
    """
    def __init__(self, target_acc=1.0):
        super().__init__()
        self.target_acc = target_acc

    def on_validation_epoch_end(self, trainer, _):
        """
        Checks validation accuracy at the end of each epoch, stopping training if the target is met.
        
        Args:
            trainer (Trainer): PyTorch Lightning trainer instance managing training.
        """
        val_acc = trainer.callback_metrics.get('val_acc')
        print(f"Current validation accuracy: {val_acc :.2f}")

def get_args(model_type: str, task_type: str):
    
    """
    Load and update arguments from a YAML configuration file.

    Args:
        depth (int): Depth of the model.
        gnn_type (str): Type of GNN layer.
        task_type (str): Task type for dataset generation.

    Returns:
        tuple: Configuration arguments and task-specific settings.
    """
    clean_args = EasyDict(model_type = model_type, task_type=task_type)
    config_path = pathlib.Path(__file__).parent / "configs/task_config.yaml"
    
    with open(config_path) as f:
        config = EasyDict(yaml.safe_load(f))

    # Update with general and task-specific configurations
    clean_args.update(config['Common'])
    clean_args.update(config['Task_specific'][model_type][task_type])
    return clean_args, config['Task_specific'][model_type][task_type]

def create_model_dir(args, task_specific):
    """
    Create a directory for model checkpoints and logs.

    Args:
        args (EasyDict): Configuration arguments.
        task_specific (dict): Task-specific settings.

    Returns:
        tuple: Model directory path and project base path.
    """
    model_name = '_'.join([f"{key}_{val}" for key, val in task_specific.items()])
    path_to_project = pathlib.Path(__file__).parent.parent
    model_dir = path_to_project / f"data/models/{args.task_type}/{args.model_type}/{model_name}"
    return str(model_dir), str(path_to_project)

def worker_init_fn(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    
def create_trainer(args, task_specific,  metric_callback):
    model_dir, _ = create_model_dir(args, task_specific)
    mode = 'min' if args.task_type == 'FacilityLocation' else 'max'
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename='{epoch}-f{val_acc:.5f}',
        save_top_k=10,
        monitor='val_acc',
        save_last=True,
        mode=mode)
    stop_callback = StopAtValAccCallback() 
    callbacks_list = [callback for callback in [checkpoint_callback, stop_callback, metric_callback] if callback]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar=True,
        check_val_every_n_epoch=args.eval_every,
        callbacks=callbacks_list,
        default_root_dir=f'{model_dir}/lightning_logs'
    )
    return trainer, checkpoint_callback

def fix_seed(seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        seed_everything(seed, workers=True)

def get_data(args):
    task_type = args.task_type
    if task_type == 'HardPair':
        set = SimpleDataset()
        dl = DataLoader(set)
        return dl, dl, dl
    if task_type == 'SubSet':
        train_ds = SubsetDataset(m = args.m, n = args.n, d = args.d, num_samples = args.num_train_samples)
        val_ds = SubsetDataset(m = args.m, n = args.n, d = args.d, num_samples = args.num_val_samples)
        test_ds = SubsetDataset(m = args.m, n = args.n, d = args.d, num_samples = args.num_val_samples)

    if task_type == 'FacilityLocation':
        train_ds = FacilityLocation(n = args.n, d = args.d, num_samples=args.num_train_samples)
        val_ds = FacilityLocation(n = args.n, d = args.d, num_samples=args.num_val_samples)
        test_ds = FacilityLocation(n = args.n, d = args.d, num_samples=args.num_val_samples)
    
    train_dl= DataLoader(train_ds,batch_size=args.bs,shuffle=True)
    test_dl= DataLoader(test_ds,batch_size=args.bs)
    val_dl= DataLoader(val_ds,batch_size=args.bs)
    return train_dl, val_dl, test_dl


def get_model(args):
    model_type = args.model_type
    if model_type == 'MasNet':
        return MonotneodeSetModel(args)
    if model_type == 'DeepSets':
        return InvariantDeepSets(args)
    if model_type == 'SetTransformer':
        return SetTransformer(args)
    if model_type == 'ReLUMasNet':
        return MonotneodeSetModel(args=args)
    