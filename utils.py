import pathlib
from easydict import EasyDict
import yaml
import random 
import os 
import torch.nn.functional as F
"""
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer                                                                                                                                                              
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from data.synthetic_datasets import return_subset_dataset
from models.monotone_models import MonotneodeModel, InvariantDeepSets, SetTransformer, IndepOneWayMonotone, NeuralSFE
from data.msweb_data_generator import return_msweb_dataset
from data.beir_datasets import get_beir_datasets
from data.amazon import get_amazon_datasets
from data.pointcloud import get_pointcloud_dataset

class MetricAggregationCallback(Callback):
    def __init__(self, eval_every=5):
        super().__init__()
        self.eval_every = eval_every
        self.all_val_metrics = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        val_acc = trainer.callback_metrics.get("val_acc", None)
        if val_acc is not None:
            if epoch not in self.all_val_metrics:
                self.all_val_metrics[epoch] = []
            self.all_val_metrics[epoch].append(val_acc.item() * 100)

    def get_best_epoch(self):
        if not self.all_val_metrics:
            return None, None, None  # Handle case where no metrics exist

        avg_metrics = {epoch: np.mean(accs) for epoch, accs in self.all_val_metrics.items()}
        best_epoch = max(avg_metrics, key=avg_metrics.get)
        
        best_mean = np.mean(self.all_val_metrics[best_epoch])
        best_std = np.std(self.all_val_metrics[best_epoch])

        return best_mean, best_std

class StopAtValAccCallback(Callback):
    """
    Callback for early stopping when validation accuracy reaches a target value.
    
    Args:
        target_acc (float): Accuracy threshold for stopping training early.
    """
    def __init__(self, target_acc=1.1, type = 'hard', earlystopping_patience: int = 5):
        super().__init__()
        self.target_acc = target_acc
        self.min_acc = 0.05
        self.best_val_acc = 0
        self.last_val_acc = 0
        self.vals_without_improvement = 0
        self.patience_epochs = 6
        self.patience_acc_diff = 1e-4
        self.type = type

    def on_validation_epoch_end(self, trainer, _):
        """
        Checks validation accuracy at the end of each epoch, stopping training if the target is met.
        
        Args:
            trainer (Trainer): PyTorch Lightning trainer instance managing training.
        """
        val_acc = trainer.callback_metrics.get('val_acc')
        fp_rate = trainer.callback_metrics.get('val_false_pos_rate')
        fn_rate = trainer.callback_metrics.get('val_false_neg_rate')
        if self.type == 'hard' and val_acc is not None and val_acc >= self.target_acc:
            trainer.should_stop = True
            print(f"Stopping training as `val_acc` reached {val_acc * 100:.2f}%")
        elif val_acc < self.min_acc:
            trainer.should_stop = True
            print(f"Abandoning training as `val_acc` fell to {val_acc * 100:.2f}%")
        elif val_acc > 1.00:
            trainer.should_stop = True
            print(f"Stopping training as `val_acc` reached {val_acc * 100:.2f}%")
        elif self.type == 'hard':
            print(f"Current validation accuracy: {val_acc * 100:.2f}%")
            print(f"Current FP rate: {fp_rate * 100:.2f}%")
            print(f"Current FN rate: {fn_rate * 100:.2f}%")
        elif self.type == 'soft':
            print(f"Current validation mse: {val_acc:.2f}")
        
        if val_acc > self.best_val_acc:
            self.vals_without_improvement = 0
            self.best_val_acc = val_acc
            
        elif abs(self.last_val_acc - val_acc) < self.patience_acc_diff:
            self.vals_without_improvement += 1
            if self.vals_without_improvement > self.patience_epochs:
                trainer.should_stop = True
                print(f"Stopping training as `val_acc` not improved since {self.patience_epochs+1} evals%")
                
        self.last_val_acc = val_acc  

def create_trainer(args, task_specific,  metric_callback):
    model_dir, _ = create_model_dir(args, task_specific)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename='{epoch}-f{val_acc:.5f}',
        save_top_k=args.save_top_k,
        monitor='val_acc',
        save_last=True,
        mode='max')
    stop_callback = StopAtValAccCallback(type = args.containment_type, earlystopping_patience=(args.earlystopping_patience//args.eval_every)) 
    callbacks_list = [callback for callback in [checkpoint_callback, stop_callback, metric_callback] if callback]
    
    if not args.distributed:
        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices = [args.DEVICE],
            strategy = 'auto',
            enable_progress_bar=True,
            check_val_every_n_epoch=args.eval_every,
            callbacks=callbacks_list,
            default_root_dir=f'{model_dir}/lightning_logs'
        )
        
    else:
        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices = [args.DEVICE],
            enable_progress_bar=True,
            check_val_every_n_epoch=args.eval_every,
            callbacks=callbacks_list,
            default_root_dir=f'{model_dir}/lightning_logs'
        )
        
    return trainer, checkpoint_callback

def get_args(**kwargs):
    
    """
    Load and update arguments from a YAML configuration file.

    Args:
        depth (int): Depth of the model.
        gnn_type (str): Type of GNN layer.
        task_type (str): Task type for dataset generation.

    Returns:
        tuple: Configuration arguments and task-specific settings.
    """
    model_type, task_type = kwargs['model_type'], kwargs['task_type']
    #task_type = kwargs['DATASET_NAME'] if task_type == 'RealWorldSubSet' else task_type
    clean_args = EasyDict(model_type = model_type, task_type = task_type)
    config_path = pathlib.Path(__file__).parent / "configs/task_config.yaml"
    
    config = None
    with open(config_path) as f:
        config = EasyDict(yaml.safe_load(f))
        
    print("Available model types:", list(config['Task_specific'].keys()))
    print("Requested model type:", model_type)
    print("Requested task type:", task_type)


    # Update with general and task-specific configurations
    clean_args.update(config['Common'])
    clean_args.update(config['Task_specific'][model_type][task_type])
    clean_args.update(kwargs)
    model_name_configs = config['Task_specific'][model_type][task_type].copy()
    model_name_configs = {key: model_name_configs[key] for key in list(model_name_configs.keys())[:-5]}
    model_name_configs['d'] = kwargs['d']
    model_name_configs['num_channels'] = kwargs['num_channels']
    model_name_configs['noise_scale'] = kwargs['noise_scale']
    model_name_configs['bs'] = kwargs['bs']
    model_name_configs['is_shallow'] = kwargs['use_shallow']
    model_name_configs['num_channels'] = kwargs['num_channels']
    if kwargs['monotone_m2']:
        clean_args['monotonic_m2_hidden'] = [kwargs['num_channels'] * hidden_factor for hidden_factor in kwargs['monotonic_m2_hidden_factor']]
        print(f"Monotonic m2 hidden: {clean_args['monotonic_m2_hidden']}")
    if kwargs['task_type'] == 'SyntheticSubSet':
        model_name_configs['min_q'] = kwargs['min_query_size']
        model_name_configs['max_q'] = kwargs['max_query_size']
        model_name_configs['min_c'] = kwargs['min_corpus_size']
        model_name_configs['max_c'] = kwargs['max_corpus_size']
    if kwargs['use_shallow']:
        model_name_configs.pop('mlp_act', None)
        model_name_configs.pop('lrelu_slope', None)
        model_name_configs.pop('hidden_dims', None)
    if kwargs['model_type'] == 'MonotoneNet': 
        model_name_configs['hat_act'] = kwargs['hat_act']
        model_name_configs['no_divide_by_c'] = kwargs['no_divide_by_c']
        if model_name_configs['hat_act'] != 'double_integral_hat':
            model_name_configs.pop('integrand_hidden_dim', None)
            model_name_configs.pop('integrand_num_hidden_layers', None)
    if kwargs['model_type'] == 'MonotoneNet' and kwargs['hat_act'] == 'trainable_hat':
        model_name_configs['temp'] = kwargs['temp']
    elif kwargs['model_type'] == 'MonotoneNet' and kwargs['hat_act'] == 'double_integral_hat':
        model_name_configs['int_temp'] = kwargs['int_temp']
        
    if kwargs['model_type'] == 'DeepSets':
        model_name_configs['no_outer_rho'] = kwargs['no_outer_rho']
        model_name_configs['monotone_m2'] = kwargs['monotone_m2']
    if kwargs['DATASET_NAME'] != 'POINTCLOUD':
        model_name_configs.pop('pointcloud_encoder', None)
        
    else:
        model_name_configs['s1_size'] = kwargs['s1_size']
    
    return clean_args, model_name_configs

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
    
    if args.task_type == 'SyntheticSubSet':
        base_dir = path_to_project / "data" / "models" / args.task_type / args.model_type
    else:
        base_dir = path_to_project / "data" / "models" / args.DATASET_NAME / args.model_type
    
    model_dir = base_dir / model_name
    os.makedirs(str(base_dir), exist_ok=True)
    return str(model_dir), str(path_to_project)

def worker_init_fn(seed: int):
    np.random.seed(seed)
    random.seed(seed)

def fix_seed(seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        seed_everything(seed, workers=True)

def get_data(args):
    task_type = args.task_type
    dataset_name = args.DATASET_NAME
    containment_type = args.containment_type
    if task_type == 'SyntheticSubSet':
        train_data, val_data, test_data = return_subset_dataset(args)
    elif dataset_name in ['MSWEB', 'MSNBC']:
        args.TRAIN_DATASET_SIZE = 25000
        args.VAL_DATASET_SIZE = 10000
        args.TEST_DATASET_SIZE = 10000
        train_data, val_data, test_data = return_msweb_dataset(args)
    elif dataset_name[:6] == 'AMAZON':
        dataset_name = dataset_name[7:]
        train_data, val_data, test_data = get_amazon_datasets(dataset_name, noise_scale = args.noise_scale, pos_neg_ratio=args.P2N)
    elif dataset_name == 'POINTCLOUD':
        train_data, val_data, test_data = get_pointcloud_dataset(args)
    elif dataset_name == 'NFCORPUS':
        train_data, val_data, test_data = get_beir_datasets('nfcorpus', already_saved=True)
    elif dataset_name == 'HOTPOTQA':
        train_data, val_data, test_data = get_beir_datasets('hotpotqa', already_saved=True)
    elif dataset_name == 'SCIFACT':
        train_data, val_data, test_data = get_beir_datasets('scifact', already_saved=True)
    train_dl= DataLoader(train_data,batch_size=args.bs)
    test_dl= DataLoader(test_data,batch_size=args.bs)
    val_dl= DataLoader(val_data,batch_size=args.bs)
    return train_dl, val_dl, test_dl

def predict(f_S1, f_S2, delta):
    """
    Predicts labels based on coordinate-wise comparison of f(S1) and f(S2) using PyTorch.

    Parameters:
    - f_S1: torch tensor of shape (batch_size, num_channels) representing f(S1)
    - f_S2: torch tensor of shape (batch_size, num_channels) representing f(S2)
    - delta: threshold for weak dominance condition (scalar)

    Returns:
    - A tensor of shape (batch_size,) with values:
        - 1 if f(S1) <= f(S2) and exists i such that f(S1)_i + delta <= f(S2)_i
        - 0 otherwise
    """
    # Ensure tensors are of float type
    # f_S1, f_S2 = f_S1.float(), f_S2.float()

    # Check if f(S1) <= f(S2) coordinate-wise (for each batch)
    """
    cond_1 = torch.all(f_S1 <= f_S2 + delta, dim=1)

    # Assign label 1 where the conditions hold
    result = torch.ones(f_S1.shape[0], dtype=torch.long, device=f_S1.device)  # Default to 1
    result[cond_1] = 0  # Set to 0 if conditions hold
    """
    S1_minus_S2_hinge = torch.sum(F.relu(f_S1 - f_S2), axis = 1) #(batch size)
    result = (S1_minus_S2_hinge < delta).long()
    
    return result



def set_loss(f_S1, f_S2, delta, y, regularizer):
    """
    Computes the loss function:
        L(f(S1), f(S2)) = max(0, min_i [f(S2)_i - f(S1)_i + delta])
    
    Args:
        f_S1 (torch.Tensor): Tensor of shape (batch_size, d), representation of S1.
        f_S2 (torch.Tensor): Tensor of shape (batch_size, d), representation of S2.
        delta (float): Margin delta, default is 0.1.
    
    Returns:
        torch.Tensor: Loss value for each batch element.
    """
    # S1 \not \subset S2, and S2 \not \subset S1
    loss_1_more_than_2 = torch.clamp(delta + f_S2 - f_S1, min=0)
    loss_2_more_than_1 = torch.clamp(delta + f_S1 - f_S2, min=0)
    q_sub_c = 1.0
    #loss_not_subset = q_sub_c * loss_1_more_than_2.sum(dim=1)[0] + (1 - q_sub_c) * loss_2_more_than_1.sum(dim=1)[0]
    loss_not_subset = q_sub_c * loss_1_more_than_2.min(dim=1)[0] + (1 - q_sub_c) * loss_2_more_than_1.min(dim=1)[0]
    # when S1 \subset S2
    loss_subset = loss_2_more_than_1.max(dim=1)[0] 
        
    # Compute final loss using labels
    #If : S1 ⊆ S2, y = 1
    #If : S1 ⊈ S2, y = 0  
    loss = (1-y) * loss_not_subset + (y) * loss_subset
    loss = loss - 0.5 * regularizer * (torch.norm(f_S1)**2 + torch.norm(f_S2)**2)
    # Return avg over batch
    return loss.mean()

def soft_set_loss(f_S1, f_S2, y,  delta: float, regularizer: float, p: float = 10.1):
    """
    f_S1, f_S2: (batch size, num channels)
    y: (batch size)-> score between 0 and 1
    """
    loss_1_more_than_2 = torch.clamp(delta + f_S2 - f_S1, min=0) #(batch size, num channels)
    loss_2_more_than_1 = torch.clamp(delta + f_S1 - f_S2, min=0)
    loss_not_subset = loss_1_more_than_2.sum(dim=1)[0]
    loss_not_subset = torch.pow(loss_not_subset, p)
    #soft_score = lambda hinge_score: torch.div(hinge_score, 1 + hinge_score)
    soft_score = lambda hinge_score: 1 - torch.exp(-p*hinge_score)
    score = soft_score(loss_not_subset)
    loss_fn = nn.MSELoss()
    return loss_fn(score, y)

def accuracy_measure(fq, fc, score, type, delta):
    if type == 'hard':
        pred = predict(fq, fc, delta)
        acc = (pred == score).float().mean()
        fp = ((pred == 1) & (score == 0)).float().sum()
        fp_rate = fp / (score == 0).float().sum() if (score == 0).float().sum() > 0 else torch.tensor(0.0)
        fn = ((pred == 0) & (score == 1)).float().sum()
        fn_rate = fn / (score == 1).float().sum() if (score == 1).float().sum() > 0 else torch.tensor(0.0)
        acc_dict = {'accuracy': acc.item(), 'false_pos_rate': fp_rate.item(), 'false_neg_rate': fn_rate.item()}
        return acc_dict
    else:
        q_minus_c_hinge = torch.sum(F.relu(fq - fc - delta), axis = 1) #(batch size)
        #score_pred = torch.div(q_minus_c_hinge, 1 + q_minus_c_hinge)
        score_pred = 1 - torch.exp(-q_minus_c_hinge)
        loss = nn.MSELoss()
        return loss(score, score_pred)

def get_model(args):
    model_type = args.model_type
    if model_type == 'MonotoneNet':
        return MonotneodeModel(args)
    if model_type == 'DeepSets':
        return InvariantDeepSets(args)
    if model_type == 'SetTransformer':
        return SetTransformer(args)
    if model_type == 'OneWayMon':
        return IndepOneWayMonotone(args)
    if model_type == 'NeuralSFE':
        return NeuralSFE(args)