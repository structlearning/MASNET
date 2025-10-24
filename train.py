import argparse
from argparse import ArgumentParser, Namespace
from easydict import EasyDict
import torch
from torch_geometric.loader import DataLoader
from lightning.pytorch.trainer import Trainer                                                                                                                                                              
import os
import sys
import gc
import pickle
"""
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
from models.lightning_model import LightningModel
from utils import (
    get_args, create_model_dir,
    create_trainer, MetricAggregationCallback, fix_seed, get_data
)

def parse_arguments() -> Namespace:
    """
    Parse command-line arguments for dataset training configuration.

    Returns:
        EasyDict: Parsed arguments in an easy-to-use dictionary.
    """
    ap = argparse.ArgumentParser(description="Train graph models on specified datasets.")
    ap.add_argument('--task_type',                      type=str,   default='RealWorldSubSet')
    ap.add_argument('--containment_type',               type=str,   default = 'hard')
    ap.add_argument('--model_type',                     type=str,   default='MonotoneNet')
    ap.add_argument('--hat_act',                        type=str,   default = 'trainable_hat')
    ap.add_argument('--distributed',                    type=bool,  default = False)
    ap.add_argument("--want_cuda",                      type=bool,  default=True)
    ap.add_argument("--has_cuda",                       type=bool,  default=torch.cuda.is_available())
    ap.add_argument("--SKEW",                           type=int,   default=0)
    ap.add_argument("--VAL_QUERIES",                    type=int,   default=100)
    ap.add_argument("--NUM_QUERIES",                    type=int,   default=500)
    ap.add_argument("--TRAIN_DATASET_SIZE",             type=int,   default = 25000)
    ap.add_argument("--VAL_DATASET_SIZE",               type=int,   default = 10000)
    ap.add_argument("--TEST_DATASET_SIZE",              type=int,   default = 10000)
    ap.add_argument("--P2N",                            type=float, default=1.0)
    ap.add_argument("--noise_scale",                    type=float, default=0.001)
    ap.add_argument("--min_query_size",                 type=int,   default=10)
    ap.add_argument("--max_query_size",                 type=int,   default=30)
    ap.add_argument("--min_corpus_size",                type=int,   default=40)
    ap.add_argument("--max_corpus_size",                type=int,   default=60)
    ap.add_argument("--DATASET_NAME",                   type=str,   default="NFCORPUS", help="MSWEB/MSNBC/NFCORPUS/AMAZON/POINTCLOUD")
    ap.add_argument("--EmbedType",                      type=str,   default="Bert768", help="OneHot/Bert768")
    ap.add_argument("--DEVICE",                         type=int,   default=0)
    ap.add_argument('--delta_loss',                     type=float, default=0.1)
    ap.add_argument('--delta_inf',                      type=float, default=0.01)
    ap.add_argument("--lr",                             type=float, default=1e-6)
    ap.add_argument("--lr_factor",                      type=float, default=0.7)
    ap.add_argument("--patience",                       type=int,   default=10)
    ap.add_argument("--alpha",                          type=float, default=4.55)
    ap.add_argument("--temp",                           type=float, default=2.50)
    ap.add_argument("--int_temp",                       type=float, default=25.00)
    ap.add_argument("--use_int_alpha",                  type=bool,  default=False)
    ap.add_argument("--pointcloud_encoder",             type=str,   default='pointnet')
    ap.add_argument("--dgcnn_k",                        type=int,   default=3)
    ap.add_argument("--bs",                             type=int,   default=64)
    ap.add_argument("--no_divide_by_c",                 type=bool,  default=False)
    ap.add_argument("--no_outer_rho", action="store_true", help="Disable outer rho")
    ap.add_argument("--monotone_m2",  action="store_true", help="Enable monotone m2")
    ap.add_argument("--monotonic_m2_hidden_factor",     type = list[int], default = [10,  10])
    ap.add_argument("--use_shallow",                    type=bool,  default=False)
    ap.add_argument("--model_path",                     type=str,   default=None)   
    ap.add_argument("--num_channels",                    type=int,   default=50) 
    ap.add_argument("--s1_size",                         type = int, default = 512)
    ap.add_argument("--unittest_dataset",                type=bool,  default = False)
    ap.add_argument("--d",                               type=int,   default = 768)
    ap.add_argument("--file_name_suffix",                 type=str,   default = "results_tuning")
    return ap.parse_args() #argparse.Namespace object


def train_graphs(args: EasyDict, task_specific: str, metric_callback) -> tuple:
    """
    Train and evaluate a graph model on a dataset.

    Args:
        args (EasyDict): Experiment configuration.
        task_specific (str): Task-specific identifier.
        metric_callback (MetricAggregationCallback): Callback for tracking metrics.
        measure_oversmoothing (bool): Whether to measure oversmoothing.

    Returns:
        tuple: (Test accuracy, MAD energy if oversmoothing is measured)
    """
    trainer, checkpoint_callback = create_trainer(args, task_specific, metric_callback)
    # Load dataset
    
    train_loader, val_loader, test_loader = get_data(args)
    print("data loaded")
    # Initialize graph model
    model = LightningModel(args=args)

    # Train model
    print('Starting training...')
    trainer.fit(model, train_loader, val_loader)
    print(f"Training done...\nBest Model in: {checkpoint_callback.best_model_path}\n")
    best_model_path = checkpoint_callback.best_model_path
    
    # Load best checkpoint if necessary
    if not args.take_last and best_model_path and os.path.isfile(best_model_path):
        print(f"Loading best model checkpoint...from: {checkpoint_callback.best_model_path}\n")
        model = LightningModel.load_from_checkpoint(
            checkpoint_callback.best_model_path, args=args, model=model.model
        )
    else:
        if not best_model_path or not os.path.isfile(best_model_path):
            print(f"No valid checkpoint found. Using current model state.")
        else:
            print(f"Using last model state as specified in args.")

    # Test model
    test_results = trainer.test(model, test_loader, verbose=False)
    test_acc_model = test_results[0]['test_acc'] * 100
    test_fp_model =  test_results[0]['test_false_pos_rate'] * 100
    test_fn_model = test_results[0]['test_false_neg_rate'] * 100
    print(f"Model testing done")

    return test_acc_model, test_fp_model, test_fn_model

def test_model(args, task_specific):
    model = LightningModel(args=args)
    train_loader, val_loader, test_loader = get_data(args)
    if not os.path.exists(args.model_path):
        print("No valid path found")
        return None
    else: 
        model = LightningModel.load_from_checkpoint(args.model_path, args=args, model=model.model)
        model_dir, _ = create_model_dir(args, task_specific)
        trainer = Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=[args.DEVICE],
            enable_progress_bar=True,
            default_root_dir=f'{model_dir}/lightning_logs'
        )
        test_results = trainer.test(model, test_loader, verbose=False)
        acc_dict = test_results[0]
        acc = acc_dict['test_acc'] * 100
        fp_rate = acc_dict['test_false_pos_rate'] * 100
        fn_rate = acc_dict['test_false_neg_rate'] * 100
        return acc, fp_rate, fn_rate
        

def tune_alpha_temp():
    """this function has issues
       need to write a clean tuning code over weekend
       #TODO
    """
    model_type, task_type, distributed, hat_act = 'MonotoneNet', 'SubSet', False, 'param'
    args, task_specific = get_args(model_type=model_type, task_type=task_type, distributed = distributed, hat_act = hat_act)
    metric_callback = MetricAggregationCallback(eval_every=args.eval_every)
    fix_seed(args.seed)
    temp_range = [1.2 + 0.05 * n for n in range(20)]
    alpha_range = [6.0 + 0.1 * n for n in range(10)]
    tuning_dir = os.path.join(os.getcwd(), 'data/models/SubSet/MonotoneNet/HyperTune')
    if not os.path.exists(tuning_dir): os.makedirs(tuning_dir)
    res_file = os.path.join(tuning_dir, 'alpha_6to7_temp_all.text')
    with open(res_file, "a") as file:
        for alpha in alpha_range:
            for temp in temp_range:
                args.alpha = alpha
                args.temp = temp
                test_res = train_graphs(args, task_specific, metric_callback)
                file.write(f"\ntask: {task_type}, model: {model_type}, m: {args.m}, n: {args.n}, d: {args.d}, alpha: {alpha}, temp: {temp}, test_acc: {test_res:.2f}%\n")
                file.flush()
                
def tune_temp(dataset_name: str, device: int):
    hat_act = "trainable_hat"
    lr = 5e-5
    noise_scale = 0
    args_basic = parse_arguments()
    args_basic.lr = lr
    args_basic.DATASET_NAME = dataset_name
    args_basic.noise_scale = noise_scale
    args_basic.hat_act = hat_act
    args_basic.eval_every=5
    args_basic.DEVICE = device
    fix_seed(0)
    train_dl, val_dl, test_dl = get_data(args_basic)
    print("data loaded")
    #temp_range = [0.6 + 0.4 * i for i in range(20)]
    refined_msweb_temp_range = [0.6 + 0.06 * i for i in range(20)]
    metric_callback = MetricAggregationCallback(eval_every=5)
    tuning_dir = os.path.join(os.getcwd(), 'HyperTune')
    if not os.path.exists(tuning_dir): os.makedirs(tuning_dir)
    res_file = os.path.join(tuning_dir, f'finer_temp_tune_abs_scale_width_{dataset_name}_trainable_hat.text')
    with open(res_file, "a") as file:
        for temp in refined_msweb_temp_range:
            args_basic.temp = temp
            args_refined, task_specific = get_args(**vars(args_basic))
            trainer, checkpoint_callback = create_trainer(args_refined, task_specific, metric_callback)
            model = LightningModel(args=args_refined)
            trainer.fit(model, train_dl, val_dl)
            best_model_path = checkpoint_callback.best_model_path
            if os.path.isfile(best_model_path):  
                model = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path, args=args_refined, model=model.model)
            test_results = trainer.test(model, test_dl, verbose=False)
            test_acc = test_results[0]['test_acc'] * 100
            test_fp =  test_results[0]['test_false_pos_rate'] * 100
            test_fn = test_results[0]['test_false_neg_rate'] * 100
            file.write(f"\n temp: {temp}, test_acc: {test_acc:.2f}%, false_pos: {test_fp:.2f}%, false_neg: {test_fn:.2f}%\n")
            file.flush()
            del model
            del trainer
            del checkpoint_callback
            gc.collect()
            torch.cuda.empty_cache()
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
            
            
def tune_temp_integral(dataset_name: str, device: int):
    hat_act = "double_integral_hat"
    lr = 5e-5
    noise_scale = 0
    args_basic = parse_arguments()
    args_basic.lr = lr
    args_basic.DATASET_NAME = dataset_name
    args_basic.noise_scale = noise_scale
    args_basic.hat_act = hat_act
    args_basic.eval_every=5
    args_basic.DEVICE = device
    fix_seed(0)
    train_dl, val_dl, test_dl = get_data(args_basic)
    print("data loaded")
    #refined_amazon_temp_range = [9.5 + 0.5 * i for i in range(8)] + [18.5 + 0.5 * i for i in range(8)]
    temp_range = [5 + i * 3 for i in range(9)]
    metric_callback = MetricAggregationCallback(eval_every=5)
    tuning_dir = os.path.join(os.getcwd(), 'HyperTune')
    if not os.path.exists(tuning_dir): os.makedirs(tuning_dir)
    res_file = os.path.join(tuning_dir, f'temp_tune_abs_scale_width_{dataset_name}_integral_hat_bothpos.text')
    with open(res_file, "a") as file:
        for temp in temp_range:
            args_basic.int_temp = temp
            args_refined, task_specific = get_args(**vars(args_basic))
            trainer, checkpoint_callback = create_trainer(args_refined, task_specific, metric_callback)
            model = LightningModel(args=args_refined)
            trainer.fit(model, train_dl, val_dl)
            best_model_path = checkpoint_callback.best_model_path
            if os.path.isfile(best_model_path):  
                model = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path, args=args_refined, model=model.model)
            test_results = trainer.test(model, test_dl, verbose=False)
            test_acc = test_results[0]['test_acc'] * 100
            test_fp =  test_results[0]['test_false_pos_rate'] * 100
            test_fn = test_results[0]['test_false_neg_rate'] * 100
            file.write(f"\n temp: {temp}, test_acc: {test_acc:.2f}%, false_pos: {test_fp:.2f}%, false_neg: {test_fn:.2f}%\n")
            file.flush()
            del model
            del trainer
            del checkpoint_callback
            gc.collect()
            torch.cuda.empty_cache()
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
            
def ablation_studies_deepsets(device: int, noise_scale: float):
    amazon_dataset_names = ['feeding', 'bedding']
    #amazon_list = []
    amazon_list = [f"AMAZON_{dataset}" for dataset in amazon_dataset_names]
    amazon_datasets = amazon_list + ['MSWEB', 'MSNBC']
    basic_args = parse_arguments()
    basic_args.model_type = 'DeepSets'
    basic_args.DEVICE = device
    basic_args.noise_scale = noise_scale
    fix_seed(0)
    metric_callback = MetricAggregationCallback(eval_every=5)
    dir_name = os.path.join(os.getcwd(), 'Ablation')
    if not os.path.exists(dir_name): os.makedirs(dir_name)
    file_name = f"all_datasets_deepsets_noise_{noise_scale}.text"
    res_file = os.path.join(dir_name, file_name)
    use_rho_list = [True, False]
    with open(res_file, "a") as file:
        for use_rho in use_rho_list:
            for dataset in amazon_datasets:
                basic_args.no_outer_rho = not use_rho
                basic_args.DATASET_NAME = dataset
                args_refined, task_specific = get_args(**vars(basic_args))
                trainer, checkpoint_callback = create_trainer(args_refined, task_specific, metric_callback)
                train_dl, val_dl, test_dl= get_data(args_refined)
                model = LightningModel(args=args_refined)
                trainer.fit(model, train_dl, val_dl)
                best_model_path = checkpoint_callback.best_model_path
                if os.path.isfile(best_model_path):  
                    model = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path, args=args_refined, model=model.model)
                test_results = trainer.test(model, test_dl, verbose=False)
                test_acc = test_results[0]['test_acc'] * 100
                test_fp =  test_results[0]['test_false_pos_rate'] * 100
                test_fn = test_results[0]['test_false_neg_rate'] * 100
                file.write(f"\n Dataset: {dataset}, Outer_present:{use_rho},  test_acc: {test_acc:.2f}%, false_pos: {test_fp:.2f}%, false_neg: {test_fn:.2f}%\n")
                file.flush()
                model = model.to('cpu')
                for dl in [train_dl, val_dl, test_dl]:
                    if hasattr(dl, '_iterator'):
                        dl._iterator = None
                    if hasattr(dl, 'worker_init_fn') and dl.worker_init_fn is not None:
                        dl.worker_init_fn = None
                del model
                del trainer
                del checkpoint_callback
                del train_dl
                del val_dl
                del test_dl
                for _ in range(3): gc.collect()
                torch.cuda.empty_cache()
                print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
                print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
                
def get_synthetic_plots_data(model_name: str, noise_scale: float, device : int, p2n: float):
    model_type = model_name
    hat_act = 'hat'
    no_outer_rho = False
    use_shallow = False
    if model_name not in ['DeepSets', 'SetTransformer']:
        name_to_model = {'ShallowTri': 'MonotoneNet', 'ShallowHat': 'MonotoneNet', 'ShallowReLU': 'DeepSets', 'DeepTri': 'MonotoneNet', 'DeepHat': 'MonotoneNet', 'DeepReLU': 'DeepSets'}
        name_to_hat = {'ShallowTri': 'hat', 'ShallowHat': 'trainable_hat', 'ShallowReLU': 'hat', 'DeepTri': 'hat', 'DeepHat': 'trainable_hat', 'DeepReLU': 'hat'}
        if model_name not in name_to_model.keys(): 
            raise ValueError(f"Supported model names are: {list(name_to_model.keys())}, Given: {model_name}")
        model_type = name_to_model[model_name]
        hat_act = name_to_hat[model_name]
        no_outer_rho = True
        use_shallow = (model_name[:7] == 'Shallow')
    query_set_size = 10
    corpus_set_sizes = [10 * (i+1) for i in range(20)]
    basic_args = parse_arguments()
    basic_args.model_type = model_type
    basic_args.hat_act = hat_act
    basic_args.use_shallow
    basic_args.no_outer_rho = no_outer_rho
    basic_args.use_shallow = use_shallow
    basic_args.task_type = 'SyntheticSubSet'
    basic_args.noise_scale = noise_scale
    basic_args.DEVICE = device
    basic_args.min_query_size = query_set_size
    basic_args.max_query_size = query_set_size
    basic_args.P2N = p2n
    fix_seed(0)
    metric_callback = MetricAggregationCallback(eval_every=5)
    dir_name = os.path.join(os.getcwd(), 'Plotdata')
    if not os.path.exists(dir_name): os.makedirs(dir_name)
    res_file_name = f"syntheticVcorpus_model_{model_name}_noise_{noise_scale}.pkl"
    res_file = os.path.join(dir_name, res_file_name)
    res_dict = {'test_acc': [], 'test_fp': [], 'test_fn': [], 'corpus': []}
    for corpus_set_size in corpus_set_sizes:
        basic_args.min_corpus_size = corpus_set_size
        basic_args.max_corpus_size = corpus_set_size
        args_refined, task_specific = get_args(**vars(basic_args)) 
        trainer, checkpoint_callback = create_trainer(args_refined, task_specific, metric_callback)
        train_dl, val_dl, test_dl= get_data(args_refined)
        model = LightningModel(args=args_refined)
        trainer.fit(model, train_dl, val_dl)
        best_model_path = checkpoint_callback.best_model_path
        if os.path.isfile(best_model_path):  
            model = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path, args=args_refined, model=model.model)
        test_results = trainer.test(model, test_dl, verbose=False)
        test_acc = test_results[0]['test_acc'] * 100
        test_fp =  test_results[0]['test_false_pos_rate'] * 100
        test_fn = test_results[0]['test_false_neg_rate'] * 100
        res_dict['test_acc'].append(test_acc)
        res_dict['test_fp'].append(test_fp)
        res_dict['test_fn'].append(test_fn)
        res_dict['corpus'].append(corpus_set_size)
        with open(res_file, 'wb') as file: pickle.dump(res_dict, file)
        model = model.to('cpu')
        for dl in [train_dl, val_dl, test_dl]:
            if hasattr(dl, '_iterator'):
                dl._iterator = None
            if hasattr(dl, 'worker_init_fn') and dl.worker_init_fn is not None:
                dl.worker_init_fn = None
        del model
        del trainer
        del checkpoint_callback
        del train_dl
        del val_dl
        del test_dl
        for _ in range(3): gc.collect()
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
            
def ablation_studies_hatfn(device: int, noise_scale: float, hat_type: str):
    supported_hat_types = ['hat', 'trainable_hat', 'double_integral_hat']
    if hat_type not in supported_hat_types:
        raise ValueError(f"Requested hat: {hat_type} not supported")
    amazon_dataset_names = ['feeding', 'bedding']
    amazon_list = [f"AMAZON_{dataset}" for dataset in amazon_dataset_names]
    amazon_datasets = ['MSNBC', 'MSWEB'] + amazon_list
    basic_args = parse_arguments()
    print("starting")
    basic_args.model_type = 'MonotoneNet'
    basic_args.DEVICE = device
    basic_args.noise_scale = noise_scale
    basic_args.hat_act = hat_type
    fix_seed(0)
    metric_callback = MetricAggregationCallback(eval_every=5)
    hat_to_model_map = {'hat': 'MASNET_TRI', 'trainable_hat': 'MASNET_HAT', 'double_integral_hat': 'MASNET_INT'}
    model_name_in_file = hat_to_model_map[hat_type]
    dir_name = os.apth.join(os.getcwd(),'Ablation')
    if not os.path.exists(dir_name): os.makedirs(dir_name)
    if hat_type not in ['double_integral_hat', 'trainable_hat']:
        file_name = f"all_datasets_{model_name_in_file}_noise_{noise_scale}_RERUN.pkl"
    elif hat_type == 'double_integral_hat':
        file_name = f"all_datasets_{model_name_in_file}_temp_{basic_args.int_temp}_noise_{noise_scale}_RERUN.pkl"
    elif hat_type == 'trainable_hat':
        file_name = f"all_datasets_{model_name_in_file}_temp_{basic_args.temp}_noise_{noise_scale}_RERUN.pkl"
    res_file = os.path.join(dir_name, file_name)
    
    results_summary = {dataset: {'acc': None, 'FP': None, 'FN': None} for dataset in amazon_datasets}
    for dataset in amazon_datasets:
        basic_args.DATASET_NAME = dataset
        args_refined, task_specific = get_args(**vars(basic_args))
        trainer, checkpoint_callback = create_trainer(args_refined, task_specific, metric_callback)
        train_dl, val_dl, test_dl= get_data(args_refined)
        model = LightningModel(args=args_refined)
        trainer.fit(model, train_dl, val_dl)
        best_model_path = checkpoint_callback.best_model_path
        if os.path.isfile(best_model_path):  
            model = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path, args=args_refined, model=model.model)
        test_results = trainer.test(model, test_dl, verbose=False)
        test_acc = test_results[0]['test_acc'] * 100
        test_fp =  test_results[0]['test_false_pos_rate'] * 100
        test_fn = test_results[0]['test_false_neg_rate'] * 100
        results_summary[dataset]['acc'] = test_acc
        results_summary[dataset]['FP'] = test_fp
        results_summary[dataset]['FN'] = test_fn
        with open(res_file, 'wb') as file: pickle.dump(results_summary, file)
        model = model.to('cpu')
        for dl in [train_dl, val_dl, test_dl]:
            if hasattr(dl, '_iterator'):
                dl._iterator = None
            if hasattr(dl, 'worker_init_fn') and dl.worker_init_fn is not None:
                dl.worker_init_fn = None
        del model
        del trainer
        del checkpoint_callback
        del train_dl
        del val_dl
        del test_dl
        for _ in range(3): gc.collect()
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
            
def run_set_transformer(device: int, noise_scale: float):
    amazon_dataset_names = ['feeding', 'bedding']
    amazon_list = [f"AMAZON_{dataset}" for dataset in amazon_dataset_names]
    amazon_datasets = amazon_list + ['MSWEB', 'MSNBC']
    basic_args = parse_arguments()
    print("starting")
    basic_args.model_type = 'SetTransformer'
    basic_args.DEVICE = device
    basic_args.noise_scale = noise_scale
    fix_seed(0)
    metric_callback = MetricAggregationCallback(eval_every=5)
    dir_name = os.path.join(os.getcwd(), 'Ablation')
    if not os.path.exists(dir_name): os.makedirs(dir_name)
    file_name = f"ST_all_noise_{noise_scale}.text"
    res_file = os.path.join(dir_name, file_name)
    with open(res_file, "a") as file:
        for dataset in amazon_datasets:
            basic_args.DATASET_NAME = dataset
            args_refined, task_specific = get_args(**vars(basic_args))
            trainer, checkpoint_callback = create_trainer(args_refined, task_specific, metric_callback)
            train_dl, val_dl, test_dl= get_data(args_refined)
            model = LightningModel(args=args_refined)
            trainer.fit(model, train_dl, val_dl)
            best_model_path = checkpoint_callback.best_model_path
            if os.path.isfile(best_model_path):  
                model = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path, args=args_refined, model=model.model)
            test_results = trainer.test(model, test_dl, verbose=False)
            test_acc = test_results[0]['test_acc'] * 100
            test_fp =  test_results[0]['test_false_pos_rate'] * 100
            test_fn = test_results[0]['test_false_neg_rate'] * 100
            file.write(f"\n Dataset: {dataset}, test_acc: {test_acc:.2f}%, false_pos: {test_fp:.2f}%, false_neg: {test_fn:.2f}%\n")
            file.flush()
            model = model.to('cpu')
            for dl in [train_dl, val_dl, test_dl]:
                if hasattr(dl, '_iterator'):
                    dl._iterator = None
                if hasattr(dl, 'worker_init_fn') and dl.worker_init_fn is not None:
                    dl.worker_init_fn = None
            del model
            del trainer
            del checkpoint_callback
            del train_dl
            del val_dl
            del test_dl
            for _ in range(3): gc.collect()
            torch.cuda.empty_cache()
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

def main():
    """
    Main function to run training experiments for various model depths.
    """
    args = parse_arguments()
    args, task_specific = get_args(**vars(args))
    #args loaded and combined from task_configs and command line
    metric_callback = MetricAggregationCallback(eval_every=args.eval_every)
    results_file = f"/mnt/nas/soutrik/Monotone-Clean/{args.file_name_suffix}.txt"
    fix_seed(args.seed)
    if args.model_path is None: 
        test_acc, test_fp, test_fn = train_graphs(args, task_specific, metric_callback)
    else:
        test_acc, test_fp, test_fn = test_model(args, task_specific)
    if args.task_type == 'SyntheticSubSet' and args.model_type == 'MonotoneNet': 
        print(f"\nDATASET: {args.task_type}, MODEL TYPE: {args.model_type}, HAT_TYPE: {args.hat_act},  min_query: {args.min_query_size}, max_query: {args.max_query_size}, min_corpus: {args.min_corpus_size}, max_corpus: {args.max_corpus_size}, d: {args.d}") 
        print(f"TEST ACC: {test_acc:.2f}%, TEST FP: {test_fp:.2f}%, TEST FN: {test_fn:.2f}%\n")
    elif args.task_type == 'SyntheticSubSet' and args.model_type == 'DeepSets':
        outer_rho_used = not args.no_outer_rho
        print(f"\nDATASET: {args.task_type}, MODEL TYPE: {args.model_type}, outer_rho: {outer_rho_used}, min_query: {args.min_query_size}, max_query: {args.max_query_size}, min_corpus: {args.min_corpus_size}, max_corpus: {args.max_corpus_size}, d: {args.d}") 
        print(f"TEST ACC: {test_acc:.2f}%, TEST FP: {test_fp:.2f}%, TEST FN: {test_fn:.2f}%\n")
    elif args.task_type == 'SyntheticSubSet':
        print(f"\nDATASET: {args.task_type}, MODEL TYPE: {args.model_type},  min_query: {args.min_query_size}, max_query: {args.max_query_size}, min_corpus: {args.min_corpus_size}, max_corpus: {args.max_corpus_size}, d: {args.d}") 
        print(f"TEST ACC: {test_acc:.2f}%, TEST FP: {test_fp:.2f}%, TEST FN: {test_fn:.2f}%\n")
    elif  args.model_type == 'MonotoneNet':
        line = (
        f"ALPHA:{args.alpha}, TEMP:{args.temp}, DATASET:{args.DATASET_NAME}, "
        f"MODEL: MASNET, HAT:{args.hat_act}, AGG:{args.agg}, "
        f"CHANNELS:{args.num_channels}, "
        f"TEST_ACC:{test_acc:.2f}%, TEST_FP:{test_fp:.2f}%, TEST_FN:{test_fn:.2f}%\n"
        )
        with open(results_file, "a") as file:
            file.write(line)
            file.flush()
        #print(f"\nDATASET: {args.DATASET_NAME}, MODEL TYPE: {args.model_type}, HAT_TYPE: {args.hat_act}, AGG_TYPE: {args.agg}, CHANNELS: {args.num_channels}, TEST ACC: {test_acc:.2f}%, TEST FP: {test_fp:.2f}%, TEST FN: {test_fn:.2f}%\n")
    elif args.model_type == 'DeepSets':
        using_masnet = args.no_outer_rho or args.monotone_m2
        model_name = 'DeepSets' if not using_masnet else 'MasNet_ReLU'
        outer_monotone = "No OuterMonotone" if not args.monotone_m2 else "With OuterMonotone"
        line = None
        ptcloud_string = f"s1 Size: {args.s1_size}, ptcloud encoder: {args.pointcloud_encoder}" if args.DATASET_NAME == "POINTCLOUD" else ""
        if not using_masnet:
            line = (f"DATASET:{args.DATASET_NAME} " + ptcloud_string + f" MODEL:{model_name}, num_channels:{args.num_channels}, AGG: {args.agg}, TEST_ACC:{test_acc:.2f}%, TEST_FP:{test_fp:.2f}%, TEST_FN:{test_fn:.2f}%\n")
        else:
            line = (f"DATASET:{args.DATASET_NAME} " + ptcloud_string + f" MODEL:{model_name}, num_channels:{args.num_channels}, AGG: {args.agg}, Outer Monotone MLP: {outer_monotone}, TEST_ACC :{test_acc:.2f}%, TEST_FP:{test_fp:.2f}%, TEST_FN:{test_fn:.2f}%\n")
        with open(results_file, "a") as file:
            file.write(line)
            file.flush()
    elif args.model_type == 'SetTransformer':
        ptcloud_string = f"s1 Size: {args.s1_size}, ptcloud encoder: {args.pointcloud_encoder}" if args.DATASET_NAME == "POINTCLOUD" else ""
        line = (f"DATASET:{args.DATASET_NAME} " + ptcloud_string + f" MODEL:{args.model_type}, num_channels:{args.num_channels}, TEST_ACC:{test_acc:.2f}%, TEST_FP:{test_fp:.2f}%, TEST_FN:{test_fn:.2f}%\n")
        with open(results_file, "a") as file:
            file.write(line)
            file.flush()
    else:
        print(f"\nDATASET: {args.DATASET_NAME}, MODEL TYPE: {args.model_type},  CHANNELS: {args.out_dim},TEST ACC: {test_acc:.2f}%, TEST FP: {test_fp:.2f}%, TEST FN: {test_fn:.2f}%\n")
        
def shallow_and_deep_inference(noise_scale: float, device: int, hat_type: str, use_shallow: bool):
    amazon_dataset_names = ['media', 'safety', 'toys', 'health', 'gear', 'feeding', 'diaper', 'bedding', 'bath', 'apparel']
    #amazon_list = []
    amazon_list = [f"AMAZON_{dataset}" for dataset in amazon_dataset_names]
    amazon_datasets = amazon_list + ['MSWEB', 'MSNBC']
    basic_args = parse_arguments()
    print("starting")
    basic_args.model_type = 'MonotoneNet'
    basic_args.DEVICE = device
    basic_args.noise_scale = noise_scale
    basic_args.hat_act = hat_type
    basic_args.use_shallow = use_shallow
    fix_seed(0)
    metric_callback = MetricAggregationCallback(eval_every=5)
    dir_name = os.path.join(os.getcwd(), 'Ablation')
    if not os.path.exists(dir_name): os.makedirs(dir_name)
    file_name = f"shallow_{basic_args.use_shallow}_noise_{noise_scale}_hat_{hat_type}.text"
    res_file = os.path.join(dir_name, file_name)
    with open(res_file, "a") as file:
        for dataset in amazon_datasets:
            basic_args.DATASET_NAME = dataset
            args_refined, task_specific = get_args(**vars(basic_args))
            trainer, checkpoint_callback = create_trainer(args_refined, task_specific, metric_callback)
            train_dl, val_dl, test_dl= get_data(args_refined)
            model = LightningModel(args=args_refined)
            trainer.fit(model, train_dl, val_dl)
            best_model_path = checkpoint_callback.best_model_path
            if os.path.isfile(best_model_path):  
                model = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path, args=args_refined, model=model.model)
            test_results = trainer.test(model, test_dl, verbose=False)
            test_acc = test_results[0]['test_acc'] * 100
            test_fp =  test_results[0]['test_false_pos_rate'] * 100
            test_fn = test_results[0]['test_false_neg_rate'] * 100
            file.write(f"\n Dataset: {dataset}, test_acc: {test_acc:.2f}%, false_pos: {test_fp:.2f}%, false_neg: {test_fn:.2f}%\n")
            file.flush()
            model = model.to('cpu')
            for dl in [train_dl, val_dl, test_dl]:
                if hasattr(dl, '_iterator'):
                    dl._iterator = None
                if hasattr(dl, 'worker_init_fn') and dl.worker_init_fn is not None:
                    dl.worker_init_fn = None
            del model
            del trainer
            del checkpoint_callback
            del train_dl
            del val_dl
            del test_dl
            for _ in range(3): gc.collect()
            torch.cuda.empty_cache()
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
            
def baseline_inference(noise_scale: float, device: int, baseline: str):
    amazon_dataset_names = ['media', 'safety', 'toys', 'health', 'gear', 'feeding', 'diaper', 'bedding', 'bath', 'apparel']
    #amazon_list = []
    amazon_list = [f"AMAZON_{dataset}" for dataset in amazon_dataset_names]
    amazon_datasets = amazon_list + ['MSWEB', 'MSNBC']
    basic_args = parse_arguments()
    print("starting")
    basic_args.model_type = baseline
    basic_args.DEVICE = device
    basic_args.noise_scale = noise_scale
    fix_seed(0)
    metric_callback = MetricAggregationCallback(eval_every=5)
    dir_name = os.path.join(os.getcwd(), 'Ablation')
    if not os.path.exists(dir_name): os.makedirs(dir_name)
    file_name = f"baseline_{baseline}_noise_{noise_scale}.text"
    res_file = os.path.join(dir_name, file_name)
    with open(res_file, "a") as file:
        for dataset in amazon_datasets:
            basic_args.DATASET_NAME = dataset
            args_refined, task_specific = get_args(**vars(basic_args))
            trainer, checkpoint_callback = create_trainer(args_refined, task_specific, metric_callback)
            train_dl, val_dl, test_dl= get_data(args_refined)
            model = LightningModel(args=args_refined)
            trainer.fit(model, train_dl, val_dl)
            best_model_path = checkpoint_callback.best_model_path
            if os.path.isfile(best_model_path):  
                model = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path, args=args_refined, model=model.model)
            test_results = trainer.test(model, test_dl, verbose=False)
            test_acc = test_results[0]['test_acc'] * 100
            test_fp =  test_results[0]['test_false_pos_rate'] * 100
            test_fn = test_results[0]['test_false_neg_rate'] * 100
            file.write(f"\n Dataset: {dataset}, test_acc: {test_acc:.2f}%, false_pos: {test_fp:.2f}%, false_neg: {test_fn:.2f}%\n")
            file.flush()
            model = model.to('cpu')
            for dl in [train_dl, val_dl, test_dl]:
                if hasattr(dl, '_iterator'):
                    dl._iterator = None
                if hasattr(dl, 'worker_init_fn') and dl.worker_init_fn is not None:
                    dl.worker_init_fn = None
            del model
            del trainer
            del checkpoint_callback
            del train_dl
            del val_dl
            del test_dl
            for _ in range(3): gc.collect()
            torch.cuda.empty_cache()
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
            
def run_all_baselines(dataset_name: str, noise_scale: float, device: int):
    args_basic = parse_arguments()
    args_basic.lr = 1e-5
    args_basic.DATASET_NAME = dataset_name
    args_basic.noise_scale = noise_scale
    args_basic.DEVICE = device
    args_basic.eval_every=5
    fix_seed(0)
    train_dl, val_dl, test_dl = get_data(args_basic)
    print("data loaded")
    metric_callback = MetricAggregationCallback(eval_every=5)
    tuning_dir = os.path.join(os.getcwd(), 'Baselines')
    if not os.path.exists(tuning_dir): os.makedirs(tuning_dir)
    res_file = os.path.join(tuning_dir, f'baselines_{dataset_name}_noise_{noise_scale}.text')
    model_configs = {'Deep'}
    with open(res_file, "a") as file:
        for model_config in model_configs:
            
            args_refined, task_specific = get_args(**vars(args_basic))
            trainer, checkpoint_callback = create_trainer(args_refined, task_specific, metric_callback)
            model = LightningModel(args=args_refined)
            trainer.fit(model, train_dl, val_dl)
            best_model_path = checkpoint_callback.best_model_path
            if os.path.isfile(best_model_path):  
                model = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path, args=args_refined, model=model.model)
            test_results = trainer.test(model, test_dl, verbose=False)
            test_acc = test_results[0]['test_acc'] * 100
            test_fp =  test_results[0]['test_false_pos_rate'] * 100
            test_fn = test_results[0]['test_false_neg_rate'] * 100
            file.write(f"\n temp: {temp}, test_acc: {test_acc:.2f}%, false_pos: {test_fp:.2f}%, false_neg: {test_fn:.2f}%\n")
            file.flush()
            del model
            del trainer
            del checkpoint_callback
            gc.collect()
            torch.cuda.empty_cache()
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

if __name__ == "__main__":
    #main()
    #device = 2
    #dataset_name = "MSWEB"
    #tune_temp(dataset_name, device)
    #tune_temp_integral(dataset_name, device)
    """
    noise_scale = 0
    use_rho = True
    ablation_studies_deepsets(use_rho, device, noise_scale)
    """
    """
    device = 1
    divide_by_c = False
    hat_type = 'relu'
    noise_scale = 0
    ablation_studies_hatfn(divide_by_c, hat_type, device, noise_scale)
    """
    """
    noise_scale = 0
    #hat_type = 'relu'
    device = 0
    run_set_transformer(noise_scale, device)
    """
    """
    noise_scale = 0.1
    device = 0
    #ablation_studies_deepsets(device, noise_scale)
    ablation_studies_hatfn(device, noise_scale)
    #run_set_transformer(device, noise_scale)
    
    device = 5
    model_name = 'ShallowHat'
    noise_scale = 0.01
    p2n = 1
    get_synthetic_plots_data(model_name, noise_scale, device, p2n)
    """
    main()
#'ShallowTri', 'ShallowHat', 'ShallowReLU', 'DeepSets', 'SetTransformer'
#hat, trainable_hat, double_integral_hat