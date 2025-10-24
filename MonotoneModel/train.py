import argparse
from easydict import EasyDict

from lightning_model import LightningModel
from utils import (
    get_args,
    create_trainer, fix_seed, get_data
)

def parse_arguments() -> EasyDict:
    """
    Parse command-line arguments for dataset training configuration.

    Returns:
        EasyDict: Parsed arguments in an easy-to-use dictionary.
    """
    parser = argparse.ArgumentParser(description="Train graph models on specified datasets.")
    parser.add_argument(
        '--task_type',
        type=str,
        default='FacilityLocation',
        choices=['FacilityLocation', 'SubSet'],
        help='Task/dataset to train on.'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default='MasNet',
        choices=['ReLUMasNet', 'DeepSets', 'MasNet', 'SetTransformer'],
        help='Model architecture to use.'
    )
    
    return EasyDict(vars(parser.parse_args()))  # Convert argparse.Namespace to EasyDict


def train_graphs(args: EasyDict, task_specific: str, metric_callback=None,test_out_of_disterbution = True) -> tuple:
    """
    Train and evaluate a graph model on a dataset.

    Args:
        args (EasyDict): Experiment configuration.
        task_specific (str): Task-specific identifier.
        metric_callback (MetricAggregationCallback): Callback for tracking metrics.

    Returns:
        float: Test accuracy
    """
    trainer, checkpoint_callback = create_trainer(args, task_specific, metric_callback)
    # Load dataset
    train_loader, val_loader, test_loader = get_data(args)

    # Initialize graph model
    model = LightningModel(args=args)

    # Train model
    print('Starting training...')
    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint if necessary
    if not args.take_last:
        print("Loading best model checkpoint...")
        try:
            model = LightningModel.load_from_checkpoint(
            checkpoint_callback.best_model_path, args=args, model=model.model)
        except:
            pass

    # Test model
    test_results = trainer.test(model, test_loader, verbose=True)
    test_acc_model = test_results[0]['test_acc']
    out_test_acc_model = 0.0
    if test_out_of_disterbution:
           args.d = 2 * args.d
           train_loader, val_loader, test_loader = get_data(args)
           out_test_results = trainer.test(model, test_loader, verbose=True)
           out_test_acc_model = out_test_results[0]['test_acc'] * 100
    return test_acc_model, out_test_acc_model

def main(config) -> tuple:
    """
    Main function to run training experiments for various model depths.

    Returns:
        tuple: (m, n, test accuracy)
    """

    # Extract command-line arguments
    task, model_type = config.task_type, config.model_type

    # Set up arguments
    args, task_specific = get_args(model_type=model_type, task_type=task)
    args.m, args.n, args.d, test_out_of_disterbution = config.m, config.n, config.d, config.test_dist                                           
    fix_seed(args.seed)                                                                         
    test_acc,out_test_acc = train_graphs(args, task_specific, test_out_of_disterbution = test_out_of_disterbution)

    print(f"Model {model_type} | Task: {task} | m: {m}, n: {n} | Accuracy: {test_acc:.2f}| Out of distr Accuracy {out_test_acc:.2f}")
    
    return (m, n, d, test_acc,out_test_acc)

if __name__ == "__main__":
    args = parse_arguments()
    if args.task_type == 'FacilityLocation':
         list_m_n = [(0, 10, 20)]   
    else:
        list_m_n = [(1, 2, 4), (1,10,4),(10,30,4),(10,100,4)]           
    results = []                
    for (m, n, d) in list_m_n:
        args.n, args.m, args.d, args.test_dist = n, m, d, False
        results.append(main(args))

    print("\nFinal Results Summary:")
    for m, n, d, acc, out_acc in results:
        print(f"m: {m}, n: {n}, d:{d} -> Test Accuracy: {acc:.2f}| Out of distrbution, {out_acc:.2f}%")