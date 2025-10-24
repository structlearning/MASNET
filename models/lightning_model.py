import torch
from torch import Tensor
from easydict import EasyDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
import lightning
import torch
from utils import soft_set_loss, set_loss, predict, get_model, accuracy_measure

class LightningModel(lightning.LightningModule):
    """
    PyTorch Lightning Module for training and evaluating a graph neural network model on various datasets.
    
    Args:
        args (EasyDict): Configuration dictionary containing model parameters.
        model (GraphModel): The graph neural network model to be used.
    """
    def __init__(self, args: EasyDict):
        super().__init__()
        self.lr = args.lr
        self.lr_factor = args.lr_factor
        self.patience = args.patience
        self.optim_type = args.optim_type
        self.weight_decay = args.weight_decay
        self.task_type = args.task_type
        self.model = get_model(args=args)
        self.loss_fun = soft_set_loss if args.containment_type == 'soft' else set_loss
        self.loss_regularizer = args.reg
        self.delta_loss = args.delta_loss
        self.delta_inf = args.delta_inf
        self.containment_type = args.containment_type
    """
    def forward(self, X: Data) -> Tensor:
        #Forward pass through the model.
        return self.model(X)
    """
    """
    def compute_node_embedding(self, X: Data) -> Tensor:
        #Compute node embeddings for the input graph data.
        return self.model.compute_node_embedding(X)
    """

    def _shared_step(self, batch: Data, stage: str):
        """
        Generic step used for training, validation, and testing.
        
        Args:
            batch (Data): A batch of graph data.
            stage (str): One of "train", "val", or "test".
        
        Returns:
            Tensor: Computed loss.
        """
        set_1, mask_1, set_2, mask_2, label = batch
        assert len(set_1.shape) == len(set_2.shape) == 3, f"Actual set shapes- 1: {set_1.shape}, 2: {set_2.shape}"
        assert len(mask_1.shape) == len(mask_2.shape) == 2, f"Actual mask shapes- 1: {mask_1.shape}, 2: {mask_2.shape}"
        result1, result2 = self.model(set_1, mask_1, set_2, mask_2)
        loss = self.loss_fun(result1, result2, delta=self.delta_loss,y = label, regularizer = self.loss_regularizer)
        acc_dict = accuracy_measure(result1, result2, label, self.containment_type, self.delta_inf)   
        self.log(f"{stage}_loss", loss, batch_size=label.size(0))
        self.log(f"{stage}_acc", acc_dict['accuracy'], batch_size=label.size(0))
        self.log(f"{stage}_false_pos_rate", acc_dict['false_pos_rate'], batch_size=label.size(0))
        self.log(f"{stage}_false_neg_rate", acc_dict['false_neg_rate'], batch_size=label.size(0))
        return loss

    def training_step(self, batch: Data, _):
        """Computes training loss and accuracy."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Data, _):
        """Computes validation loss and accuracy."""
        with torch.no_grad():
            return self._shared_step(batch, "val")

    def test_step(self, batch: Data, _):
        """Computes test loss and accuracy."""
        with torch.no_grad():
            loss = self._shared_step(batch, "test")
            return loss

    def configure_optimizers(self):
        """
        Sets up the optimizer and learning rate scheduler.
        
        Returns:
            Tuple[List[Optimizer], Dict]: List containing the optimizer and a dictionary with the 
                                          learning rate scheduler configuration.
        """
        optimizer_cls = getattr(torch.optim, self.optim_type, torch.optim.Adam)
        optimizer = optimizer_cls(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        lr_scheduler = ReduceLROnPlateau(optimizer, factor=self.lr_factor, mode='min', patience = self.patience)
        return [optimizer], {"scheduler": lr_scheduler, "interval": "epoch", "monitor": "train_loss"}
 