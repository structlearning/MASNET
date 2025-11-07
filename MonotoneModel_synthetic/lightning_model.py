import torch
from torch import Tensor
from easydict import EasyDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning
import torch
from utils import get_model
from general_utils import set_loss, predict

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
        self.optim_type = args.optim_type
        self.weight_decay = args.wd
        self.task_type = args.task_type
        self.model = get_model(args=args)
        self.loss_fun = torch.nn.MSELoss() if self.task_type == 'FacilityLocation' else set_loss

    def forward(self, X) -> Tensor:
        """Forward pass through the model."""
        return self.model(X)

    def compute_node_embedding(self, X) -> Tensor:
        """Compute node embeddings for the input graph data."""
        return self.model.compute_node_embedding(X)

    def _shared_step(self, batch, stage: str):
        """
        Generic step used for training, validation, and testing.
        
        Args:
            batch (Data): A batch of graph data.
            stage (str): One of "train", "val", or "test".
        
        Returns:
            Tensor: Computed loss.
        """
        
        set_1, set_2, label = batch
        pred, pred2 = self.model(set_1, set_2)
        if self.task_type == 'FacilityLocation':
              loss = torch.nn.MSELoss()(pred, label)
              acc = torch.abs(pred - label).mean()    
        else:
              loss = self.loss_fun(pred, pred2, delta=0.1, y = label)
              pred = predict(pred, pred2)
              acc = (pred == label).float().mean()
        self.log(f"{stage}_loss", loss, batch_size=label.size(0))
        self.log(f"{stage}_acc", acc, batch_size=label.size(0))
        return loss

    def training_step(self, batch, _):
        """Computes training loss and accuracy."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        """Computes validation loss and accuracy."""
        with torch.no_grad():
            return self._shared_step(batch, "val")

    def test_step(self, batch, _):
        """Computes test loss and accuracy."""
        with torch.no_grad():
            return self._shared_step(batch, "test")

    def configure_optimizers(self):
        """
        Sets up the optimizer and learning rate scheduler.
        
        Returns:
            Tuple[List[Optimizer], Dict]: List containing the optimizer and a dictionary with the 
            learning rate scheduler configuration.
        """
        optimizer_cls = getattr(torch.optim, self.optim_type, torch.optim.Adam)
        optimizer = optimizer_cls(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        mode = 'min' if self.task_type == 'FacilityLocation' else 'max'
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=self.lr_factor, mode=mode)
        return [optimizer], {"scheduler": lr_scheduler, "interval": "epoch", "monitor": "train_acc"}
