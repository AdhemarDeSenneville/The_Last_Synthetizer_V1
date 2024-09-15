
from .VAE_training import LitAutoEncoder
from ..data.data import AudioDataset
from pytorch_lightning import Trainer
from torch.utils.data import random_split, DataLoader, TensorDataset, Dataset
from pytorch_lightning.callbacks import *

class TrainerWarper():

    def __init__(
            self,
            model_cfg,
            data_cfg,
            optimizer_cfg,
            loss_cfg,
            training_cfg,
        ) -> None:

        # Model init
        self.model_warper = LitAutoEncoder(model_cfg, data_cfg, optimizer_cfg, loss_cfg)

        # Data init
        batch_size = data_cfg.pop('batch_size')
        num_workers = data_cfg.pop('num_workers')

        self.dataset = AudioDataset(**data_cfg)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
        )

        # Trainer init
        training_cfg['callbacks'] = instantiate_callbacks(training_cfg['callbacks'])
        self.trainer = Trainer(**training_cfg)
    
    def train(self):

        self.trainer.fit(self.model_warper, self.dataloader)


def instantiate_callbacks(callbacks_config):
    callback_objects = []
    
    for callback_name, params in callbacks_config:
        callback_class = globals()[callback_name]
        callback_objects.append(callback_class(**params))
    
    return callback_objects