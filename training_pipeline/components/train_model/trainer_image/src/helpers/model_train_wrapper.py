import math
import os
from tqdm import trange, tqdm

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from typing import Dict, Callable, Union

from helpers.utils import dict_to_str, dict_to_device
from helpers.train_logger import TrainLogger
from helpers.model import BertClf


class BertClfTrainer(BertClf):
     

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,    
        logger: TrainLogger,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
        num_epochs: int,
        filepath: Union[str, os.PathLike],
        max_grad_norm: float = 1.,
        cooldown: int = 5
    ):
        
        self.global_step = 0
                
        cooldown_counter = 0
        train_str = "Epoch {}{}"
        str_suffix = lambda x: "" if len(x)==0 else ", " + dict_to_str(x)
        train_iterator = trange(num_epochs, desc=train_str.format(0, str_suffix({})), leave=True, position=0)
        for epoch in train_iterator:

            self._step(
                train_loader,
                loss_fn,
                logger,               
                optimizer,
                scheduler,
                max_grad_norm
            )
            result = self.evaluate(
                val_loader,
                loss_fn,
                metrics,
            )
            
            train_iterator.set_description(
                train_str.format(epoch, str_suffix(result)), refresh=True
            )

            logger.validation_loss(epoch, result)

            if logger.is_best(result):
                self.save_checkpoint(filepath)
                cooldown_counter = 0
            else:
                cooldown_counter += 1
                
            if cooldown_counter > 5:
                break

                
    def _step(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        max_grad_norm: float
    ) -> float:
        
        self.train()
        
        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)
        for step, (inputs, labels) in enumerate(epoch_iterator):        
        
            inputs = dict_to_device(inputs, self.device)
            outputs = self(**inputs)
            loss = loss_fn(outputs, labels.float().to(self.device))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            self.zero_grad()
            
            logger.step_loss(self.global_step, loss, scheduler.get_last_lr()[0]) 
            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)
            self.global_step += 1
       

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
    ) -> dict:
        
        self.eval()
        
        output_list = []
        val_iterator = tqdm(val_loader, desc="evaluating", leave=False, position=1)
        for i, (inputs, labels) in enumerate(val_iterator):
            
            inputs = dict_to_device(inputs, self.device)
            logits = self(**inputs)
            output_list.append((
                logits.cpu(),
                labels
            ))
            
        p, l = list(zip(*output_list))
        logits = torch.cat(p, dim=0)
        labels = torch.cat(l, dim=0)   
        
        eval_loss = loss_fn(logits, labels.float()).item()    
        
        preds = (logits > 0).long()
        result = {metric_name: metric(preds, labels) for metric_name, metric in metrics.items()}
        result["loss"] = eval_loss

        return result