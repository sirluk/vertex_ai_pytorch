import os
# Current image pytorch-gpu.1-12 is < python 3.8 and does not support cached_property
# from functools import cached_property

import torch
from torch import nn
from transformers import BertModel

from typing import Union

from helpers.model_heads import ClfHead


class BertClf(nn.Module):
    
    @property
    def encoder_module(self) -> nn.Module:
        if isinstance(self.encoder, nn.DataParallel):
            return self.encoder.module
        else:
            return self.encoder
    
    @property # @cached_property
    def model_name(self) -> str:
        return self.encoder_module.config._name_or_path
    
    @property # @cached_property
    def hidden_dim(self) -> int:
        if self.is_large_model:
            return self.encoder_module.embeddings.word_embeddings.embedding_dim * 4
        else:
            return self.encoder_module.embeddings.word_embeddings.embedding_dim
    
    @property # @cached_property
    def total_layers(self) -> int:
        possible_keys = ["num_hidden_layers", "n_layer"]
        cfg = self.encoder_module.config
        for k in possible_keys:
            if k in cfg.__dict__:
                return getattr(cfg, k) + 1 # +1 for embedding layer
        raise Exception("number of layers of pre trained model could not be determined")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        dropout: float = .3,
        n_hidden: int = 0,
        **kwargs
    ):
        super().__init__()
        
        self.encoder = BertModel.from_pretrained(model_name, **kwargs)
        self.num_classes = num_classes
        self.dropout = dropout
        self.n_hidden = n_hidden
        self.is_large_model = (self.total_layers>10)
        
        self.forward_encoder_fn = self.forward_encoder_large if self.is_large_model else self.forward_encoder_standard
        
        self.classifier = ClfHead([self.hidden_dim]*(n_hidden+1), num_classes, dropout=dropout)
        
    def forward_encoder_large(self, **x):
        bert_output = self.encoder(**x, output_hidden_states=True)
        return torch.cat([h[:,0] for h in bert_output.hidden_states[-4:]], dim=1) # concatenate hidden states of last 4 layers           
        
    def forward_encoder_standard(self, **x):
        return self.encoder(**x)[0][:,0]        

    def forward(self, **x):
        emb = self.forward_encoder_fn(**x)
        return self.classifier(emb)
        
    def save_checkpoint(
        self,
        filepath: Union[str, os.PathLike]
    ) -> None:
        info_dict = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "n_hidden": self.n_hidden,
            "encoder_state_dict": self.encoder_module.state_dict(),
            "classifier_state_dict": self.classifier.state_dict()
        }

        torch.save(info_dict, filepath)
        return filepath

    @classmethod
    def load_checkpoint(cls, filepath: Union[str, os.PathLike], map_location: Union[str, torch.device] = torch.device('cpu')) -> nn.Module:
        info_dict = torch.load(filepath, map_location=map_location)

        cls_instance = cls(
            info_dict['model_name'],
            info_dict['num_classes'],
            info_dict['dropout'],
            info_dict['n_hidden']
        )
        cls_instance.encoder.load_state_dict(info_dict['encoder_state_dict'])
        cls_instance.classifier.load_state_dict(info_dict['classifier_state_dict'])

        cls_instance.eval()

        return cls_instance
    
    def to(self, device: Union[list, Union[str, torch.device]], *args, **kwargs) -> None:
        self._remove_parallel()
        if isinstance(device, list):
            super().to(device[0])
            if len(device)>1:
                asssert_fn = lambda x: x=="cuda" if isinstance(x, str) else x.type=="cuda"
                assert all([asssert_fn(d) for d in device]), "if list of devices is given, all must be of type 'cuda'"
                self.encoder = nn.DataParallel(self.encoder, device_ids=device)
        else:
            super().to(device)

    def cpu(self):
        self._remove_parallel()
        super().cpu()

    def cuda(self, *args, **kwargs) -> None:
        self._remove_parallel()
        super().cuda(*args, **kwargs)

    def _remove_parallel(self) -> None:
        if isinstance(self.encoder, nn.DataParallel):
            self.encoder = self.encoder.module