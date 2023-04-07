import shutil
import argparse
import json
import torch
from torch import nn
from torch import optim
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from functools import partial

from helpers.data_handler import get_data
from helpers.model_train_wrapper import BertClfTrainer
from helpers.train_logger import TrainLogger
from helpers.utils import get_path_from_uri, f1score, accuracy, get_device

torch.manual_seed(0)

def train_model(hparams: argparse.Namespace):
    
    # Get device
    device = get_device()
    
    base_dir_fuse = get_path_from_uri(hparams.base_dir, "fuse")
    model_dir_fuse = get_path_from_uri(hparams.model_dir, "fuse")
    checkpoint_dir_fuse = get_path_from_uri(hparams.checkpoint_dir, "fuse")
    tensorboard_dir_fuse = get_path_from_uri(hparams.tensorboard_dir, "fuse")
    train_data_dir_fuse = get_path_from_uri(hparams.training_data_uri, "fuse")
    
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(hparams.model_name)
    
    # get data
    dl_train, dl_val, label_map = get_data(
        train_data_dir_fuse,
        tokenizer,
        hparams.batch_size,
        debug=hparams.debug
    )

    # get model
    model = BertClfTrainer(
        hparams.model_name,
        len(label_map),
        hparams.dropout,
        hparams.n_hidden
    )
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    metrics = {
        "f1_micro": partial(f1score, average="micro"), 
        "acc": accuracy
    }
    
    optimizer = optim.AdamW(model.parameters(), lr=hparams.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=(hparams.num_epochs * len(dl_train))
    )

    logger_id = str(base_dir_fuse).split("gcs")[1].replace("/", "_").strip("_")
    logger = TrainLogger(tensorboard_dir_fuse, f"logger_{logger_id}")
    
    checkpoint_dir_fuse.mkdir(parents=True, exist_ok=True)
    filepath_checkpoint = checkpoint_dir_fuse / hparams.model_name_output
    filepath_model = model_dir_fuse / hparams.model_name_output

    model.fit(
        train_loader = dl_train,
        val_loader = dl_val,
        logger = logger,
        optimizer = optimizer,
        scheduler = scheduler,
        loss_fn = loss_fn,
        metrics = metrics,
        num_epochs = hparams.num_epochs,
        filepath = filepath_checkpoint
    )
    
    model_dir_fuse.mkdir(parents=True, exist_ok=True)    
    shutil.copyfile(filepath_checkpoint, filepath_model)
            
    with open(model_dir_fuse / hparams.output_file_label_map, "w") as f:
        json.dump(label_map, f)

    return "DONE", 200
