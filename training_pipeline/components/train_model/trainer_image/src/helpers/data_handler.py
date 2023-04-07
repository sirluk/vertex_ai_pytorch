import os
import csv
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

from skmultilearn.model_selection import iterative_train_test_split


class TextDS(Dataset):
    def __init__(self, text, tokenizer, max_len):
        self.text = text
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int):
        tokenized_sample = self.tokenizer(self.text[idx], padding='max_length', max_length=self.max_len, truncation=True, return_tensors="pt")
        token_ids_sample = tokenized_sample['input_ids']
        token_type_ids_sample = tokenized_sample['token_type_ids']
        attention_masks_sample = tokenized_sample['attention_mask']
        return token_ids_sample, token_type_ids_sample, attention_masks_sample


def multiprocess_tokenization(text_list, tokenizer, max_len, num_workers=16):
    ds = TextDS(text_list, tokenizer, max_len)
    _loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=num_workers, drop_last=False)
    token_ids = []
    token_type_ids = []
    attention_masks = []
    for tokenized_batch, token_type_ids_batch, attention_masks_batch in _loader:
        token_ids.append(tokenized_batch)
        token_type_ids.append(token_type_ids_batch)
        attention_masks.append(attention_masks_batch)

    token_ids = torch.cat(token_ids, dim=0).squeeze(1)
    token_type_ids = torch.cat(token_type_ids, dim=0).squeeze(1)
    attention_masks = torch.cat(attention_masks, dim=0).squeeze(1)

    return token_ids, token_type_ids, attention_masks


def get_data_loader(ds, shuffle, batch_size):
    
    def batch_fn(batch):
        input_ids, token_type_ids, attention_masks, labels = [torch.stack(l) for l in zip(*batch)]
        x = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_masks
        }
        return x, labels
    
    return DataLoader(ds, shuffle=shuffle, batch_size=batch_size, drop_last=False, collate_fn=batch_fn)


def get_data(
    data_path,
    tokenizer,
    batch_size = 16,
    val_size = .1,
    max_length = 200,
    n_occ_cutoff = 0,
    debug = False
):   
    text_key = "sentence"
    label_key = "label"

    data_dict = []

    with open(data_path) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
                try:
                    data_dict.append(dict(zip(keys, row)))
                except NameError:
                      keys = dict(zip(row, [[] for _ in range(len(row))]))

    # if debug only use subset of data
    if debug:
        cutoff = min(int(batch_size*10), len(data_dict))
        data_dict = data_dict[:cutoff]

    # reshape list of dicts to dict of lists with relevant keys
    keys = [text_key, label_key]
    x = [[d[k] for k in keys] for d in data_dict]
    data = dict(zip(keys, zip(*x)))
    
    # generate label map
    counts = Counter([x for l in data[label_key] for x in l])
    label_map = dict(enumerate(
        set([k for k,v in counts.items() if v>n_occ_cutoff])
    ))

    # labels to dummies
    label_ids = torch.tensor([[1 if v in row else 0 for v in label_map.values()] for row in data[label_key]], dtype=torch.long)
    
    # train test split
    idx = torch.randperm(len(label_ids))
    train_idx, y_train, val_idx, y_val = iterative_train_test_split(idx.unsqueeze(1), label_ids[idx], test_size = 0.1)
    
    # tokenize texts
    input_ids, token_type_ids, attention_masks = multiprocess_tokenization(list(data[text_key]), tokenizer, max_length)

    # create datasets and data loaders
    ds_train = TensorDataset(
        input_ids[train_idx].squeeze(1),
        token_type_ids[train_idx].squeeze(1),
        attention_masks[train_idx].squeeze(1),
        y_train
    )
    ds_val = TensorDataset(
        input_ids[val_idx].squeeze(1),
        token_type_ids[val_idx].squeeze(1),
        attention_masks[val_idx].squeeze(1),
        y_val
    )

    dl_train = get_data_loader(ds_train, shuffle=True, batch_size=batch_size)
    dl_val = get_data_loader(ds_val, shuffle=False, batch_size=batch_size)
    
    return dl_train, dl_val, label_map