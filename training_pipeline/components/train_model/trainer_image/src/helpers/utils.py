import torch
from pathlib import Path
from werkzeug.utils import secure_filename
import tempfile
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from typing import Union, List


def f1score(predictions: torch.Tensor, labels: torch.Tensor, **kwargs): # -> Union[float, ArrayLike[float]]:
    pred_np = predictions.numpy()
    labels_np = labels.numpy()
    return f1_score(labels_np, pred_np, **kwargs)


def accuracy(predictions: torch.Tensor, labels: torch.Tensor, balanced: bool = False) -> float:
    pred_np = predictions.numpy()
    labels_np = labels.numpy()
    if balanced:
        return balanced_accuracy_score(labels_np, pred_np)
    return accuracy_score(labels_np, pred_np)


def dict_to_device(data: dict, device: str = "cpu"):
    return {k:v.to(device) for k,v in data.items()}


def dict_to_str(d: dict) -> str:
    return ", ".join([f"{k}: {v}" for k,v in d.items()])


def get_file_path(filename, secure=True):
    if secure:
        filename = secure_filename(filename)
    return os.path.join(tempfile.gettempdir(), filename)


def get_path_from_uri(uri, path_type="local"):
    basepath = "/".join(uri.split("/")[2:])
    if path_type=="fuse":
        return Path("/gcs/" + basepath)
    elif path_type=="local":
        return Path(get_file_path(basepath, secure=False))

    
def get_device() -> List[torch.device]:
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        device = [torch.device(f"cuda:{int(i)}") for i in range(n_gpu)]
    else:
        device = [torch.device("cpu")]
    print(f"Device: {device}")
    return device