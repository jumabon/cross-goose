import datetime
import os
import random

import imageio
import matplotlib
import numpy as np
import tifffile
import torch
from matplotlib.colors import ListedColormap


def remap(arr: np.array, min_out: float, max_out: float, axis=None):
    vmin = np.min(arr, axis=axis, keepdims=True)
    vmax = np.max(arr, axis=axis, keepdims=True)

    return ((arr - vmin) / (vmax - vmin)) * (max_out-min_out) + min_out


def remap_torch(arr: torch.Tensor, min_out: float, max_out: float, dim=None):
    vmin = torch.min(arr, dim=dim, keepdims=True)
    vmax = torch.max(arr, dim=dim, keepdims=True)

    return ((arr - vmin) / (vmax - vmin)) * (max_out-min_out) + min_out


def norm_q(arr: np.ndarray, q: float = 0.05) -> np.ndarray:
    q1, q99 = np.quantile(arr, [q, 1-q])
    return np.clip((arr - q1) / (q99 - q1), 0.0, 1.0)


def default(a, b):
    if a is None:
        return b
    return a


def get_timestamp():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')


def default_file_source(folders, filename, raise_error: bool = True):
    """tries to load filename from folders[0] then folders[1] etc, until none
    """
    for f in folders:
        path = os.path.join(f, filename)
        if os.path.exists(path):
            return path
    if raise_error:
        raise FileNotFoundError(
            f"could not find {filename} in any of {folders}")
    else:
        return None


def imread(path: str):
    filename = os.path.split(path)[-1]
    ext = filename.split('.')[-1]
    if ext == 'png':
        return imageio.imread(path)
    elif ext == 'tif':
        return tifffile.imread(path)
    else:
        raise ValueError(ext)


def write_flows_stack(flows, path):
    tifffile.imwrite(
        path,
        data=flows.astype(np.float32),
        ome=True,
        photometric='minisblack',
        # compression='zlib',
        metadata={'axes': 'ZCYX', }
    )


def append_to_dict_items(log_dict: dict, update: dict):
    for k, v in update.items():
        if k in log_dict:
            log_dict[k].append(v)
        else:
            log_dict[k] = [v]


def get_labels_cmap(base_cmap: str = 'RdYlGn'):
    base_cmap = matplotlib.colormaps[base_cmap].resampled(512)
    newcolors = base_cmap(np.linspace(0, 1, 512))
    newcolors[0, :] = [0, 0, 0, 1]
    return ListedColormap(newcolors)


def random_adjective():
    return random.choice(list(open('crossgoose/misc/english-adjectives.txt', encoding='utf-8')))[:-1]
