import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

import lightning
import numpy as np
import tifffile
import torch
from natsort import natsorted
from torch import nn
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision.transforms import InterpolationMode, v2
from tqdm import tqdm

from crossgoose.cellpose.transforms import get_pad_yx, normalize99
from crossgoose.dynamics import extented_diffusion
from crossgoose.mask_utils import convert_labels_to_onehot
from crossgoose.utils import default, imread, remap, write_flows_stack

ImageNormalization = Literal['M1P1', 'N99']


def normalize_image(image: np.ndarray, image_normalization: ImageNormalization) -> np.ndarray:
    if image_normalization == 'M1P1':
        return remap(image, -1, 1)
    elif image_normalization == 'N99':
        return normalize99(image)
    else:
        raise ValueError


@dataclass
class AugmentationParams():
    patch_size: int | None
    flip_h: bool
    flip_v: bool
    rotate: bool
    rot90: bool
    scale_range: Tuple[float, float] | None
    deterministic_patch: bool = False


def random_transform(
    image: torch.Tensor,
    labels: torch.Tensor,
    overlap_mask: torch.Tensor,
    flows: torch.Tensor,
    patch_size: int | None,
    flip_h: bool,
    flip_v: bool,
    rotate: bool,
    rot90: bool,
    scale_range: Tuple[float, float] | None = None,
    v_fill_img: float = -1,
    deterministic_patch: bool = False,
    relabel: bool=True,
):
    # expects no bach dim
    transforms = dict()
    assert len(image.shape) == 2
    assert len(flows.shape) == 4, flows.shape

    if scale_range is not None or rotate:
        if scale_range is not None:
            scale = torch.rand(size=(1,)) * \
                (scale_range[1]-scale_range[0]) + scale_range[0]
        else:
            scale = 1.0
        if rotate:
            angle = float(torch.rand(size=(1,)) * 360 - 180)
        else:
            angle = 0.0
        affine_transform = {
            'scale': scale,
            'angle': angle,
            'translate': (0, 0),
            'shear': (0, 0)
        }
        image = v2.functional.affine(
            image.unsqueeze(dim=0),
            **affine_transform,
            interpolation=InterpolationMode.BILINEAR,
            fill=v_fill_img
        )[0]
        labels = v2.functional.affine(
            labels.unsqueeze(dim=0),
            **affine_transform,
            interpolation=InterpolationMode.NEAREST,
            fill=0
        )[0].long()
        if overlap_mask is not None:
            overlap_mask = v2.functional.affine(
                overlap_mask.unsqueeze(dim=0),
                **affine_transform,
                interpolation=InterpolationMode.NEAREST,
                fill=0
            )[0].long()
        flows = v2.functional.affine(
            flows,
            **affine_transform,
            interpolation=InterpolationMode.BILINEAR,
            fill=0
        )

        if angle != 0.0:
            theta = -affine_transform['angle']/360*2*np.pi
            rot_matrix = torch.tensor([[np.cos(theta), -np.sin(theta)],
                                       [np.sin(theta), np.cos(theta)]]).float()

            n, _, h, w = flows.shape
            flows_flat = flows.permute(1, 0, 2, 3).reshape(2, -1)
            flows_flat_rot = torch.matmul(rot_matrix, flows_flat)
            flows = flows_flat_rot.reshape(2, n, h, w).permute(1, 0, 2, 3)

        # renorm flows
        min_float = torch.nextafter(torch.zeros((1,), dtype=flows.dtype),
                                    torch.ones((1,)))
        flows = flows / (min_float + (flows**2).sum(dim=1, keepdims=True)**0.5)

        transforms['affine'] = affine_transform

    if patch_size is None:
        h, w = image.shape
        ypad1, ypad2, xpad1, xpad2 = get_pad_yx(
            h, w, div=16, extra=1, min_size=None)
        patch_size = max(h+xpad1+xpad2, w+ypad1+ypad2)

    if any(s < patch_size for s in image.shape):
        # print(f"shape {tuple(image.shape)} too small for patch size {patch_size}")
        d0 = max(0, patch_size - image.shape[0])
        d1 = max(0, patch_size - image.shape[1])
        padding = (
            d1 // 2, d1 - d1//2, d0 // 2, d0 - d0//2
        )
        image = nn.functional.pad(image, padding, value=v_fill_img)
        labels = nn.functional.pad(labels, padding, value=0)
        if overlap_mask is not None:
            overlap_mask = nn.functional.pad(overlap_mask, padding, value=0)
        flows = nn.ReplicationPad2d(padding)(flows)

        assert len(image.shape) == 2
        assert len(flows.shape) == 4, flows.shape

        assert all(
            s >= patch_size for s in image.shape), "oops messed up the padding"

    if deterministic_patch:
        i = image.shape[0]//2 - patch_size//2
        j = image.shape[1]//2 - patch_size//2
        h = patch_size
        w = patch_size
    else:
        i, j, h, w = v2.RandomCrop.get_params(
            image,
            output_size=(patch_size, patch_size)
        )

    transforms['crop'] = (i, j, h, w)

    image = v2.functional.crop(image, i, j, h, w)
    labels = v2.functional.crop(labels, i, j, h, w)
    if overlap_mask is not None:
        overlap_mask = v2.functional.crop(overlap_mask, i, j, h, w)
    flows = v2.functional.crop(flows, i, j, h, w)

    if flip_h and (torch.rand(1) > 0.5):
        image = v2.functional.hflip(image)
        labels = v2.functional.hflip(labels)
        if overlap_mask is not None:
            overlap_mask = v2.functional.hflip(overlap_mask)
        flows = v2.functional.hflip(flows)
        flows[:, 1] = -flows[:, 1]
        transforms['flip_h'] = True
    else:
        transforms['flip_h'] = False

    if flip_v and (torch.rand(1) > 0.5):
        image = v2.functional.vflip(image)
        labels = v2.functional.vflip(labels)
        if overlap_mask is not None:
            overlap_mask = v2.functional.vflip(overlap_mask)
        flows = v2.functional.vflip(flows)
        flows[:, 0] = -flows[:, 0]
        transforms['flip_v'] = True
    else:
        transforms['flip_v'] = False

    if rot90:
        k = int(torch.randint(0, 4, size=(1,)))
        image = torch.rot90(image, k=k, dims=(0, 1))
        labels = torch.rot90(labels, k=k, dims=(0, 1))
        if overlap_mask is not None:
            overlap_mask = torch.rot90(overlap_mask, k=k, dims=(0, 1))
        flows = torch.rot90(flows, k=k, dims=(-2, -1))
        theta = k*np.pi/2
        rot_matrix = torch.tensor([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]]).float()

        n, _, h, w = flows.shape
        flows_flat = flows.permute(1, 0, 2, 3).reshape(2, -1)
        flows_flat_rot = torch.matmul(rot_matrix, flows_flat)
        flows = flows_flat_rot.reshape(2, n, h, w).permute(1, 0, 2, 3)
        assert flows.shape == (n, 2, h, w)
        transforms['rot'] = k

    if relabel:
        labels_new_to_old = {}
        current_labels = torch.unique(labels, sorted=True).cpu().numpy()
        current_labels = natsorted(list(set(current_labels) - set([0])))

        new_flows = torch.zeros(
            (len(current_labels), 2) + labels.shape, dtype=flows.dtype)
        new_labels = torch.zeros_like(labels)

        for i, l in enumerate(current_labels):
            new_l = i+1
            try:
                new_labels[labels == l] = new_l
                flows_l = flows[l-1]
                # flows[l-1] corresponds to flows for label l
                new_flows[new_l-1] = flows_l
                labels_new_to_old[new_l] = l
            except Exception as e:
                print(
                    f"failed to process old label {l} into label {new_l}\n{new_labels.shape=}\n{new_flows.shape=}\n{flows.shape=}\n{current_labels=}")
                raise e

        labels = new_labels
        flows = new_flows

        transforms['labels_new_to_old'] = labels_new_to_old
    else:
        transforms['labels_new_to_old'] = None

    return image, labels, overlap_mask, flows, transforms


class FlowDataset(Dataset):

    def __init__(
            self,
            data_dir,
            subset,
            augmentation_params: AugmentationParams,
            recompute_flows: bool,
            center_method: str,
            alpha_heat: float = 0.95,
            cuda_flow_compute: bool = True,
            closure_radius: int | None = None,
            bootstrap_factor: int = 1,
            lazy_flow_computing: bool = False,
            return_overlap_map: bool = False,
            keep_data_in_memory: bool = False,
            image_normalization: ImageNormalization = 'M1P1',
    ):
        self.alpha_heat = alpha_heat
        self.aug_params = augmentation_params
        self.closure_radius = closure_radius
        self.bootstrap_factor = bootstrap_factor
        self.lazy_flow_computing = lazy_flow_computing
        self.center_method = center_method
        self.cuda_flow_compute = cuda_flow_compute
        self.return_overlap_map = return_overlap_map
        self.keep_data_in_memory = keep_data_in_memory
        self.image_normalization = image_normalization

        self.name = f"{os.path.split(data_dir)[-1]}-{subset}"

        directory = os.path.join(data_dir, subset)
        directory = Path(directory)

        self.directory = directory

        self._fetch_files()    

        assert len(self.images_files) > 0
        assert len(self.images_files) == len(self.masks_files)
        if not len(self.images_files) == len(self.masks_onehot_files):
            print(f"[{self.name}] missing onehot masks: generating ...")
            self.generate_one_hot_masks()
            self._fetch_files()  
            assert len(self.images_files) == len(self.masks_onehot_files), (self.images_files,self.masks_onehot_files)

        self.flow_files = []
        desc = f"[{self.name}] " + ("recomputing" if recompute_flows else "checking/computing") + \
            f" flows for {subset} data"
        for index, filename in enumerate(tqdm(self.images_files, desc=desc)):
            m = re.match(r'(.*)\_img\.(tif|png)', filename)
            assert m is not None, f"image file {filename} does not match pattern .*_img.tif"
            image_name = m.group(1)
            flow_file = f"{image_name}_flows_alpha{alpha_heat}_{center_method}_c{default(closure_radius,0)}.tif"
            flow_file_path = os.path.join(self.directory, flow_file)
            compute_flow = ((not os.path.exists(flow_file_path))
                            and (not self.lazy_flow_computing))
            compute_flow = compute_flow or recompute_flows
            if compute_flow:
                self.compute_flow(flow_file_path, index,
                                  center_method, cuda_flow_compute)
            self.flow_files.append(flow_file)

        self.n_images = len(self.images_files)

        if self.keep_data_in_memory:
            self.buffer = {}
        else:
            self.buffer = None

    def _fetch_files(self):
        
        dir_file = os.listdir(self.directory)

        self.images_files = natsorted(
            [f for f in dir_file if re.search(r'.*\_img\.(tif|png)', f)])
        self.masks_files = natsorted(
            [f for f in dir_file if re.search(r'.*\_masks\.(tif|png)', f)])
        self.masks_onehot_files = natsorted(
            [f for f in dir_file if re.search(r'.*\_masks_onehot\.(tif|png)', f)])

    def generate_one_hot_masks(self):
        for masks_file in tqdm(self.masks_files, desc=f'computing onehot masks for {self.name}'):
            name, _ = masks_file.split('.')
            file = os.path.join(self.directory,f"{name}_onehot.tif")
            if not os.path.exists(file):
                labels = imread(os.path.join(
                    self.directory, masks_file))
                mask_onehot = convert_labels_to_onehot(
                    labels, closure_radius=self.closure_radius)
                
                tifffile.imwrite(
                    file,
                    data=mask_onehot,
                    compression='zlib',
                    metadata={'axes': 'ZYX'},
                )

    def compute_flow(
        self,
        flow_file_path: str,
        image_index: int,
        center_method: str,
        cuda_flow_compute: bool
    ):
        device = None if cuda_flow_compute else torch.device('cpu')

        labels_one_hot = imread(os.path.join(
            self.directory, self.masks_onehot_files[image_index]))
        flows, _, _ = extented_diffusion(
            masks_onehot=labels_one_hot,
            alpha_out=self.alpha_heat,
            center_method=center_method,
            device=device
        )
        write_flows_stack(flows, flow_file_path)
        del flows, labels_one_hot

    def __len__(self):
        return self.n_images * self.bootstrap_factor

    def _get_raw_item(self, index):
        if self.keep_data_in_memory and index in self.buffer:
            data = self.buffer[index]
        else:
            data = {}
            image = imread(os.path.join(
                self.directory, self.images_files[index]))
            if len(image.shape) == 3:
                chan_dim = np.argmin(image.shape)
                image = np.mean(image, axis=chan_dim)
            image = self.norm_image(image)
            data['image'] = torch.tensor(image)

            labels = imread(os.path.join(
                self.directory, self.masks_files[index]))
            data['labels'] = torch.tensor(labels, dtype=torch.long)

            if self.keep_data_in_memory:
                self.buffer[index] = data

            flow_file = self.flow_files[index]
            flow_file_path = os.path.join(self.directory, flow_file)
            if flow_file_path is None and self.lazy_flow_computing:
                flow_file_path = os.path.join(self.directory, flow_file)
                print(
                    f"[{self.name}] i'm lazy (or the file is missing), i'm just now computing the flow for image {index}, hold on a sec...")
                self.compute_flow(flow_file_path, index,
                                  self.center_method,
                                  self.cuda_flow_compute)
            else:
                assert os.path.exists(flow_file_path)

            flows = imread(flow_file_path)
            assert len(flows.shape) == 4
            data['flows'] = torch.tensor(flows, dtype=torch.float)

            #sanity check
            assert np.max(labels) == flows.shape[0], f"got max label {np.max(labels)} and {flows.shape[0]} flow slices"

            if self.return_overlap_map:
                labels_oh = imread(os.path.join(self.directory,
                                                self.masks_onehot_files[index]))
                overlap_mask = np.sum(labels_oh, axis=0) > 1
                overlap_mask = torch.from_numpy(overlap_mask)
                assert overlap_mask.shape == labels.shape
            else:
                overlap_mask = None
            data['overlap_mask'] = overlap_mask
        return data

    def norm_image(self, image: np.ndarray) -> np.ndarray:
        return normalize_image(image, self.image_normalization)

    def __getitem__(self, index):
        index = index % self.n_images

        data = self._get_raw_item(index)

        if self.image_normalization == 'M1P1':
            v_fill_img = -1
        elif self.image_normalization == 'N99':
            v_fill_img = 0.0
        else:
            v_fill_img = 0.0

        image, labels, overlap_mask, flows, transforms = random_transform(
            data['image'], data['labels'],
            data['overlap_mask'], data['flows'],
            v_fill_img=v_fill_img,
            **self.aug_params.__dict__
        )

        nb_instances = int(flows.shape[0])
        ret = {
            'image': image.unsqueeze(dim=0).float(),
            'labels': labels,
            'flows': flows,
            'nb_instances': nb_instances,
            'source': self.images_files[index],
            'transforms': transforms
        }
        if self.return_overlap_map:
            ret['overlap_mask'] = overlap_mask
        return ret


def collate_tensor_pad_to_size(batch: List[torch.Tensor], value=0) -> torch.Tensor:
    """collates tensors by padding the first dim to be the same

    Args:
        batch (List[torch.Tensor]): tensors to collate
        value (_type_, optional): padding value. Defaults to 0.

    Returns:
        torch.Tensor: stacked tensors with padding
    """
    elem = batch[0]
    assert isinstance(elem, torch.Tensor)
    # expects tensors of shape n,c,h,w
    n_max = max([e.shape[0] for e in batch])
    base_pad = (0, 0)*3
    return torch.stack([
        nn.functional.pad(
            input=e,
            pad=base_pad + (0, n_max - e.shape[0]),
            value=value
        ) for e in batch
    ], dim=0)


def flow_data_collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        clone = copy.copy(elem)
        for key in elem:
            if key == 'flows':
                clone[key] = collate_tensor_pad_to_size(
                    [d[key] for d in batch])
            elif key == 'transforms':
                clone[key] = [d[key] for d in batch]
            else:
                clone[key] = default_collate([d[key] for d in batch])
        return clone
    else:
        return default_collate(batch)


class MultiDataset(Dataset):
    def __init__(self, data_dir: List[str], **kwargs):
        self.datasets = [
            FlowDataset(data_dir=d, **kwargs) for d in data_dir
        ]
        self.sizes = [d.__len__() for d in self.datasets]
        self.sizes_cumsum = np.cumsum(self.sizes)
        self.offset = np.concatenate(
            (np.array([0]), self.sizes_cumsum), axis=0)

    def __len__(self):
        return sum(self.sizes)

    def __getitem__(self, index):
        d_idx = np.argmax(index < self.sizes_cumsum)
        return self.datasets[d_idx].__getitem__(index=index-self.offset[d_idx])


class FlowDataModule(lightning.LightningDataModule):
    def __init__(
            self,
            data_root:str,
            dataset: str | List[str],
            batch_size: int,
            num_workers: int,
            augmentation_params: AugmentationParams,
            validation: bool = False,
            alpha_heat=0.95,
            recompute_flows: bool = False,
            closure_radius: int | None = 8,
            cuda_flow_compute: bool = True,
            center_method: str = 'dist',
            bootstrap_factor: int = 1,
            prefetch_factor: int = 2,
            lazy_flow_computing: bool = False,
            return_overlap_map: bool = False,
            keep_data_in_memory: bool = True,
            val_batch_size: int | None = None,
            val_patch_size: int | None = None,
            image_normalization: ImageNormalization = 'M1P1'
    ):
        super().__init__()
        data_dir = os.path.join(data_root,dataset)

        self.data_dir = data_dir
        self.prefetch_factor = prefetch_factor
        self.train_data, self.test_data = None, None
        self.val_data = None
        self.batch_size = batch_size
        self.val_batch_size = default(val_batch_size, batch_size)
        self.num_workers = num_workers
        self.validation = validation

        self.dataset_params_train = dict(
            alpha_heat=alpha_heat,
            augmentation_params=augmentation_params,
            recompute_flows=recompute_flows,
            closure_radius=closure_radius,
            cuda_flow_compute=cuda_flow_compute,
            center_method=center_method,
            bootstrap_factor=bootstrap_factor,
            lazy_flow_computing=lazy_flow_computing,
            return_overlap_map=return_overlap_map,
            keep_data_in_memory=keep_data_in_memory,
            image_normalization=image_normalization
        )
        self.dataset_params_val = dict(
            alpha_heat=alpha_heat,
            augmentation_params=AugmentationParams(
                patch_size=default(
                    val_patch_size, augmentation_params.patch_size),
                flip_h=False,
                flip_v=False,
                rotate=False,
                scale_range=None,
                rot90=None,
                deterministic_patch=True
            ),
            bootstrap_factor=1,
            recompute_flows=recompute_flows,
            closure_radius=closure_radius,
            cuda_flow_compute=cuda_flow_compute,
            center_method=center_method,
            lazy_flow_computing=lazy_flow_computing,
            return_overlap_map=return_overlap_map,
            keep_data_in_memory=keep_data_in_memory,
            image_normalization=image_normalization,
        )

    def setup(self, stage):
        if isinstance(self.data_dir, list):
            data_class = MultiDataset
        else:
            data_class = FlowDataset

        self.train_data = data_class(
            subset="train",
            data_dir=self.data_dir,
            **self.dataset_params_train
        )
        self.test_data = data_class(
            subset="test",
            data_dir=self.data_dir,
            **self.dataset_params_val
        )

        if self.validation:
            self.val_data = data_class(
                subset="val",
                data_dir=self.data_dir,
                **self.dataset_params_val
            )

    def val_dataloader(self):
        if self.validation is not None:
            return DataLoader(
                self.val_data,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                collate_fn=flow_data_collate_fn,
                shuffle=False,
                prefetch_factor=self.prefetch_factor
            )
        else:
            return None

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=flow_data_collate_fn,
            shuffle=True,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=flow_data_collate_fn,
            shuffle=False,
            prefetch_factor=self.prefetch_factor
        )
