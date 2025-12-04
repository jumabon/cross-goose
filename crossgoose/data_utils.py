import glob
import os
import re
import shutil
from typing import Dict, List, Tuple

import imageio
import natsort
import numpy as np
import tifffile
from tqdm import tqdm

from crossgoose.data_synthetic import make_synth_data


def make_dataset(
    images: str,
    labels: str,
    save_path: str,
    seed: int = 0,
    val_frac: float = 0.15
):
    assert os.path.exists(images)
    assert os.path.exists(labels)

    all_images = {}
    pat_img = re.compile(r'.*\_([A-Z][0-9]{2})\_w2\_.*\.(?:tif|tiff)')
    for filename in os.listdir(images):
        match = pat_img.match(filename)
        if match is not None:
            img_name = match.group(1)
            assert img_name not in all_images
            all_images[img_name] = filename

    all_labels = {}
    pat_lbl = re.compile(r'([A-Z][0-9]{2})\_([0-9]{2})\_.*\.png')
    for filename in os.listdir(labels):
        match = pat_lbl.match(filename)
        if match is not None:
            img_name = match.group(1)
            label_id = int(match.group(2))
            if img_name in all_labels:
                all_labels[img_name][label_id] = filename
            else:
                all_labels[img_name] = {label_id: filename}

    print(f"found {len(all_images)} images")
    print(f"found {len(all_labels)} labels")
    assert len(all_images) > 0
    assert len(all_labels) > 0

    rng = np.random.default_rng(seed)
    nb_image = len(all_images)
    nb_train = int((nb_image//2)*(1-val_frac))
    nb_val = int((nb_image/2)*val_frac)
    all_id = rng.permutation(nb_image)
    train_ids = all_id[:nb_train]
    val_ids = all_id[nb_train:nb_train+nb_val]
    # test_ids = all_id[nb_train:]

    label_colormap = (rng.random((3, 256)) *
                      (3 * 2**16)).astype(np.uint16)
    label_colormap[:, 0] = 0

    train_dir = os.path.join(save_path, 'train')
    val_dir = os.path.join(save_path, 'val')
    test_dir = os.path.join(save_path, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    sorted_keys = natsort.natsorted(list(all_images.keys()))

    for i, img_name in enumerate(tqdm(sorted_keys)):
        if i in train_ids:
            save_dir = train_dir
        elif i in val_ids:
            save_dir = val_dir
        else:
            save_dir = test_dir

        img = tifffile.imread(os.path.join(images, all_images[img_name]))

        med = np.quantile(img, q=0.5)
        ii, jj = np.nonzero(img > med)
        crop = (slice(ii.min(), ii.max()), slice(jj.min(), jj.max()))

        img_crop = img[crop]

        h, w = img_crop.shape
        tifffile.imwrite(
            os.path.join(save_dir, img_name + '_img.tif'),
            data=img_crop,
            photometric='minisblack',
            imagej=True,
            metadata={'axes': 'YX', }
        )
        try:
            labels_ids = natsort.natsorted(all_labels[img_name].keys())
        except KeyError as e:
            print(f"missing {img_name} in {all_labels.keys()=}")
            raise e
        mask_proj = np.zeros((h, w), dtype=np.uint8)
        mask_onehot = np.zeros((len(labels_ids), h, w), dtype=np.uint8)
        for i, label_id in enumerate(labels_ids):
            bin_mask = imageio.imread(
                os.path.join(labels, all_labels[img_name][label_id]))
            bin_mask = bin_mask > 0
            bin_mask = bin_mask[crop]
            mask_proj[bin_mask] = i + 1
            mask_onehot[i][bin_mask] = 1

        tifffile.imwrite(
            os.path.join(save_dir, img_name + '_masks.tif'),
            data=mask_proj,
            photometric='minisblack',
            colormap=label_colormap,
            imagej=True,
            metadata={'axes': 'YX', }
        )
        tifffile.imwrite(
            os.path.join(save_dir, img_name + '_masks_onehot.tif'),
            data=mask_onehot,
            photometric='minisblack',
            imagej=True,
            metadata={'axes': 'ZYX', }
        )


# def pre_compute_flows(data_dir: str, recompute_flows: bool = False, center_method: str = 'graph_center_weighted', validation: bool = False, alpha_heat: float = 0.95):
#     data = FlowDataModule(
#         # data_dir="/home/jmabon/Documents/Data/BBBC010-C-Elegans/Dataset_v01",
#         data_dir=data_dir,
#         batch_size=6,
#         num_workers=1,
#         patch_size=256,
#         alpha_heat=alpha_heat,
#         flip_h=True,
#         flip_v=True,
#         rotate=False,
#         recompute_flows=recompute_flows,
#         cuda_flow_compute=True,
#         center_method=center_method,
#         validation=validation
#     )

#     data.setup("fit")
#     train_dataloader = data.train_dataloader()
#     batch = train_dataloader._get_iterator().__next__()
#     for k, v in batch.items():
#         if hasattr(v, 'shape') and len(v.shape) > 1:
#             print(f"{k}: {type(v)} {v.shape}")
#         else:
#             print(f"{k}: {type(v)} {v}")


def make_synth_dataset(
    src_dataset: str,
    dst_dataset: str,
    seed: int,
    shape: Tuple[int, int],
    n_images: Dict[str, int],
    padding: int,
    cell_count_range: List[int],
    rotate: bool,
    blur_range: List[float],
    alpha: float,
    noise_range: List[float],
    max_intersection: int,
    limit_overlaps: int,
    stacking: str = "alpha",
    bg_range:List=[0.5,1.0],
):

    for subset in ['train','test','val']:
        dst_subset_dir = os.path.join(dst_dataset, subset)
        src_subset_dir = os.path.join(src_dataset, subset)
        os.makedirs(dst_subset_dir, exist_ok=True)
        if subset in n_images:
            print(f"making {n_images[subset]} synth images for {subset} subset")
            make_synth_data(
                src_dir=src_subset_dir,
                seed=seed,
                shape=shape,
                n_images=n_images[subset],
                save_dir=dst_subset_dir,
                padding=padding,
                cell_count_range=cell_count_range,
                rotate=rotate,
                blur_range=blur_range,
                alpha=alpha,
                noise_range=noise_range,
                max_intersection=max_intersection,
                limit_overlaps=limit_overlaps,
                stacking=stacking,
                bg_range=bg_range
            )
        else:
            print(f"no synth image for {subset} subset, just copying")

        copy_files = glob.glob(f'{src_subset_dir}/*')
        for p in tqdm(copy_files, desc=f'copying raw {subset} files'):
            f = os.path.split(p)[-1] 
            shutil.copy(
                src=os.path.join(src_subset_dir, f),
                dst=os.path.join(dst_subset_dir, f)
            )

