

import os
import re
from typing import List, Tuple

import numpy as np
import skimage
import tifffile
from natsort import natsorted
from scipy.ndimage import find_objects
from tqdm import tqdm

from crossgoose.perlin_2d import generate_fractal_noise_2d
from crossgoose.utils import imread, remap


def compute_intersections(src_dir: str):
    dir_file = os.listdir(src_dir)
    masks_onehot_files = natsorted(
        [f for f in dir_file if re.search(r'.*\_masks_onehot\.(tif|png)', f)])
    masks_onehot = [imread(os.path.join(src_dir, f))
                    for f in tqdm(masks_onehot_files, desc='loading masks')]

    data = []
    for mask in tqdm(masks_onehot, desc='computing'):
        n_labels = mask.shape[0]
        all_idx = set(np.arange(n_labels))
        for k in range(n_labels):
            this_mask = mask[k]
            others = mask[list(all_idx - set([k]))].max(axis=0)
            intersection = this_mask & others
            f_inter = np.sum(intersection)
            data.append(f_inter)

    return data


def _ensure_gs(arr: np.ndarray) -> np.ndarray:
    if len(arr.shape) == 2:
        return arr
    elif len(arr.shape) == 3:
        return np.mean(arr, axis=np.argmin(arr.shape))
    else:
        raise ValueError(arr.shape)


def load_files(src_dir: str):

    dir_file = os.listdir(src_dir)

    images_files = natsorted(
        [f for f in dir_file if re.search(r'.*\_img\.(tif|png)', f)])

    masks_files = natsorted(
        [f for f in dir_file if re.search(r'.*\_masks\.(tif|png)', f)])

    assert len(images_files) == len(masks_files)

    images = [_ensure_gs(remap(imread(os.path.join(src_dir, f)), 0, 1))
              for f in tqdm(images_files, desc="loading images")]
    labels = [imread(os.path.join(src_dir, f))
              for f in tqdm(masks_files, desc="loading labels")]

    return images, labels


def isolate_cells(image, label, filter_connected: bool):
    slices = find_objects(label)

    sliced_images = []
    sliced_masks = []

    for i, s in enumerate(slices):
        l = i+1
        loc_mask = label[s] == l
        loc_image = image[s]

        if filter_connected:
            _, num_components = skimage.measure.label(
                loc_mask, return_num=True)
            save_flag = num_components == 1
        else:
            save_flag = True

        if save_flag:
            sliced_masks.append(loc_mask)
            sliced_images.append(loc_image)

    return sliced_images, sliced_masks


class WormImageGenerator:

    def __init__(self, src_dir, seed: int = 0):
        self.src_dir = src_dir
        self.rng = np.random.default_rng(seed)
        self._load_slices()

    def _load_slices(self):
        images, labels = load_files(self.src_dir)
        self.sliced_images = []
        self.sliced_masks = []
        for img, lbl in zip(images, tqdm(labels, desc="extracting slices")):
            s_img, s_lbl = isolate_cells(img, lbl, filter_connected=True)
            self.sliced_images = self.sliced_images + s_img
            self.sliced_masks = self.sliced_masks + s_lbl

    def get_n_worms(self):
        return len(self.sliced_images)

    def get_worm_index(self, i: int):
        return self.sliced_images[i], self.sliced_masks[i]

    def make_image(
        self,
        shape: Tuple[int, int],
        padding: int,
        n_worm: int,
        rotate: bool,
        alpha: float = 0.9,
        noise_fac: float = 0.01,
        stacking: str = "alpha",
        limit_overlaps: int = -1,
        max_intersection: float | None = None,
        blur: float = 0,
        bg_range: List[float] = [0.5, 1.0]
    ):
        shape_pad = (shape[0]+2*padding, shape[1]+2*padding)
        image = self.make_bg(
            shape_pad,
            bg_min=bg_range[0],
            bg_max=bg_range[1],
            noise_fac=0.1,
            noise_res=2,
            octaves=3
        )
        labels = np.zeros(shape_pad, dtype=np.uint)
        labels_oh = np.zeros((n_worm,) + shape_pad, dtype=bool)
        overlap = np.zeros(shape_pad, dtype=np.uint)
        n_worm_lib = self.get_n_worms()

        for k in range(n_worm):
            l = k+1

            valid_cell_flag = False
            tries = 0

            while not valid_cell_flag:
                tries += 1

                idx = self.rng.choice(n_worm_lib)
                img, msk = self.get_worm_index(idx)
                if rotate:
                    angle = self.rng.random() * 360
                    img = skimage.transform.rotate(
                        img, angle, resize=True, mode='edge')
                    msk = skimage.transform.rotate(
                        msk, angle, resize=True, mode='constant', cval=0)

                s0, s1 = msk.shape
                i, j = self.rng.integers(
                    (0, 0), (shape_pad[0]-s0, shape_pad[1]-s1))

                mask_glob = np.zeros(shape_pad, dtype=float)
                mask_glob[i:i+s0, j:j+s1] = msk

                valid_cell_flag = True

                if limit_overlaps > -1:
                    new_overlap = overlap + mask_glob
                    max_overlap = np.max(new_overlap)
                    if max_overlap > limit_overlaps:
                        valid_cell_flag = False

                if max_intersection is not None:
                    intersection = (mask_glob > 0) & (labels > 0)
                    intersection = np.sum(intersection)
                    if intersection > max_intersection:
                        valid_cell_flag = False

                if (not valid_cell_flag) and tries > 100:
                    print(
                        f"Cannot add new cell with contraint {limit_overlaps=}")
                    break

            mask_filtered = remap(
                skimage.filters.gaussian(mask_glob, sigma=1.0), 0, 1)

            if stacking == "multiply":
                img_glob = np.ones_like(image)
            else:
                img_glob = np.zeros_like(image)
            img_glob[i:i+s0, j:j+s1] = img

            mask_bin = mask_filtered > 0.75
            labels[mask_bin] = l
            labels_oh[l-1][mask_bin] = 1
            overlap[mask_bin] += 1

            fac = mask_filtered * alpha

            # image[mask_glob>0] = alpha * img[msk>0] + (1-alpha) * image[mask_glob>0]

            if stacking == "alpha":
                image = (1-fac) * image + fac * img_glob
            elif stacking == "substract":
                image = np.maximum(0, image - fac * img_glob)
            elif stacking == "multiply":
                image = ((image) * (1-fac * img_glob))
            else:
                raise ValueError(stacking)

        image = image[padding:-padding, padding:-padding]
        labels = labels[padding:-padding, padding:-padding]
        labels_oh = labels_oh[:, padding:-padding, padding:-padding]
        assert image.shape == shape, image.shape
        assert labels.shape == shape, labels.shape
        assert labels_oh.shape[1:] == shape, labels.shape

        if blur > 0:
            image = skimage.filters.gaussian(image, sigma=blur)

        noise = self.rng.normal(size=shape)
        image = (1-noise_fac) * image + (noise_fac) * noise
        image = remap(image, 0, 1)

        return {
            "image": image,
            "labels": labels,
            "labels_oh": labels_oh
        }

    def make_bg(
        self,
        shape: Tuple[int, int],
        bg_min,
        bg_max,
        noise_fac: float,
        octaves: int = 3,
        noise_res: int = 2
    ):
        # image = np.zeros((h, w))
        h, w = shape
        ii, jj = np.mgrid[:h, :w]
        dist = (np.square(ii - (h/2)) + np.square(jj - (w/2)))
        bg = remap(-dist, bg_min, bg_max)

        large_noise = self.rng.normal(size=(h // 16, w // 16))
        large_noise = skimage.transform.resize(large_noise, (h, w), order=1)

        noise = generate_fractal_noise_2d(
            shape, (noise_res, noise_res), octaves=octaves, rng=self.rng)

        return (1-noise_fac)*bg + noise_fac*noise


def make_synth_data(
    src_dir: str,
    seed: int,
    shape: Tuple[int, int],
    n_images: int,
    save_dir: str,
    padding: int,
    cell_count_range: List[int],
    rotate: bool,
    blur_range: List[float],
    alpha: float,
    noise_range: List[float],
    max_intersection: int,
    limit_overlaps: int,
    stacking: str = "alpha",
    bg_range: List[float] = [0.5, 1.0]
):
    wig = WormImageGenerator(
        src_dir=src_dir, seed=seed
    )
    rng = wig.rng

    label_colormap = (rng.random((3, 256)) *
                      (3 * 2**16)).astype(np.uint16)
    label_colormap[:, 0] = 0

    blur_a = blur_range[0]
    blur_b = blur_range[1] - blur_range[0]

    noise_a = noise_range[0]
    noise_b = noise_range[1] - noise_range[0]

    for i in tqdm(range(n_images), desc='making images'):

        img_name = f"S{i:05}"

        n_worm = rng.integers(cell_count_range[0], cell_count_range[1])

        blur = rng.random() * blur_b + blur_a
        noise_fac = rng.random() * noise_b + noise_a

        res = wig.make_image(
            shape=shape,
            padding=padding,
            n_worm=n_worm,
            rotate=rotate,
            alpha=alpha,
            noise_fac=noise_fac,
            stacking=stacking,
            limit_overlaps=limit_overlaps,
            max_intersection=max_intersection,
            blur=blur,
            bg_range=bg_range
        )

        tifffile.imwrite(
            os.path.join(save_dir, img_name + '_img.tif'),
            data=res['image'].astype(np.float32),
            photometric='minisblack',
            imagej=True,
            metadata={'axes': 'YX', }
        )
        tifffile.imwrite(
            os.path.join(save_dir, img_name + '_masks.tif'),
            data=res['labels'].astype(np.uint8),
            photometric='minisblack',
            colormap=label_colormap,
            imagej=True,
            metadata={'axes': 'YX', }
        )
        tifffile.imwrite(
            os.path.join(save_dir, img_name + '_masks_onehot.tif'),
            data=res['labels_oh'].astype(np.uint8),
            photometric='minisblack',
            imagej=True,
            metadata={'axes': 'ZYX', }
        )
