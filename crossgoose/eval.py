import os
import re
import time
from dataclasses import dataclass
from typing import List

import jsonargparse
import numpy as np
import pandas as pd
import tifffile
import torch
import yaml
from natsort import natsorted
from tqdm import tqdm

from crossgoose.cellpose.metrics import (aggregated_jaccard_index,
                                         average_precision)
from crossgoose.data import ImageNormalization, normalize_image
from crossgoose.mask_utils import SaveFormat, save_masks
from crossgoose.models import Ckptcriterion, CrossGooseModel
from crossgoose.utils import append_to_dict_items, imread


def _eval_model(
        model_path: str,
        results_dir: str,
        dataset: str,
        subset: str,
        ckpt_crit: Ckptcriterion,
        image_normalization: ImageNormalization,
        thresholds:List[float]
):
    model = CrossGooseModel.load_model(model_path, ckpt_crit=ckpt_crit)
    model = model.to('cuda')
    model.eval()
    if '/' in model_path:
        model_name = os.path.split(model_path)[-1]
    else:
        model_name = model_path
    print(f"model {model_name} loaded ")

    dataset_name = os.path.split(dataset)[-1]

    results_dir = os.path.join(results_dir, dataset_name, subset, model_name)
    os.makedirs(results_dir, exist_ok=True)

    # load images
    data_dir = os.path.join(dataset, subset)
    dir_file = os.listdir(data_dir)
    assert len(dir_file) > 0, f"found no files in {data_dir}"

    images_files = natsorted(
        [f for f in dir_file if re.search(r'.*\_img\.(tif|png)', f)])

    masks_files = natsorted(
        [f for f in dir_file if re.search(r'.*\_masks\.(tif|png)', f)])

    assert len(images_files) == len(
        masks_files), (len(images_files), len(masks_files))

    label_colormap = (np.random.random((3, 256)) *
                      (3 * 2**16)).astype(np.uint16)
    label_colormap[:, 0] = 0

    n_images = len(images_files)
    image_logs = {}
    masks_pred = []
    masks_true = []
    # segment each image
    for i in tqdm(range(n_images), desc='running inference'):
        img_name = images_files[i].split('.')[0]
        image = imread(os.path.join(data_dir, images_files[i]))
        masks_true.append(imread(os.path.join(data_dir, masks_files[i])))

        if len(image.shape) == 3:
            chan_dim = np.argmin(image.shape)
            image = np.mean(image, axis=chan_dim)
        h, w = image.shape
        image = normalize_image(image, image_normalization)

        image = torch.tensor(image)
        image = image.unsqueeze(dim=0).unsqueeze(dim=0).float()
        start = time.perf_counter()
        results = model.segment_image(image=image)
        end = time.perf_counter()
        results['timings']['total'] = end - start

        masks_pred.append(results['mask'])
        save_masks(
            results['mask'],
            os.path.join(results_dir, img_name + '_pred_mask.tif'),
            format=SaveFormat.IMAGEJ
        )
        tifffile.imwrite(
            os.path.join(results_dir, img_name + '_pred_cellprob.tif'),
            data=(results['cellprob']*255).astype(np.uint8),
            photometric='minisblack',
            imagej=True,
            metadata={'axes': 'YX', }
        )
        if 'overlap' in results:
            tifffile.imwrite(
                os.path.join(results_dir, img_name + '_pred_overlap.tif'),
                data=(results['overlap']*255).astype(np.uint8),
                photometric='minisblack',
                imagej=True,
                metadata={'axes': 'YX', }
            )

        log_i = {
            'image': img_name,
            'image_h': h, 'image_w': w,
            **{f"timing_{k}": v for k, v in results['timings'].items()}
        }

        append_to_dict_items(image_logs, log_i)

    aji = aggregated_jaccard_index(
        masks_true=masks_true,
        masks_pred=masks_pred
    )
    ap, tp, fp, fn = average_precision(
        masks_true=masks_true,
        masks_pred=masks_pred,
        threshold=thresholds
    )

    image_logs['aggregated_jaccard_index'] = aji.tolist()
    for i, t in enumerate(thresholds):
        image_logs[f'ap_{t}'] = ap[:, i].tolist()
        image_logs[f'tp_{t}'] = tp[:, i].tolist()
        image_logs[f'fp_{t}'] = fp[:, i].tolist()
        image_logs[f'fn_{t}'] = fn[:, i].tolist()

    image_logs_df = pd.DataFrame.from_records(image_logs)
    image_logs_df.to_csv(os.path.join(results_dir, 'results.csv'))

    # aggregated results
    agg_results = {}
    agg_keys = [f'ap_{t}' for t in thresholds] + \
        ['aggregated_jaccard_index', 'timing_total']

    for k in agg_keys:
        agg_results[k] = float(np.mean(image_logs[k]))

    with open(os.path.join(results_dir, 'results_aggregated.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(agg_results, f)


@dataclass
class ModelEvalConfig:
    model: str
    dataset: str
    image_normalization: ImageNormalization = 'M1P1'
    ckpt_crit: Ckptcriterion = 'best_ap_0.75'
    subset: str = 'test'


def eval_models(
    evaluations: ModelEvalConfig | List[ModelEvalConfig],
    thresholds: List[float],
    results_dir: str = 'results',
    data_root: str = 'data'
):
    if not isinstance(evaluations, list):
        evaluations = [evaluations]

    for exp in tqdm(evaluations, desc='evaluating models on datasets'):
        _eval_model(
            model_path=exp.model,
            results_dir=results_dir,
            dataset=os.path.join(data_root, exp.dataset),
            subset=exp.subset,
            image_normalization=exp.image_normalization,
            ckpt_crit=exp.ckpt_crit,
            thresholds=thresholds
        )


if __name__ == "__main__":
    jsonargparse.auto_cli(eval_models, as_positional=False)
