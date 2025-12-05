from __future__ import annotations

import glob
import os
import pathlib
import re
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Literal

import lightning
import numpy as np
import torch
import yaml
from lightning.pytorch.cli import LightningArgumentParser
from torch import nn

from crossgoose.cellpose import models as cp_models
from crossgoose.cellpose.dynamics import get_masks_torch
from crossgoose.cellpose.metrics import average_precision
from crossgoose.cellpose.resnet_torch import batchconv
from crossgoose.cellpose.transforms import get_pad_yx
from crossgoose.data import FlowDataModule


class SamplingMethod(Enum):
    FOLLOW_FLOWS = "follow_flows"
    RANDOM_ON_CELL = "random_on_cell"


class CPBackbone(nn.Module):
    def __init__(self, device, load_pretrained: bool = True):
        super().__init__()
        self.device = device

        self.nchan = 2
        nclasses = 3
        self.nbase = [32, 64, 128, 256]
        self.nbase = [self.nchan, *self.nbase]
        diam_mean = 30

        net = cp_models.CPnet(
            nbase=self.nbase, nout=nclasses, sz=3, mkldnn=False,
            max_pool=True, diam_mean=diam_mean).to(self.device)

        if load_pretrained:
            pretrained_model, diam_mean, _, _ = cp_models.get_model_params(
                pretrained_model='cyto3',
                model_type=None,
                pretrained_model_ortho=None,
                default_model='cyto3')

            net.load_model(pretrained_model, device=self.device)

        self.downsample = net.downsample
        self.make_style = net.make_style
        self.upsample = net.upsample
        # self.output = net.output

    def forward(self, data):

        c = data.shape[1]
        if c != self.nchan:
            raise ValueError(
                f"data.shape[1]={c} does not mach n_chan={self.nchan}")

        # the cellpose way
        T0 = self.downsample(data)

        style = self.make_style(T0[-1])

        T1 = self.upsample(style, T0, False)
        # T1 is of feature size 32
        # T2 = self.output(T1)

        return T1, style, T0


def linblock(in_channels, out_channels):
    return nn.Sequential(
        nn.ReLU(),
        nn.Linear(in_channels, out_channels),
    )


class FlowFunction(ABC, nn.Module):

    @abstractmethod
    def forward(self, e0, et):
        raise NotImplementedError


class FlowAttention(FlowFunction):
    def __init__(
        self,
        embedding_dim: int,
        key_dim: int,
        nb_key: int,
        value_dim: int = 2
    ):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.nb_key = nb_key
        self.embedding_dim = embedding_dim

        self.query_module = linblock(embedding_dim, key_dim)
        self.key_module = linblock(embedding_dim, nb_key*key_dim)
        self.value_module = linblock(embedding_dim, nb_key*value_dim)

    def forward(self, e0, et):

        n, f = e0.shape
        assert tuple(et.shape) == (n, f), (et.shape, (n, f))

        query = self.query_module(e0).view(n, 1, self.key_dim)
        key = self.key_module(et).view(n, self.nb_key, self.key_dim)
        value = self.value_module(et).view(n, self.nb_key, self.value_dim)

        # https://docs.pytorch.org/docs/2.4/generated/torch.nn.functional.scaled_dot_product_attention.html
        return nn.functional.scaled_dot_product_attention(  # pylint: disable=E1102
            query=query,
            key=key,
            value=value
        ).squeeze(dim=-2)


class FlowLinear(FlowFunction):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: List[int],
        value_dim: int = 2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        all_dims = hidden_dims + [value_dim]
        modules = []
        last_dim = embedding_dim*2
        for dim in all_dims:
            modules.append(
                linblock(
                    in_channels=last_dim,
                    out_channels=dim
                ))
            last_dim = dim

        self.transform = nn.Sequential(
            *modules
        )

    def forward(self, e0, et):
        n, f = e0.shape
        assert tuple(et.shape) == (n, f)
        e0et = torch.concat([e0, et], dim=1)

        return self.transform(e0et)


Ckptcriterion = Literal['last', 'best_ap_0.5', 'best_ap_0.75', 'best_ap_0.9']


class CrossGooseModel(lightning.LightningModule):
    def __init__(
        self,
        flow_fn: FlowFunction,
        embeddings_dim: int = 8,
        crit_flow_weight: float = 0.1,
        crit_cellprob_weight: float = 2.0,
        n_steps: int = 200,
        n_samples: int = 50,
        sampling_method: SamplingMethod = SamplingMethod.FOLLOW_FLOWS,
        random_on_cell_sigma: float = 4.0,
        overlap_focus_multiplier: float = 1.0,
        backbone_realease_delay: int | None = None
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.embeddings_dim = embeddings_dim
        self.sampling_method = sampling_method
        self.random_on_cell_sigma = random_on_cell_sigma
        self.overlap_focus_multiplier = overlap_focus_multiplier
        self.backbone_realease_delay = backbone_realease_delay

        self.backbone = CPBackbone(device=self.device)
        self.backbone_out = self.backbone.nbase[1]

        self.flow_fn = flow_fn

        out_c = (2*self.embeddings_dim) + 1

        self.embedding_head = batchconv(
            in_channels=self.backbone_out,
            out_channels=out_c,
            sz=1
        )

        self.criterion_flow = nn.MSELoss(reduction="mean")
        self.criterion_cellprob = nn.BCEWithLogitsLoss(reduction="mean")

        self.flow_fac = 5.
        self.crit_flow_weight = crit_flow_weight
        self.crit_cellprob_weight = crit_cellprob_weight

        if self.backbone_realease_delay:
            self.backbone.requires_grad_(False)
            print(
                f"[0] freezing backbone training until epoch={self.backbone_realease_delay}")

        self.save_hyperparameters(ignore=['flow_fn', 'positional_embedding'])

    @staticmethod
    def load_model(model: str = 'default', ckpt_crit: Ckptcriterion = 'last') -> CrossGooseModel:
        """loads a model
        Args:
            model (str): path to a directory or name of a default model (eg 'default')
                If path to a directory, it should have format like: 
                    ├── checkpoints
                    │   ├── best-epoch=163-step=181712-val_ap_0.5=0.9320.ckpt
                    │   ├── best-epoch=223-step=248192-val_ap_0.9=0.2635.ckpt
                    │   ├── best-epoch=38-step=43212-val_ap_0.75=0.8247.ckpt
                    │   └── last.ckpt
                    └── config.yaml
            ckpt_crit (Ckptcriterion) : if model is a directory, 
                one of 'last', 'best_ap_0.5', 'best_ap_0.75', 'best_ap_0.9'

        Returns:
            CrossGooseModel: loaded model
        """
        if os.path.isdir(model):
            return CrossGooseModel._load_model_from_dir(model, ckpt_crit=ckpt_crit)
        else:
            models_dir = pathlib.Path(
                __file__).parent.resolve().joinpath('models')
            if not os.path.exists(models_dir):
                raise FileNotFoundError(
                    f"could not find models dir at {str(models_dir)}")
            available_models = os.listdir(models_dir)
            if not model in available_models:
                raise FileNotFoundError(
                    f"no model {model} in exisitng models {available_models}")

            ckpt_path = os.path.join(models_dir, model, 'weights.ckpt')
            config_path = os.path.join(models_dir, model, 'config.yaml')
            return CrossGooseModel._load_from_ckpt(
                ckpt_path=ckpt_path,
                config_path=config_path
            )

    @staticmethod
    def _load_from_ckpt(
        ckpt_path: str, config_path: str
    ) -> CrossGooseModel:
        assert os.path.exists(ckpt_path)
        assert os.path.exists(config_path)

        parser = LightningArgumentParser()
        parser.add_class_arguments(CrossGooseModel, 'model')
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_cfg_dict = yaml.safe_load(f)
        cfg_dict = {'model': loaded_cfg_dict['model']}
        loaded_cfg = parser.parse_object(cfg_dict)
        exp = parser.instantiate_classes(loaded_cfg)
        model: CrossGooseModel = exp.model
        with open(ckpt_path, 'rb') as f:
            state_dict = torch.load(f, weights_only=True)
        model.load_state_dict(state_dict['state_dict'])
        print(f"loaded model from {ckpt_path}")
        return model

    @staticmethod
    def _load_model_from_dir(
        model_dir: str,
        ckpt_crit: Ckptcriterion = 'last'
    ) -> CrossGooseModel:
        ckpt_dir = os.path.join(model_dir, 'checkpoints')
        checkpoints = glob.glob(ckpt_dir+'/*.ckpt')
        if len(checkpoints) > 0:
            if ckpt_crit == 'last':
                ckpt = [f for f in checkpoints if re.search(
                    'last', os.path.split(f)[-1])]
                assert len(ckpt) == 1
                ckpt_path = ckpt[0]
            elif 'best_ap' in ckpt_crit:
                if ckpt_crit == 'best_ap_0.5':
                    re_pat = re.compile(r'best-.*-val_ap_0\.5=(.*)\.ckpt')
                elif ckpt_crit == 'best_ap_0.75':
                    re_pat = re.compile(r'best-.*-val_ap_0\.75=(.*)\.ckpt')
                elif ckpt_crit == 'best_ap_0.9':
                    re_pat = re.compile(r'best-.*-val_ap_0\.9=(.*)\.ckpt')
                else:
                    raise ValueError(ckpt_crit)

                files = []
                metrics = []
                for p in checkpoints:
                    filename = os.path.split(p)[-1]
                    m = re_pat.match(filename)
                    if m is not None:
                        files.append(p)
                        metrics.append(float(m.group(1)))

                best_i = np.argmax(metrics)
                ckpt_path = files[best_i]
            else:
                raise ValueError(ckpt_crit)
        else:
            raise FileNotFoundError(f"not *ckpt files at {ckpt_dir}")
        config_path = os.path.join(model_dir, 'config.yaml')
        return CrossGooseModel._load_from_ckpt(
            ckpt_path=ckpt_path,
            config_path=config_path
        )

    def image_to_maps(self, image, apply_sigmoids: bool = False):
        assert len(image.shape) == 4

        T0, _, _ = self.backbone(image)
        T1 = self.embedding_head(T0)

        res = {'T1': T1}

        res['emb_grid_0'] = T1[:, :self.embeddings_dim]
        res['emb_grid_t'] = T1[:, self.embeddings_dim:2*self.embeddings_dim]
        cp_est = T1[:, -1]
        if apply_sigmoids:
            res['cp_est'] = nn.functional.sigmoid(cp_est)
        else:
            res['cp_est'] = cp_est

        return res

    def forward(self, image: torch.Tensor, u0: torch.Tensor, ut: torch.Tensor):
        _, c, _, _ = image.shape

        if c == 1:
            image = torch.tile(image, (1, 2, 1, 1))

        features = self.image_to_maps(image, apply_sigmoids=True)
        emb_grid_0 = features['emb_grid_0']
        emb_grid_t = features['emb_grid_t']

        e0 = self._gather_emb_batch(
            emb_grid_0, u0)
        et = self._gather_emb_batch(
            emb_grid_t, ut)

        dP = self.flow_fn(e0, et)

        return dP, features['cp_est'], features['T1'], features.get('overlap_est', None)

    def _gather_emb_batch(self, raster: torch.Tensor, u: torch.Tensor):
        # expects raster of shape (B,dim_emb,H,W)
        idx0 = u[:, 0].long()
        idx1 = u[:, 1].long()
        idx2 = u[:, 2].long()
        gathered_emb = raster[idx0, :, idx1, idx2]
        return gathered_emb

    def _sample_uts(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
        u0: torch.Tensor,
        l0: torch.Tensor
    ):
        if self.sampling_method == SamplingMethod.FOLLOW_FLOWS:
            _, log_ut, _ = self.follow_flow(
                image, n_steps=self.n_steps, as_numpy=False, u0=u0
            )
            step_samples = np.linspace(
                0, self.n_steps, self.n_samples, dtype=int)
            sample_weights = [
                1 / self.n_samples for _ in range(self.n_samples)]
            ut_samples = log_ut[step_samples]
        elif self.sampling_method == SamplingMethod.RANDOM_ON_CELL:
            batch_size, _, h, w = image.shape
            min_bound = torch.tensor([0, 0], device=u0.device)
            max_bound = torch.tensor([h, w], device=u0.device)-1
            ut_samples = torch.zeros(
                (self.n_samples,)+u0.shape, dtype=torch.float, device=u0.device)
            ut_samples[:, :, 0] = u0[:, 0]
            for b in range(batch_size):
                batch_pt_mask = u0[:, 0] == b
                unique_labels = torch.unique(l0[batch_pt_mask])
                for l in unique_labels:
                    labels_pt_mask = (l0 == l) & batch_pt_mask
                    label_pts = torch.nonzero(labels[b] == l)
                    n = int(torch.sum(labels_pt_mask))
                    m = label_pts.shape[0]

                    samples_idx = torch.randint(
                        0, m, size=(self.n_samples, n))
                    samples = label_pts[samples_idx].float()

                    pert = self.random_on_cell_sigma * \
                        torch.randn_like(samples)
                    samples = torch.clamp(
                        samples + pert,
                        min=min_bound, max=max_bound
                    )
                    ut_samples[:, labels_pt_mask, 1:] = samples
            sample_weights = [
                1 / self.n_samples for _ in range(self.n_samples)
            ]
        else:
            raise ValueError(self.sampling_method)

        return ut_samples, sample_weights

    def _sample_gtflows_batch(self, gt_flows: torch.Tensor, l0: torch.Tensor, ut: torch.Tensor):
        # gt flows are a tensor of shape B,N_max,2,H,W
        # N_max the max nb of instances accross the batch
        # flow for label l is gt_flows[:,l-1]

        # index on source/current batch (u0[:,0] should be same as ut[:,0])
        idx0 = ut[:, 0].long()
        idx1 = (l0-1).long()  # check flow for source label
        idx2 = ut[:, 1].long()  # check flows at current position
        idx3 = ut[:, 2].long()  # same

        flows = gt_flows[idx0, idx1, :, idx2, idx3]
        return flows

    def training_step(self, batch, batch_idx):

        labels: torch.Tensor = batch['labels']
        flow_raster_gt: torch.Tensor = batch['flows']
        image: torch.Tensor = torch.tile(batch['image'], (1, 2, 1, 1))

        batch_size = flow_raster_gt.shape[0]

        loss_dict = {'loss': 0.0}

        u0 = torch.nonzero(labels)
        l0 = labels[u0[:, 0], u0[:, 1], u0[:, 2]]
        u0 = u0.float()

        # sample points
        ut_samples, sample_weights = self._sample_uts(
            image=image, u0=u0, labels=labels, l0=l0
        )

        features = self.image_to_maps(image, apply_sigmoids=False)
        emb_grid_0 = features['emb_grid_0']
        emb_grid_t = features['emb_grid_t']

        cell_gt = (labels > 0).float()
        loss_dict['loss_cp'] = self.criterion_cellprob(
            features['cp_est'], cell_gt
        ) * self.crit_cellprob_weight

        loss_dict['loss'] += loss_dict['loss_cp']

        # get u0
        e0 = self._gather_emb_batch(emb_grid_0, u0)

        loss_steps = 0
        for i in range(self.n_samples):
            ut = ut_samples[i].detach()
            et = self._gather_emb_batch(emb_grid_t, ut)
            flow_est = self.flow_fn(e0, et)
            flow_gt = self._sample_gtflows_batch(
                gt_flows=flow_raster_gt,
                l0=l0,
                ut=ut
            )
            loss_i = self.criterion_flow(flow_est, self.flow_fac * flow_gt)
            loss_steps = loss_steps + loss_i * sample_weights[i]

        loss_dict['loss_steps'] = loss_steps
        loss_dict['loss'] += loss_dict['loss_steps']

        self.log_dict(
            loss_dict, prog_bar=True,
            logger=True, on_step=False, on_epoch=True,
            batch_size=batch_size
        )

        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        images: torch.Tensor = batch['image']
        batch_size, _, h, w = images.shape
        thresholds = [0.5, 0.75, 0.9]
        masks_true = [batch['labels'][i].squeeze().cpu().numpy()
                      for i in range(batch_size)]
        masks_pred = []
        for i in range(batch_size):
            image = images[[i]]
            results = self.segment_image(image=image)
            masks_pred.append(results['mask'])

        ap, _, _, _ = average_precision(
            masks_true=masks_true,
            masks_pred=masks_pred,
            threshold=thresholds
        )
        log = {f"val_ap_{t}": float(np.nanmean(ap[:, i]))
               for i, t in enumerate(thresholds)}

        self.log_dict(log, batch_size=batch_size, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    @torch.no_grad
    def follow_flow(
        self,
        image,
        n_steps: int,
        as_numpy: bool = True,
        u0: torch.Tensor | None = None,
        skip_logging: bool = False,
    ):

        _, c, h, w = image.shape
        if c == 1:
            image = torch.tile(image, (1, 2, 1, 1))

        features = self.image_to_maps(image, apply_sigmoids=True)
        emb_grid_t = features['emb_grid_t']
        emb_grid_0 = features['emb_grid_0']

        fg = features['cp_est'] > 0.5

        if u0 is None:
            u0 = torch.nonzero(fg)

        min_bound = torch.tensor([0, 0], device=u0.device)
        max_bound = torch.tensor([h, w], device=u0.device)-1

        e0 = self._gather_emb_batch(
            emb_grid_0, u0)

        ut = u0.clone().float()
        log_ut = [u0.clone().float()]
        for _ in range(n_steps):
            et = self._gather_emb_batch(
                emb_grid_t, ut)
            flow_est = self.flow_fn(e0, et) / self.flow_fac
            ut_next = torch.concat((
                ut[:, [0]],
                torch.clamp(
                    ut[:, 1:] + flow_est,
                    min=min_bound, max=max_bound
                )
            ), dim=1)
            if skip_logging:
                log_ut = [ut_next]
            else:
                log_ut.append(ut_next)
            ut = ut_next

        if skip_logging:
            log_ut = log_ut[0]
        else:
            log_ut = torch.stack(log_ut, dim=0)

        if as_numpy:
            log_ut = log_ut.cpu().numpy()
            fg = fg.cpu().numpy()

        return fg, log_ut, features

    @torch.no_grad
    def segment_image(
        self,
        image: torch.Tensor
    ):
        assert len(image.shape) == 4, "expects grayscale images (for now)"
        b, c, h, w = image.shape
        assert b == 1
        assert c in [1, 2]

        timings = {}
        results = {}

        ypad1, ypad2, xpad1, xpad2 = get_pad_yx(
            h, w, div=16, extra=1, min_size=None)

        image_padded = nn.functional.pad(
            image, (xpad1, xpad2, ypad1, ypad2), value=-1)

        # follow flows
        start = time.perf_counter()
        fg, last_pt, features = self.follow_flow(
            image_padded.to(self.device), n_steps=self.n_steps,
            skip_logging=True)
        end = time.perf_counter()
        timings['follow_flow'] = end - start

        cp_est = features['cp_est'].cpu().numpy()[0]
        cp_est = cp_est[ypad1:-ypad2, xpad1:-xpad2]
        results['cellprob'] = cp_est

        # compute masks
        start = time.perf_counter()

        b = 0
        batch_mask = last_pt[:, 0] == b
        batch_pt = torch.from_numpy(
            last_pt[batch_mask][:, 1:]).permute(1, 0).long()
        fgb = fg[b]
        shape0 = fgb.shape
        inds = np.nonzero(fgb)

        masks = get_masks_torch(
            pt=batch_pt,
            inds=inds,
            shape0=shape0
        )

        masks = masks[ypad1:-ypad2, xpad1:-xpad2]
        assert masks.shape == (h, w)

        end = time.perf_counter()
        timings['compute_masks'] = end - start

        results['mask'] = masks
        results['timings'] = timings
        return results

    def on_train_epoch_start(self):
        trainer = self.trainer
        epoch = trainer.current_epoch
        if self.backbone_realease_delay == epoch:
            self.backbone.requires_grad_(True)
            print(f"[{epoch}] releasing backbone training")
