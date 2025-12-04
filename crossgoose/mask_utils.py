from enum import Enum
from typing import Literal

import edt
import networkx as nx
import numpy as np
import scipy.ndimage as ndi
import skimage
import tifffile
from skimage import morphology

from crossgoose.graph_utils import (get_networkx_graph_from_array,
                                    largest_component)


def relabel(mask: np.ndarray, shuffle: bool = False, shuffle_seed: int = 0):
    new_labels = np.zeros_like(mask)
    unique_labels = np.unique(mask)
    if shuffle:
        rng = np.random.default_rng(shuffle_seed)
        unique_labels = rng.permutation(unique_labels)
    i = 1
    for l in unique_labels:
        if l != 0:
            new_labels[mask == l] = i
            i += 1
    return new_labels


def most_distant_point(mask):
    dist = edt.edt(mask, black_border=True)
    return np.unravel_index(np.argmax(dist, axis=None), dist.shape)


def graph_barycenter(mask):
    # mask = morphology.closing(mask, footprint=morphology.disk(8))
    s = morphology.skeletonize(mask)
    g = get_networkx_graph_from_array(s > 0)
    g = largest_component(g)
    return nx.barycenter(g)[0]


def graph_center(mask):
    # mask = morphology.closing(mask, footprint=morphology.disk(8))
    s = morphology.skeletonize(mask)
    g = get_networkx_graph_from_array(s > 0)
    g = largest_component(g)
    return nx.center(g)[0]


def graph_center_weighted(mask):
    # mask = morphology.closing(mask, footprint=morphology.disk(8))
    s = morphology.skeletonize(mask)
    g = get_networkx_graph_from_array(s > 0)
    g = largest_component(g)
    dist = edt.edt(mask)
    values = {}
    for e in g.edges():
        u, v = e
        d1 = dist[u[0], u[1]]
        d2 = dist[v[0], v[1]]
        values[(u, v)] = (d1 + d2)/2
    nx.set_edge_attributes(g, values, 'thickness')
    return nx.center(g, weight='thickness')[0]


CENTER_METHODS = {
    'mass': ndi.center_of_mass,
    'dist': most_distant_point,
    'graph_barycenter': graph_barycenter,
    'graph_center': graph_center,
    'graph_center_weighted': graph_center_weighted
}

CenterMethod = Literal['mass', 'dist', 'graph_barycenter',
                       'graph_center', 'graph_center_weighted']


def get_centers(masks, slices, one_hot_masks: bool = False, method: CenterMethod = 'mass'):
    c_fun = CENTER_METHODS[method]
    if one_hot_masks:
        centers = [c_fun(masks[i][slices[i]] > 0) for i in range(len(slices))]
    else:
        centers = [c_fun(masks[slices[i]] == (i+1))
                   for i in range(len(slices))]
    centers = np.array([np.array([centers[i][0] + slices[i][0].start, centers[i][1] + slices[i][1].start])
                        for i in range(len(slices))])
    exts = np.array([(slc[0].stop - slc[0].start) +
                    (slc[1].stop - slc[1].start) + 2 for slc in slices])
    return centers, exts


def clean_unconnected(mask):
    mask = mask.copy()
    labels = np.unique(mask)
    new_label = np.max(labels) + 1
    for l in labels:
        if l != 0:
            submask, _ = ndi.label(mask == l)
            sub_uniqe = np.unique(submask)
            if len(sub_uniqe) > 2:
                for ll in sub_uniqe:
                    if ll != 0:
                        mask[submask == ll] = new_label
                        new_label += 1
    mask = relabel(mask)
    return mask


class SaveFormat(Enum):
    IMAGEJ = 'ImageJ'
    OME = 'OME'


def save_masks(masks: np.ndarray, file: str, format: SaveFormat):

    rng = np.random.default_rng(0)

    if format == SaveFormat.IMAGEJ:
        mask_data_type = np.uint8
        imwrite_params = dict(
            imagej=True
        )
    elif format == SaveFormat.OME:
        mask_data_type = np.uint16
        imwrite_params = dict(
            ome=True
        )
    else:
        raise ValueError(format)

    max_label = masks.max()
    if max_label >= np.iinfo(mask_data_type).max:
        raise OverflowError(
            f"labels max value {max_label} exeed max {mask_data_type} ({np.iinfo(mask_data_type).max})")

    masks = masks.astype(mask_data_type)

    label_colormap = (rng.random((3, 2**(masks.itemsize*8)))
                      * (3 * 2**16)).astype(np.uint16)
    label_colormap[:, 0] = 0

    tifffile.imwrite(
        file,
        data=masks,
        colormap=label_colormap,
        compression='zlib',
        metadata={'axes': 'YX'},
        **imwrite_params
    )


def convert_labels_to_onehot(
    labels: np.ndarray,
    closure_radius: int | None = None,
) -> np.ndarray:
    max_label = np.max(labels)
    masks_oh = np.zeros((max_label,) + labels.shape, dtype=np.uint)

    if closure_radius:
        disk = skimage.morphology.disk(radius=closure_radius)
    else:
        disk = None

    for l in range(max_label+1):
        if l != 0:
            mask_bin = labels == l
            if closure_radius:
                mask_bin = skimage.morphology.closing(mask_bin, disk)
            masks_oh[l-1] = (mask_bin).astype(np.uint)


    return masks_oh