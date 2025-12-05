import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from crossgoose.cellpose import transforms


def dp_to_rgb(dP, normalize: bool = False):
    """ dP is 2 x Y x X => 'optic' flow representation 

    Parameters
    -------------

    dP: 2xLyxLx array
        Flow field components [dy,dx]

    """

    norm = np.sqrt(np.sum(dP**2, axis=0))

    if normalize:
        norm = transforms.normalize99(norm)
    mag = 255 * np.clip(norm, 0, 1.)
    angles = np.arctan2(dP[1], dP[0]) + np.pi
    a = 2
    mag /= a
    rgb = np.zeros((*dP.shape[1:], 3), "uint8")
    rgb[..., 0] = np.clip(mag * (np.cos(angles) + 1), 0, 255).astype("uint8")
    rgb[..., 1] = np.clip(
        mag * (np.cos(angles + 2 * np.pi / 3) + 1), 0, 255).astype("uint8")
    rgb[..., 2] = np.clip(
        mag * (np.cos(angles + 4 * np.pi / 3) + 1), 0, 255).astype("uint8")
    return rgb


def show_flow_lines(
    mu: np.ndarray,
    ax: plt.Axes,
    n_steps: int,
    init_pts: np.ndarray | None = None,
    pts_period: int = 10,
    clip: bool = True
):
    shape = mu[0].shape

    if init_pts is None:
        pts = np.mgrid[:shape[0]:pts_period, :shape[1]:pts_period]
        pts = np.transpose(pts, (1, 2, 0)).reshape(-1, 2)
    else:
        pts = init_pts.copy()

    a_min = np.array([0, 0])
    a_max = np.array(shape) - 1

    log_pts = [pts]
    for i in range(n_steps):
        pts = log_pts[-1]
        if clip:
            pts = np.clip(pts, a_min=a_min, a_max=a_max)
        grad = mu[:, pts[:, 0].astype(int), pts[:, 1].astype(int)].T
        pts = pts + grad
        log_pts.append(pts)
    log_pts = np.stack(log_pts, axis=0)

    # ax.imshow(dp_to_rgb(mu))
    ax.plot(log_pts[:, :, 1], log_pts[:, :, 0], color='white',
            alpha=0.65, zorder=1, linewidth=0.4)

def get_custom_mask_cmap():
    base_cmap = mpl.colormaps['RdYlGn'].resampled(256)
    newcolors = base_cmap(np.linspace(0, 1, 256))
    # newcolors = base_cmap((np.linspace(0, 1, 256)%(1/8))*8)
    # newcolors = base_cmap(rng.random(256))
    newcolors[0, :] = [0,0,0,1]
    return ListedColormap(newcolors)