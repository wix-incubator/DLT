# code taken from https://github.com/ktrk115/const_layout/blob/master/metric.py
# and https://github.com/google-research/google-research/blob/master/layout-blt/utils/metrics.py
import os

import numpy as np
import multiprocessing as mp
import torch
from itertools import chain
from scipy.optimize import linear_sum_assignment
from pytorch_fid.fid_score import calculate_frechet_distance
from evaluation.fid.layoutnet import LayoutNet
from evaluation.utils import convert_xywh_to_ltrb


class LayoutFID:
    def __init__(self, dataset_name, device='cpu'):
        self.num_label = 13 if dataset_name == 'rico' else 5

        self.model = LayoutNet(self.num_label).to(device)

        # load pre-trained LayoutNet
        file_dir = os.path.dirname(os.path.abspath(__file__))
        tmpl = os.path.join(file_dir, 'fid_pretrained/layoutnet_{}.pth.tar')
        state_dict = torch.load(tmpl.format(dataset_name), map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.requires_grad_(False)
        self.model.eval()

        self.real_features = []
        self.fake_features = []

    def collect_features(self, bbox, label, padding_mask, real=False):
        if real and type(self.real_features) != list:
            return

        feats = self.model.extract_features(bbox.detach(), label, padding_mask)
        features = self.real_features if real else self.fake_features
        features.append(feats.cpu().numpy())

    def compute_score(self):
        feats_1 = np.concatenate(self.fake_features)
        self.fake_features = []

        if type(self.real_features) == list:
            feats_2 = np.concatenate(self.real_features)
            self.real_features = feats_2
        else:
            feats_2 = self.real_features

        mu_1 = np.mean(feats_1, axis=0)
        sigma_1 = np.cov(feats_1, rowvar=False)
        mu_2 = np.mean(feats_2, axis=0)
        sigma_2 = np.cov(feats_2, rowvar=False)

        return calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2)


def compute_iou(box_1, box_2):
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    elif isinstance(box_1, torch.Tensor):
        lib = torch
    else:
        raise NotImplementedError(type(box_1))

    l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = lib.maximum(l1, l2)
    r_min = lib.minimum(r1, r2)
    t_max = lib.maximum(t1, t2)
    b_min = lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max),
                   lib.zeros_like(a1[0]))

    au = a1 + a2 - ai
    iou = ai / au

    return iou


def __compute_maximum_iou_for_layout(layout_1, layout_2):
    score = 0.
    (bi, li), (bj, lj) = layout_1, layout_2
    N = len(bi)
    for l in list(set(li.tolist())):
        _bi = bi[np.where(li == l)]
        _bj = bj[np.where(lj == l)]
        n = len(_bi)
        ii, jj = np.meshgrid(range(n), range(n))
        ii, jj = ii.flatten(), jj.flatten()
        iou = compute_iou(_bi[ii], _bj[jj]).reshape(n, n)
        ii, jj = linear_sum_assignment(iou, maximize=True)
        score += iou[ii, jj].sum().item()
    return score / N


def __compute_maximum_iou(layouts_1_and_2):
    layouts_1, layouts_2 = layouts_1_and_2
    N, M = len(layouts_1), len(layouts_2)
    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray([
        __compute_maximum_iou_for_layout(layouts_1[i], layouts_2[j])
        for i, j in zip(ii, jj)
    ]).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)
    return scores[ii, jj]


def __get_cond2layouts(layout_list):
    out = dict()
    for bs, ls in layout_list:
        cond_key = str(sorted(ls.tolist()))
        if cond_key not in out.keys():
            out[cond_key] = [(bs, ls)]
        else:
            out[cond_key].append((bs, ls))
    return out


def compute_maximum_iou(layouts_1, layouts_2, n_jobs=None):
    c2bl_1 = __get_cond2layouts(layouts_1)
    keys_1 = set(c2bl_1.keys())
    c2bl_2 = __get_cond2layouts(layouts_2)
    keys_2 = set(c2bl_2.keys())
    keys = list(keys_1.intersection(keys_2))
    args = [(c2bl_1[key], c2bl_2[key]) for key in keys]
    with mp.Pool(n_jobs) as p:
        scores = p.map(__compute_maximum_iou, args)
    scores = np.asarray(list(chain.from_iterable(scores)))
    return scores.mean().item()


def compute_overlap(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.3 Overlapping Loss

    bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0)
    bbox = bbox.permute(2, 0, 1)

    l1, t1, r1, b1 = convert_xywh_to_ltrb(bbox.unsqueeze(-1))
    l2, t2, r2, b2 = convert_xywh_to_ltrb(bbox.unsqueeze(-2))
    a1 = (r1 - l1) * (b1 - t1)

    # intersection
    l_max = torch.maximum(l1, l2)
    r_min = torch.minimum(r1, r2)
    t_max = torch.maximum(t1, t2)
    b_min = torch.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max),
                     torch.zeros_like(a1[0]))

    diag_mask = torch.eye(a1.size(1), dtype=torch.bool,
                          device=a1.device)
    ai = ai.masked_fill(diag_mask, 0)

    ar = torch.nan_to_num(ai / a1)

    return ar.sum(dim=(1, 2)) / mask.float().sum(-1)


def compute_alignment(bbox, mask, discretize=True):
    # Attribute-conditioned Layout GAN
    # 3.6.4 Alignment Loss

    bbox = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = convert_xywh_to_ltrb(bbox)
    xc, yc = bbox[0], bbox[1]
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)
    # for fair comparison
    if discretize:
        X = torch.round(X * 32) / 32

    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.
    X = X.permute(0, 3, 2, 1)
    X[~mask] = 1.
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.), 0.)

    X = -torch.log(1 - X)

    return X.sum(-1) / mask.float().sum(-1)


def get_layout_iou(layout, pad_mask):
    layout = np.array(layout, dtype=np.float32)
    layout_channels = []
    for bbox, cur_mask in zip(layout, pad_mask):
        if not cur_mask:
            continue
        bbox = bbox * 200

        canvas = np.zeros((200, 200, 1), dtype=np.float32)
        width, height = bbox[2], bbox[3]
        center_x, center_y = bbox[0], bbox[1]
        # Avoid round behevior at 0.5.
        min_x = round(center_x - width / 2. + 1e-4)
        max_x = round(center_x + width / 2. + 1e-4)
        min_y = round(center_y - height / 2. + 1e-4)
        max_y = round(center_y + height / 2. + 1e-4)
        canvas[min_x:max_x, min_y:max_y] = 1.
        layout_channels.append(canvas)
    if not layout_channels:
        return 0.
    sum_layout_channel = np.sum(np.concatenate(layout_channels, axis=-1), axis=-1)
    overlap_area = np.sum(np.greater(sum_layout_channel, 1.))
    bbox_area = np.sum(np.greater(sum_layout_channel, 0.))
    if bbox_area == 0.:
        return 0.
    return overlap_area / bbox_area


def get_alignment_loss_numpy(layout):
    """Calculates alignment loss of bounding boxes.

  Rewrites the function in the layoutvae:
  alignment_loss_lib.py by numpy.

  Args:
    layout: [asset_num, asset_dim] float array. An iterable of normalized
      bounding box coordinates in the format (x_min, y_min, x_max, y_max), with
      (0, 0) at the top-left coordinate.

  Returns:
    Alignment loss between bounding boxes.
  """

    a = layout
    b = layout
    a, b = a[None, :, None], b[:, None, None]
    cartesian_product = np.concatenate(
        [a + np.zeros_like(b), np.zeros_like(a) + b], axis=2)

    left_correlation = left_similarity(cartesian_product)
    center_correlation = center_similarity(cartesian_product)
    right_correlation = right_similarity(cartesian_product)
    correlations = np.stack(
        [left_correlation, center_correlation, right_correlation], axis=2)
    min_correlation = np.sum(np.min(correlations, axis=(1, 2)))
    return min_correlation


def left_similarity(correlated):
    """Calculates left alignment loss of bounding boxes.

  Args:
    correlated: [assets_num, assets_num, 2, 4]. Combinations of all pairs of
      assets so we can calculate the similarity between these bounding boxes
      in parallel.
  Returns:
    Left alignment similarities between all pairs of assets in the layout.
  """

    remove_diagonal_entries_mask = np.zeros(
        (correlated.shape[0], correlated.shape[0]))
    np.fill_diagonal(remove_diagonal_entries_mask, np.inf)
    correlations = np.min(
        np.abs(correlated[:, :, 0, :2] - correlated[:, :, 1, :2]), axis=-1)
    return correlations + remove_diagonal_entries_mask


def right_similarity(correlated):
    """Calculates right alignment loss of bounding boxes."""

    correlations = np.min(
        np.abs(correlated[:, :, 0, 2:] - correlated[:, :, 1, 2:]), axis=-1)
    remove_diagonal_entries_mask = np.zeros(
        (correlated.shape[0], correlated.shape[0]))
    np.fill_diagonal(remove_diagonal_entries_mask, np.inf)
    return correlations + remove_diagonal_entries_mask


def center_similarity(correlated):
    """Calculates center alignment loss of bounding boxes."""

    x0 = (correlated[:, :, 0, 0] + correlated[:, :, 0, 2]) / 2
    y0 = (correlated[:, :, 0, 1] + correlated[:, :, 0, 3]) / 2

    centroids0 = np.stack([x0, y0], axis=2)

    x1 = (correlated[:, :, 1, 0] + correlated[:, :, 1, 2]) / 2
    y1 = (correlated[:, :, 1, 1] + correlated[:, :, 1, 3]) / 2
    centroids1 = np.stack([x1, y1], axis=2)

    correlations = np.min(np.abs(centroids0 - centroids1), axis=-1)
    remove_diagonal_entries_mask = np.zeros(
        (correlated.shape[0], correlated.shape[0]))
    np.fill_diagonal(remove_diagonal_entries_mask, np.inf)

    return correlations + remove_diagonal_entries_mask
