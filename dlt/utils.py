import colorsys
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def HSVToRGB(h, s, v):
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
    return int(255 * r), int(255 * g), int(255 * b)


def getDistinctColors(n):
    huePartition = 1.0 / (n + 1)
    return (HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n))


colors_f = getDistinctColors(25)
colors = []
for c in colors_f:
    colors.append(c)


def masked_acc(real, pred, mask):
    accuracies = torch.logical_and(real.eq(torch.argmax(pred, dim=-1)), mask)
    return accuracies.sum() / mask.sum()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_cross_entropy(a, b, mask):
    b_c = torch.nn.functional.one_hot(b, num_classes=a.shape[-1])
    a_c = F.log_softmax(a, dim=2)

    loss = (-a_c * b_c).sum(axis=2)
    non_zero_elements = sum_flat(mask)
    loss = sum_flat(loss * mask.float())
    loss = loss / (non_zero_elements + 0.0001)
    return loss


def masked_l2(a, b, mask):
    loss = F.mse_loss(a, b, reduction='none')
    loss = sum_flat(loss * mask.float())
    non_zero_elements = sum_flat(mask)

    mse_loss_val = (non_zero_elements > 0) * (loss / (non_zero_elements + 0.00000001))
    return mse_loss_val


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


def plot_sample(box, classes, z_indexes, color_mapping_dict, width=360, height=360):
    thickness = -1
    canvas = (255 * np.ones((height, width, 3))).astype('uint8')
    # sort boxes and classes by z_index
    if z_indexes:
        sort_ixs = np.argsort(z_indexes)
        box = box[sort_ixs]
        classes = classes[sort_ixs]

    for ii, (t, c) in enumerate(zip(box, classes)):
        xs = int((t[0] - t[2] / 2) * width)
        ys = int((t[1] - t[3] / 2) * height)
        xe = int((t[0] + t[2] / 2) * width)
        ye = int((t[1] + t[3] / 2) * height)
        canvas = cv2.rectangle(canvas, (xs, ys), (xe, ye), color=color_mapping_dict[c], thickness=thickness)
        cv2.rectangle(canvas, (xs, ys), (xe, ye), (255, 255, 255), 2)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas


def draw_layout_opacity(box, classes, z_indexes, color_mapping_dict, width=360, height=360, opacity=0.8):
    canvas = (255 * np.ones((height, width, 3))).astype('uint8')
    # sort boxes and classes by z_index
    if z_indexes:
        sort_ixs = np.argsort(z_indexes)
        box = box[sort_ixs]
        classes = classes[sort_ixs]

    for ii, (t, c) in enumerate(zip(box, classes)):

        xs = int((t[0] - t[2] / 2) * width)
        ys = int((t[1] - t[3] / 2) * height)
        xe = int((t[0] + t[2] / 2) * width)
        ye = int((t[1] + t[3] / 2) * height)

        overlay = canvas.copy()

        canvas = cv2.rectangle(canvas, (xs, ys), (xe, ye), color=color_mapping_dict[c], thickness=-1)
        canvas = cv2.addWeighted(overlay, opacity, canvas, 1 - opacity, 0)
        canvas = cv2.rectangle(canvas, (xs, ys), (xe, ye), color=color_mapping_dict[c], thickness=2)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas


