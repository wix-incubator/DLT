import numpy as np
import torch
import random
from enum import IntEnum
from itertools import product, combinations


def convert_xywh_to_ltrb_batch(bbox):
    """
    bbox: [N, 4], numpy array
    """
    xc, yc, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]
