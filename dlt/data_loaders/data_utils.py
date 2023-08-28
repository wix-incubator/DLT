import numpy as np


def norm_bbox(H, W, element):
    x1, y1, width, height = element['bbox']
    xc = x1 + width / 2.
    yc = y1 + height / 2.
    b = [xc / W, yc / H,
         width / W, height / H]
    return b


def is_valid_comp(comp, W, H):
    x1, y1, width, height = comp['bbox']
    x2, y2 = x1 + width, y1 + height
    if x1 < 0 or y1 < 0 or W < x2 or H < y2:
        return False

    if x2 <= x1 or y2 <= y1:
        return False

    return True


def mask_loc(bbox_shape, r_mask=1.0):
    n, _ = bbox_shape
    ind_mask = np.random.choice(range(n), int(n * r_mask), replace=False)
    mask = np.zeros(bbox_shape)
    mask[ind_mask, :2] = 1
    full_mask_cat = np.zeros(n).astype('long')
    return mask, full_mask_cat


def mask_size(bbox_shape, r_mask=1.0):
    n, _ = bbox_shape
    ind_mask = np.random.choice(range(n), int(n * r_mask), replace=False)
    mask = np.zeros(bbox_shape)
    mask[ind_mask, 2:] = 1
    full_mask_cat = np.zeros(n).astype('long')
    return mask, full_mask_cat


def mask_whole_box(bbox_shape, r_mask=1.0):
    n, _ = bbox_shape
    ind_mask = np.random.choice(range(n), int(n * r_mask), replace=False)
    mask = np.zeros(bbox_shape)
    mask[ind_mask, :4] = 1
    full_mask_cat = np.zeros(n).astype('long')
    return mask, full_mask_cat


def mask_all(bbox_shape):
    n, _ = bbox_shape
    mask = np.ones(bbox_shape)
    full_mask_cat = np.ones(n).astype('long')
    return mask, full_mask_cat


def mask_cat(bbox_shape, r_mask=1.0):
    n, dim = bbox_shape
    ind_mask = np.random.choice(range(n), int(n * r_mask), replace=False)
    mask = np.zeros(bbox_shape)
    full_mask_cat = np.zeros(n).astype('long')
    full_mask_cat[ind_mask] = 1
    return mask, full_mask_cat


def mask_random_box_and_cat(bbox_shape, r_mask_box=1.0, r_mask_cat=1.0):
    n, _ = bbox_shape
    func_options = [mask_loc, mask_size, [mask_loc, mask_size], mask_whole_box]
    ix = np.random.choice(range(len(func_options)), 1)[0]
    func_mask_box = func_options[ix]
    if isinstance(func_mask_box, list):
        mask_box = np.zeros(bbox_shape)
        for func in func_mask_box:
            m, _ = func(bbox_shape, r_mask_box)
            mask_box += m
        all_cat_mask = np.zeros(n).astype('long')
    else:
        mask_box, all_cat_mask = func_mask_box(bbox_shape, r_mask_box)
    _, full_mask_cat = mask_cat(bbox_shape, r_mask_cat)
    cat_mask_options = [all_cat_mask, full_mask_cat]
    return mask_box, cat_mask_options[np.random.choice(len(cat_mask_options), 1)[0]]
