import torch
import numpy as np
import pickle
from path import Path
from evaluation.metrics import get_layout_iou, get_alignment_loss_numpy, compute_overlap, compute_alignment, LayoutFID
from evaluation.utils import convert_xywh_to_ltrb_batch


def compute_benchmarks(gt_layouts, predicted_layouts, dataset_name='publaynet', device='cpu'):
    """
    Computes the benchmarks for the given layouts
    # shape (num_layouts/batch_size, max_num_elements, 5)
    # 5: xc, yc, w, h, class
    xc, yc: center coordinates
    w, h: width and height
    values in range [0, 1]
    :param gt_layouts: the ground truth layouts
    :param predicted_layouts: the predicted layouts
    :param dataset_name: the name of the dataset for fid model initialization
    :param device: the device to use for fid calculation
    :return: the benchmarks
    """
    assert dataset_name in ("publaynet", "rico", "magazine")
    fid_model = LayoutFID(dataset_name, device=device)
    gt_categories = gt_layouts[:, :, 4]
    predicted_categories = predicted_layouts[:, :, 4]
    gt_layouts_boxes = gt_layouts[:, :, :4]
    predicted_layouts_boxes = predicted_layouts[:, :, :4]
    mask = predicted_categories > 0
    padding_mask = ~mask

    # compute iou for all layouts
    piou_gt = np.array([get_layout_iou(layout, cur_mask) for layout, cur_mask in zip(gt_layouts_boxes, mask)])
    piou_pred = np.array(
        [get_layout_iou(layout, cur_mask) for layout, cur_mask in zip(predicted_layouts_boxes, mask)])

    #  compute the alignment loss
    alignment_gt = np.array([get_alignment_loss_numpy(convert_xywh_to_ltrb_batch(layout[cur_mask])) for
                             layout, cur_mask in zip(gt_layouts_boxes, mask)])
    alignment_pred = np.array([get_alignment_loss_numpy(convert_xywh_to_ltrb_batch(layout[cur_mask])) for
                               layout, cur_mask in zip(predicted_layouts_boxes, mask)])

    # compute the fid score
    fid_cat_gt = torch.tensor(gt_categories, dtype=torch.long)
    fid_cat_gt = torch.clip(fid_cat_gt - 1, 0, fid_model.num_label)
    fid_cat_pred = torch.tensor(predicted_categories, dtype=torch.long)
    fid_cat_pred = torch.clip(fid_cat_pred - 1, 0, fid_model.num_label)

    fid_model.collect_features(torch.tensor(gt_layouts_boxes,
                                            dtype=torch.float32), fid_cat_gt, torch.tensor(padding_mask),
                               real=True)
    fid_model.collect_features(torch.tensor(predicted_layouts_boxes, dtype=torch.float32),
                               fid_cat_pred,
                               torch.tensor(padding_mask))
    fid_score = fid_model.compute_score()

    # compute overlap
    gt_overlap = compute_overlap(torch.tensor(gt_layouts_boxes), torch.tensor(mask)).numpy()
    pred_overlap = compute_overlap(torch.tensor(predicted_layouts_boxes), torch.tensor(mask)).numpy()

    # compute alignment2
    alignment2_gt = compute_alignment(torch.tensor(gt_layouts_boxes), torch.tensor(mask)).numpy()
    alignment2_pred = compute_alignment(torch.tensor(predicted_layouts_boxes), torch.tensor(mask)).numpy()

    pr_print = lambda x: round(x * 100, 5)
    print('Real data:')
    print('piou_gt: ', pr_print(piou_gt.mean()))
    print('alignment_gt: ', alignment_gt.mean())
    print('overlap_gt: ', pr_print(gt_overlap.mean()))
    print('alignment2_gt: ', pr_print(alignment2_gt.mean()))
    print('_____________________________')
    print('Predicted data:')
    print('FID: ', fid_score)
    print('piou_pred: ', pr_print(piou_pred.mean()))
    print('alignment_pred: ', alignment_pred.mean())
    print('overlap_pred: ', pr_print(pred_overlap.mean()))
    print('alignment2_pred: ', pr_print(alignment2_pred.mean()))
    return {'iou_gt': piou_gt.mean(), 'iou_pred': piou_pred.mean(), 'alignment_gt': alignment_gt.mean(),
            'alignment_pred': alignment_pred.mean(), 'fid': fid_score, 'overlap_gt': gt_overlap.mean(),
            'overlap_pred': pred_overlap.mean(), 'alignment2_gt': alignment2_gt.mean(),
            'alignment2_pred': alignment2_pred.mean()}


if __name__ == '__main__':
    res_path = Path('/Users/mykolam/tmp/dlt_playground/logs/magazine_final/samples/results_all.pkl')
    with open(res_path, 'rb') as f:
        res = pickle.load(f)
    res['dataset_val'] = np.concatenate(res['dataset_val'], axis=0)
    res['predicted_val'] = np.concatenate(res['predicted_val'], axis=0)
    rev_scale = lambda x: ((x / 2) + 1) / 2
    res['dataset_val'][:, :, :4] = rev_scale(res['dataset_val'][:, :, :4])
    res['predicted_val'][:, :, :4] = rev_scale(res['predicted_val'][:, :, :4])
    compute_benchmarks(res['dataset_val'], res['predicted_val'])
