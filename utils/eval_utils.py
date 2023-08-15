import torch
import numpy as np

from utils.box_utils import bbox_iou, xywh2xyxy


def trans_vg_eval_val(pred_boxes, gt_boxes):
    batch_size = pred_boxes.shape[0]
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu = torch.sum(iou >= 0.5) / float(batch_size)

    return iou, accu

def trans_vg_eval_test(pred_boxes, gt_boxes, sum=True):
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu = torch.sum(iou >= 0.5) if sum else iou >= 0.5
    return iou, accu

def eval_category(category_id_list, iou, accu):
    # category_id_list包含 [1,2,3...8]id子类，此处需要 -1 表示从0编码
    category_id_list = category_id_list.cpu().numpy()
    sub = list(set(category_id_list)).__len__()
    iou = iou.cpu().numpy()
    accu = accu.cpu().numpy()
    category_iou = [0] * sub
    category_accu = [0] * sub
    sub_num = [0] * sub
    for (id, iou_, accu_) in zip(category_id_list, iou, accu):
        category_iou[id-1] += iou_
        category_accu[id-1] += accu_
        sub_num[id-1] += 1
    category_iou = [i / s for i, s in zip(category_iou, sub_num)]
    category_accu = [a / s for a, s in zip(category_accu, sub_num)]
    return category_iou, category_accu


