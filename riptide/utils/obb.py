
import torch
from shapely.geometry import Polygon
import shapely

def obb_iou(pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor) -> torch.Tensor:
    """Compute IoU matrix between 2 sets of oriented bounding boxes (oOBB).
    Format of OBB is [x1, y1, x2, y2, x3, y3, x4, y4].
    
    Args:
        pred_bboxes: Tensor of shape (N, 8)
        gt_bboxes: Tensor of shape (M, 8)

    Returns:
        Tensor of shape (N, M)
    """
    pred_polys: list[Polygon] = []
    gt_polys: list[Polygon] = []
    for pred in pred_bboxes:
        coords = [(pred[idx].item(), pred[idx+1].item()) for idx in range(0, 8 , 2)]
        pred_poly = Polygon(coords)
        shapely.prepare(pred_poly)
        pred_polys.append(pred_poly)
    for gt in gt_bboxes:
        coords = [(gt[idx].item(), gt[idx+1].item()) for idx in range(0, 8 , 2)]
        gt_poly = Polygon(coords)
        shapely.prepare(gt_poly)
        gt_polys.append(gt_poly)
    
    ious = torch.zeros(pred_bboxes.size(0), gt_bboxes.size(0))
    for i, pred_poly in enumerate(pred_polys):
        for j, gt_poly in enumerate(gt_polys):
            union = pred_poly.union(gt_poly)
            intersection = pred_poly.intersection(gt_poly)
            iou = intersection.area / union.area
            ious[i, j] = iou
    return ious