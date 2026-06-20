from typing import Dict

import torch
import torch.nn.functional as F



# ============================================================
# Focal Loss
# ============================================================

def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:

    prob = torch.sigmoid(logits)

    ce_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    p_t = (
        prob * targets
        +
        (1 - prob) * (1 - targets)
    )

    modulating = (
        1 - p_t
    ).pow(gamma)


    alpha_t = (
        alpha * targets
        +
        (1-alpha)*(1-targets)
    )


    loss = (
        alpha_t
        *
        modulating
        *
        ce_loss
    )


    if reduction == "mean":
        return loss.mean()

    if reduction == "sum":
        return loss.sum()

    if reduction == "none":
        return loss

    raise ValueError(
        f"Unsupported reduction: {reduction}"
    )



# ============================================================
# Box utils
# ============================================================

def box_area(
    boxes: torch.Tensor
):

    return (
        (boxes[:,2]-boxes[:,0]).clamp(min=0)
        *
        (boxes[:,3]-boxes[:,1]).clamp(min=0)
    )



# ============================================================
# CIoU Loss (Đã bổ sung tham số reduction)
# ============================================================

def ciou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    reduction: str = "sum"  # Bổ sung tham số nhận cấu hình từ hàm tính loss chính
):

    if pred_boxes.numel() == 0:
        return pred_boxes.new_zeros(())


    eps = 1e-7


    # IoU

    x1 = torch.maximum(
        pred_boxes[:,0],
        target_boxes[:,0]
    )

    y1 = torch.maximum(
        pred_boxes[:,1],
        target_boxes[:,1]
    )

    x2 = torch.minimum(
        pred_boxes[:,2],
        target_boxes[:,2]
    )

    y2 = torch.minimum(
        pred_boxes[:,3],
        target_boxes[:,3]
    )


    inter = (
        (x2-x1).clamp(min=0)
        *
        (y2-y1).clamp(min=0)
    )


    area_pred = box_area(pred_boxes)
    area_gt = box_area(target_boxes)


    union = (
        area_pred
        +
        area_gt
        -
        inter
    )


    iou = inter/(union+eps)



    # center distance

    pred_cx = (
        pred_boxes[:,0]
        +
        pred_boxes[:,2]
    )/2


    pred_cy = (
        pred_boxes[:,1]
        +
        pred_boxes[:,3]
    )/2


    gt_cx = (
        target_boxes[:,0]
        +
        target_boxes[:,2]
    )/2


    gt_cy = (
        target_boxes[:,1]
        +
        target_boxes[:,3]
    )/2



    rho2 = (
        (pred_cx-gt_cx)**2
        +
        (pred_cy-gt_cy)**2
    )



    # enclosing box

    cx1 = torch.minimum(
        pred_boxes[:,0],
        target_boxes[:,0]
    )

    cy1 = torch.minimum(
        pred_boxes[:,1],
        target_boxes[:,1]
    )

    cx2 = torch.maximum(
        pred_boxes[:,2],
        target_boxes[:,2]
    )

    cy2 = torch.maximum(
        pred_boxes[:,3],
        target_boxes[:,3]
    )


    c2 = (
        (cx2-cx1)**2
        +
        (cy2-cy1)**2
        +
        eps
    )



    # aspect ratio

    pw = (
        pred_boxes[:,2]
        -
        pred_boxes[:,0]
    )

    ph = (
        pred_boxes[:,3]
        -
        pred_boxes[:,1]
    )


    tw = (
        target_boxes[:,2]
        -
        target_boxes[:,0]
    )

    th = (
        target_boxes[:,3]
        -
        target_boxes[:,1]
    )


    v = (
        4/(torch.pi**2)
        *
        (
            torch.atan(tw/(th+eps))
            -
            torch.atan(pw/(ph+eps))
        )**2
    )


    with torch.no_grad():

        alpha = (
            v
            /
            (
                1-iou+v+eps
            )
        )


    ciou = (
        iou
        -
        rho2/c2
        -
        alpha*v
    )

    loss = 1.0 - ciou

    # Xử lý gom tụ theo reduction đầu vào
    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")



# ============================================================
# Total loss
# ============================================================

def reduce_loss_dict(
    losses: Dict[str, torch.Tensor]
):

    return sum(
        value
        for key,value in losses.items()
        if key.startswith("loss_")
    )