from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

try:
    from .main import distance_to_bbox, make_grid_points
except ImportError:
    from main import distance_to_bbox, make_grid_points

import torch
import torch.nn.functional as F

DEFAULT_SIZE_RANGES: Dict[int, Tuple[float, float]] = {
    8: (0.0, 64.0),
    16: (64.0, 128.0),
    32: (128.0, float("inf")),
}

@dataclass
class Assignment:
    points: torch.Tensor
    strides: torch.Tensor
    labels: torch.Tensor
    box_targets: torch.Tensor
    positive_mask: torch.Tensor


def flatten_outputs(
    cls_scores: Sequence[torch.Tensor],
    box_preds: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten multi-level dense outputs to [B, N * num_classes] and [B, N, 4]."""
    flat_scores = []
    flat_boxes = []

    for cls_score, box_pred in zip(cls_scores, box_preds):
        # Biến đổi cls_score về dạng [B, H * W * num_classes] để tính Focal Loss đa nhãn
        flat_scores.append(
            cls_score.permute(0, 2, 3, 1).reshape(cls_score.shape[0], -1)
        )
        flat_boxes.append(
            box_pred.permute(0, 2, 3, 1).reshape(box_pred.shape[0], -1, 4)
        )

    return torch.cat(flat_scores, dim=1), torch.cat(flat_boxes, dim=1)


def build_points(
    cls_scores: Sequence[torch.Tensor],
    strides: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build all feature-map center points and their stride values."""
    device = cls_scores[0].device
    all_points = []
    all_strides = []

    for cls_score, stride in zip(cls_scores, strides):
        _, _, height, width = cls_score.shape
        points = make_grid_points(height, width, stride, device)
        all_points.append(points)
        all_strides.append(
            torch.full((points.shape[0],), stride, dtype=torch.float32, device=device)
        )

    return torch.cat(all_points, dim=0), torch.cat(all_strides, dim=0)


def assign_single_image(
    points: torch.Tensor,
    strides: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,       # Thêm tham số nhận nhãn thực tế của từng box
    num_classes: int = 7,          # Thêm số lượng class mặc định là 7
    center_radius: float = 2.5,
    size_ranges: Optional[Mapping[int, Tuple[float, float]]] = None,
) -> Assignment:
    num_points = points.shape[0]

    # Khởi tạo ma trận nhãn mục tiêu dạng One-hot [num_points, num_classes] điền sẵn toàn bộ số 0
    labels = points.new_zeros((num_points, num_classes))
    box_targets = points.new_zeros((num_points, 4))
    positive_mask = torch.zeros((num_points,), dtype=torch.bool, device=points.device)

    if gt_boxes.numel() == 0:
        return Assignment(
            points=points,
            strides=strides,
            labels=labels.reshape(-1),  # Làm phẳng đồng bộ ra kích thước [num_points * num_classes]
            box_targets=box_targets,
            positive_mask=positive_mask,
        )
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] != 4:
        raise ValueError(f"gt_boxes must have shape [N, 4], got {tuple(gt_boxes.shape)}")
    if gt_labels.ndim != 1 or gt_labels.shape[0] != gt_boxes.shape[0]:
        raise ValueError(
            f"gt_labels must have shape [N] matching gt_boxes, got "
            f"{tuple(gt_labels.shape)} labels for {tuple(gt_boxes.shape)} boxes"
        )

    # 1. Xác định xem điểm neo (anchor point) có nằm bên trong gt_box hay không
    x_centers = points[:, 0]
    y_centers = points[:, 1]

    lt_distances = x_centers[:, None] - gt_boxes[:, 0]
    top_distances = y_centers[:, None] - gt_boxes[:, 1]
    rb_distances = gt_boxes[:, 2] - x_centers[:, None]
    bottom_distances = gt_boxes[:, 3] - y_centers[:, None]

    delta_boxes = torch.stack(
        (lt_distances, top_distances, rb_distances, bottom_distances), dim=-1
    )
    is_inside_boxes = delta_boxes.min(dim=-1).values > 0.0

    # 2. Cơ chế Center Sampling: Điểm neo có nằm sát tâm vật thể (bán kính center_radius) không
    gt_centers_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
    gt_centers_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5

    center_lt_x = x_centers[:, None] - (gt_centers_x - center_radius * strides[:, None])
    center_lt_y = y_centers[:, None] - (gt_centers_y - center_radius * strides[:, None])
    center_rb_x = (gt_centers_x + center_radius * strides[:, None]) - x_centers[:, None]
    center_rb_y = (gt_centers_y + center_radius * strides[:, None]) - y_centers[:, None]

    delta_centers = torch.stack(
        (center_lt_x, center_lt_y, center_rb_x, center_rb_y), dim=-1
    )
    is_inside_centers = delta_centers.min(dim=-1).values > 0.0

    # 3. Chỉ gán GT cho level phù hợp với kích thước object, tránh object nhỏ
    # học ở stride quá thô và object lớn học ở stride quá mịn.
    if size_ranges is None:
        size_ranges = DEFAULT_SIZE_RANGES

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_sizes = torch.maximum(gt_widths, gt_heights)

    point_min_sizes = torch.empty_like(strides)
    point_max_sizes = torch.empty_like(strides)
    for stride_value in strides.unique():
        stride_key = int(stride_value.item())
        if stride_key not in size_ranges:
            raise ValueError(f"Missing size range for stride {stride_key}")
        min_size, max_size = size_ranges[stride_key]
        stride_mask = strides == stride_value
        point_min_sizes[stride_mask] = min_size
        point_max_sizes[stride_mask] = max_size

    is_inside_size_range = (
        (gt_sizes[None, :] >= point_min_sizes[:, None])
        & (gt_sizes[None, :] < point_max_sizes[:, None])
    )

    # Kết hợp cả 3 điều kiện: nằm trong box, gần tâm, và đúng level kích thước
    is_candidate = is_inside_boxes & is_inside_centers & is_inside_size_range

    # 4. Giải quyết chồng lấn (Nếu điểm neo trúng nhiều vật thể, chọn vật thể có diện tích nhỏ nhất)
    gt_area = box_area(gt_boxes)
    gt_area = gt_area.repeat(num_points, 1)
    gt_area[~is_candidate] = float("inf")

    min_area, matched_gt = gt_area.min(dim=1)
    positive_mask = torch.isfinite(min_area)

    if positive_mask.any():
        matched_gts = matched_gt[positive_mask]
        matched_classes = gt_labels[matched_gts].long()  # Lấy id lớp thực tế của box trúng tuyển
        
        # Điền giá trị 1.0 vào vị trí One-hot tương ứng của lớp đó
        labels[positive_mask, matched_classes] = 1.0
        box_targets[positive_mask] = gt_boxes[matched_gts]

    return Assignment(
        points=points,
        strides=strides,
        labels=labels.reshape(-1),  # Làm phẳng [num_points * num_classes] để khớp dạng của flat_cls
        box_targets=box_targets,
        positive_mask=positive_mask,
    )


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    width = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    height = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return width * height


def decode_flat_boxes(
    points: torch.Tensor,
    strides: torch.Tensor,
    flat_box_preds: torch.Tensor,
) -> torch.Tensor:
    """Decode box distances to xyxy boxes.

    flat_box_preds: [B, N, 4]
    points: [N, 2]
    strides: [N]
    """
    distances = F.softplus(flat_box_preds) * strides[None, :, None]

    decoded = []
    for image_distances in distances:
        decoded.append(distance_to_bbox(points, image_distances))

    return torch.stack(decoded, dim=0)


def build_batch_assignments(
    cls_scores: Sequence[torch.Tensor],
    gt_boxes: Sequence[torch.Tensor],
    gt_labels: Sequence[torch.Tensor],  # Bổ sung danh sách nhãn thực tế của batch
    strides: Sequence[int],
    num_classes: int = 7,                 # Thêm cấu hình num_classes
    center_radius: float = 2.5,
    size_ranges: Optional[Mapping[int, Tuple[float, float]]] = None,
) -> List[Assignment]:
    """Build anchor assignments across a complete batch."""
    points, point_strides = build_points(cls_scores, strides)

    assignments = [
        assign_single_image(
            points,
            point_strides,
            boxes,
            lbls,                       # Truyền mảng nhãn của từng ảnh vào
            num_classes=num_classes,
            center_radius=center_radius,
            size_ranges=size_ranges,
        )
        for boxes, lbls in zip(gt_boxes, gt_labels)
    ]

    return assignments
