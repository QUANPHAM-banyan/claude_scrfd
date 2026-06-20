"""Pure PyTorch SCRFD-style detector for traffic and license plates (7 classes).

Reference pieces from the original repository:
- ``backbones/mobilenet.py``: compact MobileNetV1 feature extractor.
- ``necks/lfpn.py``: lightweight top-down feature pyramid.
- ``dense_heads/scrfd_head.py``: dense SCRFD head idea.

This rewrite supports multi-class classification and multi-class NMS.
"""

import argparse
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Đảm bảo import hàm nms của torchvision cho quá trình hậu xử lý
try:
    from torchvision.ops import nms
except ImportError:
    raise ImportError("Please install torchvision to use the nms operator.")


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, groups: int = 1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            ConvBNReLU(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )


class MobileNetV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage0 = nn.Sequential(
            ConvBNReLU(3, 16, kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableConv(16, 32, stride=1),
        )
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 64, stride=1),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stage0(x)
        c1 = self.stage1(c1)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        return c1, c2, c3


class LFPN(nn.Module):
    def __init__(self, in_channels_list: Sequence[int], out_channels: int) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        for i in range(len(laterals) - 1, 0, -1):
            size = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=size, mode="nearest")
        outputs = [conv(x) for conv, x in zip(self.fpn_convs, laterals)]
        return outputs


class DetectionHead(nn.Module):
    def __init__(self, num_classes: int = 7, in_channels: int = 32, anchors_per_level: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.anchors_per_level = anchors_per_level
        self.strides = [8, 16, 32]

        self.cls_convs = nn.ModuleList([
            nn.Conv2d(in_channels, num_classes * anchors_per_level, kernel_size=1)
            for _ in self.strides
        ])
        self.box_convs = nn.ModuleList([
            nn.Conv2d(in_channels, 4 * anchors_per_level, kernel_size=1)
            for _ in self.strides
        ])

    def forward(self, features: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_scores = [conv(x) for conv, x in zip(self.cls_convs, features)]
        box_preds = [conv(x) for conv, x in zip(self.box_convs, features)]
        return cls_scores, box_preds


class SCRFDPlateDetector(nn.Module):
    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.backbone = MobileNetV1()
        self.neck = LFPN(in_channels_list=[64, 128, 256], out_channels=32)
        self.head = DetectionHead(num_classes=num_classes, in_channels=32, anchors_per_level=1)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        features = self.backbone(x)
        features = self.neck(features)
        cls_scores, box_preds = self.head(features)
        return cls_scores, box_preds

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        score_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        max_detections: int = 100,
    ) -> List[torch.Tensor]:
        self.eval()
        cls_scores, box_preds = self(images)
        batch_size = images.shape[0]
        results = []

        for batch_idx in range(batch_size):
            image_boxes = []
            image_scores = []
            image_labels = []

            for cls_score, box_pred, stride in zip(cls_scores, box_preds, self.head.strides):
                _, _, height, width = cls_score.shape
                points = make_grid_points(height, width, stride, images.device)

                # Biến đổi scores về dạng: [H * W, num_classes]
                scores = cls_score[batch_idx].sigmoid().permute(1, 2, 0).reshape(-1, self.head.num_classes)
                # Biến đổi khoảng cách box về dạng: [H * W, 4]
                distances = box_pred[batch_idx].permute(1, 2, 0).reshape(-1, 4) * stride

                # Duyệt qua từng class để lọc ngưỡng score
                for class_idx in range(self.head.num_classes):
                    cls_scores_per_class = scores[:, class_idx]
                    keep = cls_scores_per_class >= score_threshold
                    if keep.any():
                        image_scores.append(cls_scores_per_class[keep])
                        image_boxes.append(distance_to_bbox(points[keep], distances[keep]))
                        image_labels.append(torch.full_like(cls_scores_per_class[keep], class_idx, dtype=torch.long))

            if not image_boxes:
                # Trả về tensor rỗng kích thước [0, 6] (x1, y1, x2, y2, score, class_id)
                results.append(images.new_zeros((0, 6)))
                continue

            boxes = torch.cat(image_boxes, dim=0)
            scores = torch.cat(image_scores, dim=0)
            labels = torch.cat(image_labels, dim=0)

            # Thực hiện Multi-class NMS bằng kỹ thuật dịch tọa độ (offset) theo class_id
            max_coordinate = boxes.max()
            offsets = labels * (max_coordinate + 1)
            boxes_for_nms = boxes + offsets[:, None]

            keep = nms(boxes_for_nms, scores, nms_threshold)[:max_detections]
            
            # Kết quả trả về gồm 6 cột: x1, y1, x2, y2, score, class_id
            results.append(torch.cat((boxes[keep], scores[keep, None], labels[keep, None].float()), dim=1))

        return results


LicensePlateDetector = SCRFDPlateDetector


def build_scrfd_plate_model() -> SCRFDPlateDetector:
    # Thay đổi mặc định khởi tạo từ 1 lên 7 classes
    return SCRFDPlateDetector(num_classes=7)


def build_license_plate_model() -> SCRFDPlateDetector:
    return build_scrfd_plate_model()


def make_grid_points(height: int, width: int, stride: int, device: torch.device) -> torch.Tensor:
    shift_x = torch.arange(0, width, device=device) * stride + stride // 2
    shift_y = torch.arange(0, height, device=device) * stride + stride // 2
    y, x = torch.meshgrid(shift_y, shift_x, indexing="ij")
    return torch.stack((x.reshape(-1), y.reshape(-1)), dim=1).float()


def distance_to_bbox(points: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return torch.stack((x1, y1, x2, y2), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    model = build_scrfd_plate_model()
    dummy = torch.randn(1, 3, args.height, args.width)
    cls_scores, box_preds = model(dummy)
    detections = model.predict(dummy, score_threshold=0.01)

    print("classification outputs:", [tuple(x.shape) for x in cls_scores])
    print("regression outputs:", [tuple(x.shape) for x in box_preds])
    print("detections shape (batch 0):", detections[0].shape)


if __name__ == "__main__":
    main()