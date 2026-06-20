"""SCRFD-style traffic object detector.

This package is our clean PyTorch rewrite.  It uses the SCRFD pattern from the
original repository:

    image -> MobileNet backbone -> LFPN neck -> dense class/box head

The face landmark/keypoint branch is intentionally not included because license
plate detection only needs objectness and bounding boxes.
"""

from .main import (
    DetectionHead,
    LFPN,
    MobileNetV1,
    SCRFDTrafficDetector,
    build_scrfd_traffic_model,
)
# from .process import (
#     detect_traffic_objects,
#     draw_detections,
#     load_model,
#     preprocess_image,
#     scale_boxes_to_original,
# )
from .train import compute_detection_loss

__all__ = [
    "DetectionHead",
    "LFPN",
    "MobileNetV1",
    "SCRFDTrafficDetector",
    "build_scrfd_traffic_model",
    # "detect_traffic_objects",
    # "draw_detections",
    # "load_model",
    # "preprocess_image",
    # "scale_boxes_to_original",
    "compute_detection_loss",
]
