"""SCRFD-style license plate detector.

This package is our clean PyTorch rewrite.  It uses the SCRFD pattern from the
original repository:

    image -> MobileNet backbone -> LFPN neck -> dense class/box head

The face landmark/keypoint branch is intentionally not included because license
plate detection only needs objectness and bounding boxes.
"""

from .main import (
    DetectionHead,
    LFPN,
    LicensePlateDetector,
    MobileNetV1,
    SCRFDPlateDetector,
    build_license_plate_model,
    build_scrfd_plate_model,
)
# from .process import (
#     detect_license_plates,
#     draw_detections,
#     load_model,
#     preprocess_image,
#     scale_boxes_to_original,
# )
from .train import compute_detection_loss

__all__ = [
    "DetectionHead",
    "LFPN",
    "LicensePlateDetector",
    "MobileNetV1",
    "SCRFDPlateDetector",
    "build_license_plate_model",
    "build_scrfd_plate_model",
    # "detect_license_plates",      # Khóa lại trong __all__
    # "draw_detections",
    # "load_model",
    # "preprocess_image",
    # "scale_boxes_to_original",
    "compute_detection_loss",
]