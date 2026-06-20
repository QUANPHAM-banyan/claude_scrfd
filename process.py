import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Sequence
import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    from .main import build_scrfd_plate_model
except ImportError:
    from main import build_scrfd_plate_model


IMAGE_SIZE = 640

# Định nghĩa danh sách 7 nhãn đồng bộ theo thứ tự trong Dataset của bạn
CLASS_NAMES = ["car", "bike", "bus", "truck", "plate", "helmet", "nohelmet"]


def load_model(
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
) -> torch.nn.Module:
    """Create the multi-class detector and optionally load trained weights."""
    runtime_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_scrfd_plate_model().to(runtime_device)

    if checkpoint:
        state = torch.load(checkpoint, map_location=runtime_device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    model.eval()
    return model


def preprocess_image(
    image_path: str,
    image_size: int = IMAGE_SIZE,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Image.Image, Tuple[float, float, float], Tuple[int, int]]:
    """Load an image and convert it to a model input tensor using Letterbox."""
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size

    # Tính toán tỷ lệ Letterbox (giữ nguyên aspect ratio)
    scale = min(image_size / original_width, image_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = image.resize((new_width, new_height), Image.BILINEAR)

    # Tạo canvas vuông màu xám nền làm đệm
    padded_image = Image.new("RGB", (image_size, image_size), (128, 128, 128))
    pad_x = (image_size - new_width) // 2
    pad_y = (image_size - new_height) // 2
    padded_image.paste(resized_image, (pad_x, pad_y))

    # Đưa về Tensor định dạng chuẩn [1, 3, H, W]
    image_data = np.array(padded_image, dtype=np.uint8)
    tensor = torch.from_numpy(image_data).permute(2, 0, 1).float()
    tensor = tensor / 255.0
    tensor = tensor.unsqueeze(0)  # Thêm chiều batch_size = 1

    if device is not None:
        tensor = tensor.to(device)

    meta_scale = (scale, float(pad_x), float(pad_y))
    meta_orig_size = (original_width, original_height)

    return tensor, image, meta_scale, meta_orig_size


def scale_boxes_to_original(
    detections: torch.Tensor,
    scale: float,
    pad_x: float,
    pad_y: float,
    original_width: int,
    original_height: int,
) -> torch.Tensor:
    """Transform predicted bounding boxes from canvas coordinates back to original image size."""
    if detections.numel() == 0:
        return detections

    scaled = detections.clone()
    # Trừ đi khoảng đệm (padding) và chia lại cho scale gốc
    scaled[:, [0, 2]] = (scaled[:, [0, 2]] - pad_x) / scale
    scaled[:, [1, 3]] = (scaled[:, [1, 3]] - pad_y) / scale

    # Gờ giới hạn tọa độ nằm trong biên ảnh gốc
    scaled[:, [0, 2]] = scaled[:, [0, 2]].clamp(0, original_width)
    scaled[:, [1, 3]] = scaled[:, [1, 3]].clamp(0, original_height)

    return scaled


def detect_traffic_objects(
    image_path: str,
    checkpoint: Optional[str] = None,
    image_size: int = IMAGE_SIZE,
    score_threshold: float = 0.35,
    nms_threshold: float = 0.45,
    device: Optional[str] = None,
) -> Tuple[List[List[float]], Image.Image]:
    """Run end-to-end multi-class inference on a single image file."""
    model = load_model(checkpoint, device)
    runtime_device = next(model.parameters()).device

    tensor, original_image, meta_scale, meta_orig = preprocess_image(
        image_path,
        image_size=image_size,
        device=runtime_device
    )

    # Dự đoán từ mô hình (kết quả đã qua Multi-class NMS)
    detections_batch = model.predict(
        tensor,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold
    )
    detections = detections_batch[0]  # Lấy kết quả của ảnh đầu tiên (và duy nhất trong batch)

    scale, pad_x, pad_y = meta_scale
    orig_w, orig_h = meta_orig
    
    # Khôi phục tọa độ về kích thước ảnh ban đầu
    scaled_detections = scale_boxes_to_original(
        detections, scale, pad_x, pad_y, orig_w, orig_h
    )

    return scaled_detections.tolist(), original_image


def draw_detections(
    image: Image.Image,
    detections: Iterable[Sequence[float]],
    output_path: str,
) -> None:
    """Draw predicted multi-class bounding boxes and labels on the image and save it."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    for detection in detections:
        # Bóc tách 6 thuộc tính: x1, y1, x2, y2, score, class_id
        x1, y1, x2, y2, score, class_id = [float(value) for value in detection]
        
        # Lấy tên nhãn tương ứng dựa vào class_id
        class_idx = int(class_id)
        class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else "unknown"
        
        label_text = f"{class_name} {score:.2f}"
        
        # Vẽ khung hình chữ nhật và viết chữ nhãn lên ảnh
        draw.rectangle((x1, y1, x2, y2), outline="cyan", width=3)
        draw.text((x1, max(0.0, y1 - 14.0)), label_text, fill="cyan")

    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-class traffic object detection.")
    parser.add_argument("image", help="Path to an input image")
    parser.add_argument("--checkpoint", help="Path to trained model weights")
    parser.add_argument("--output", default="traffic_result.jpg", help="Path for drawn result")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="Model input size")
    parser.add_argument("--score", type=float, default=0.35, help="Score threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", help="Device, for example cpu or cuda")
    args = parser.parse_args()

    detections, image = detect_traffic_objects(
        args.image,
        checkpoint=args.checkpoint,
        image_size=args.size,
        score_threshold=args.score,
        nms_threshold=args.nms,
        device=args.device,
    )

    draw_detections(image, detections, args.output)
    print(f"Done! Found {len(detections)} objects. Result saved to {args.output}")


if __name__ == "__main__":
    main()