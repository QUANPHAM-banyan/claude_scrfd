import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    """Simple multi-class dataset for traffic detection.

    Expected JSON format:

    [
      {
        "image": "images/traffic_001.jpg",
        "boxes": [[x1, y1, x2, y2], [x1, y1, x2, y2]],
        "labels": [0, 4]  # 0: car, 1: bike, 2: bus, 3: truck, 4: plate, 5: helmet, 6: nohelmet
      }
    ]

    Image paths can be absolute or relative to ``image_root``.
    """

    def __init__(
        self,
        annotation_file: str,
        image_root: str = ".",
        image_size: int = 640,
    ):
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)
        self.image_size = image_size

        with self.annotation_file.open("r", encoding="utf-8") as handle:
            self.samples = json.load(handle)

        if not isinstance(self.samples, list):
            raise ValueError("annotation_file must contain a list of samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        image_path = Path(sample["image"])
        if not image_path.is_absolute():
            image_path = self.image_root / image_path

        # 1. Đọc ảnh gốc
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        # 2. Tính toán tỷ lệ Letterbox (giữ nguyên aspect ratio)
        scale = min(self.image_size / orig_w, self.image_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize ảnh theo tỷ lệ chuẩn
        resized_img = image.resize((new_w, new_h), Image.BILINEAR)

        # Tạo một canvas trống hình vuông kích thước target_size 
        new_image = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        
        # Tính toán khoảng đệm (padding) để đưa ảnh vào giữa canvas
        pad_x = (self.image_size - new_w) // 2
        pad_y = (self.image_size - new_h) // 2
        new_image.paste(resized_img, (pad_x, pad_y))

        # Chuyển đổi thành Tensor
        data = torch.frombuffer(new_image.tobytes(), dtype=torch.uint8).clone()
        image_tensor = data.reshape(self.image_size, self.image_size, 3).permute(2, 0, 1).float()
        image_tensor = image_tensor / 255.0

        # 3. Cập nhật lại tọa độ Bounding Box theo Letterbox
        boxes = torch.tensor(sample.get("boxes", []), dtype=torch.float32).reshape(-1, 4)
        if boxes.numel() > 0:
            # Tọa độ mới = (Tọa độ cũ * scale) + khoảng pad tương ứng
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y
            
            # Gờ giới hạn tránh box vượt biên
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, self.image_size)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, self.image_size)

        # 4. Đọc nhãn lớp (labels) tương ứng với từng box
        labels = torch.tensor(sample.get("labels", []), dtype=torch.long).reshape(-1)

        return {
            "image": image_tensor,
            "boxes": boxes,
            "labels": labels,
            "path": str(image_path),
            "original_size": torch.tensor([orig_w, orig_h], dtype=torch.float32),
        }


def collate_traffic_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    return {
        "images": images,
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "paths": [item["path"] for item in batch],
        "original_sizes": torch.stack([item["original_size"] for item in batch], dim=0),
    }
