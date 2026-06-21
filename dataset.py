import json
import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageEnhance
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
        indices: Optional[Sequence[int]] = None,
        augment: bool = False,
        hflip_prob: float = 0.5,
        color_jitter: float = 0.2,
        scale_jitter: float = 0.2,
    ):
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter
        self.scale_jitter = scale_jitter

        with self.annotation_file.open("r", encoding="utf-8") as handle:
            self.samples = json.load(handle)

        if not isinstance(self.samples, list):
            raise ValueError("annotation_file must contain a list of samples")
        if indices is not None:
            self.samples = [self.samples[index] for index in indices]

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
        boxes = torch.tensor(sample.get("boxes", []), dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(sample.get("labels", []), dtype=torch.long).reshape(-1)

        if self.augment:
            image, boxes, labels = self._apply_train_augmentations(image, boxes, labels)

        # 2. Tính toán tỷ lệ Letterbox (giữ nguyên aspect ratio)
        aug_w, aug_h = image.size
        scale = min(self.image_size / aug_w, self.image_size / aug_h)
        new_w = int(aug_w * scale)
        new_h = int(aug_h * scale)
        
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
        if boxes.numel() > 0:
            # Tọa độ mới = (Tọa độ cũ * scale) + khoảng pad tương ứng
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y
            
            # Gờ giới hạn tránh box vượt biên
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, self.image_size)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, self.image_size)

        return {
            "image": image_tensor,
            "boxes": boxes,
            "labels": labels,
            "path": str(image_path),
            "original_size": torch.tensor([orig_w, orig_h], dtype=torch.float32),
        }

    def _apply_train_augmentations(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        width, height = image.size

        if boxes.numel() > 0 and random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = width - x2
            boxes[:, 2] = width - x1

        if self.scale_jitter > 0:
            scale = random.uniform(1.0 - self.scale_jitter, 1.0 + self.scale_jitter)
            image, boxes, labels = self._scale_jitter_image(image, boxes, labels, scale)

        if self.color_jitter > 0:
            image = self._color_jitter_image(image)

        return image, boxes, labels

    def _scale_jitter_image(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scale: float,
    ) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        width, height = image.size
        scaled_w = max(1, int(width * scale))
        scaled_h = max(1, int(height * scale))
        image = image.resize((scaled_w, scaled_h), Image.BILINEAR)

        if boxes.numel() > 0:
            boxes = boxes * scale

        if scale >= 1.0:
            max_left = max(0, scaled_w - width)
            max_top = max(0, scaled_h - height)
            left = random.randint(0, max_left) if max_left > 0 else 0
            top = random.randint(0, max_top) if max_top > 0 else 0
            image = image.crop((left, top, left + width, top + height))
            if boxes.numel() > 0:
                boxes[:, [0, 2]] -= left
                boxes[:, [1, 3]] -= top
        else:
            canvas = Image.new("RGB", (width, height), (128, 128, 128))
            max_left = width - scaled_w
            max_top = height - scaled_h
            left = random.randint(0, max_left) if max_left > 0 else 0
            top = random.randint(0, max_top) if max_top > 0 else 0
            canvas.paste(image, (left, top))
            image = canvas
            if boxes.numel() > 0:
                boxes[:, [0, 2]] += left
                boxes[:, [1, 3]] += top

        if boxes.numel() > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            keep = (box_w > 2.0) & (box_h > 2.0)
            boxes = boxes[keep]
            labels = labels[keep]

        return image, boxes, labels

    def _color_jitter_image(self, image: Image.Image) -> Image.Image:
        min_factor = 1.0 - self.color_jitter
        max_factor = 1.0 + self.color_jitter
        enhancers = (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color)
        for enhancer_class in enhancers:
            factor = random.uniform(min_factor, max_factor)
            image = enhancer_class(image).enhance(factor)
        return image


def collate_traffic_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    return {
        "images": images,
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "paths": [item["path"] for item in batch],
        "original_sizes": torch.stack([item["original_size"] for item in batch], dim=0),
    }
