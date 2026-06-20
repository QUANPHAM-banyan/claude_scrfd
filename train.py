import argparse
import os
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

try:
    from .assigner import build_batch_assignments, decode_flat_boxes, flatten_outputs
    from .dataset import PlateDataset, collate_plate_batch
    from .losses import ciou_loss, reduce_loss_dict, sigmoid_focal_loss
    from .main import build_scrfd_plate_model
except ImportError:
    from assigner import build_batch_assignments, decode_flat_boxes, flatten_outputs
    from dataset import PlateDataset, collate_plate_batch
    from losses import ciou_loss, reduce_loss_dict, sigmoid_focal_loss
    from main import build_scrfd_plate_model


def compute_detection_loss(
    cls_scores: Sequence[torch.Tensor],
    box_preds: Sequence[torch.Tensor],
    gt_boxes: Sequence[torch.Tensor],
    gt_labels: Sequence[torch.Tensor],
    strides: Sequence[int],
    num_classes: int = 7,
    center_radius: float = 2.5,
    cls_loss_weight: float = 1.0,
    box_loss_weight: float = 2.0,
) -> Dict[str, torch.Tensor]:
    flat_cls, flat_boxes = flatten_outputs(cls_scores, box_preds)
    
    assignments = build_batch_assignments(
        cls_scores, gt_boxes, gt_labels, strides,
        num_classes=num_classes, center_radius=center_radius,
    )

    target_labels = torch.stack([item.labels for item in assignments], dim=0)
    positive_masks = torch.stack([item.positive_mask for item in assignments], dim=0)
    target_boxes = torch.stack([item.box_targets for item in assignments], dim=0)

    loss_cls = sigmoid_focal_loss(flat_cls, target_labels, reduction="sum")

    num_positives = positive_masks.sum().item()
    if num_positives > 0:
        points = assignments[0].points
        point_strides = assignments[0].strides
        
        decoded_boxes = decode_flat_boxes(points, point_strides, flat_boxes)
        pred_boxes_pos = decoded_boxes[positive_masks]
        target_boxes_pos = target_boxes[positive_masks]
        
        loss_box = ciou_loss(pred_boxes_pos, target_boxes_pos, reduction="sum")
        normalizer = max(1.0, float(num_positives))
    else:
        loss_box = flat_cls.new_zeros(())
        normalizer = 1.0

    loss_cls = (loss_cls / normalizer) * cls_loss_weight
    loss_box = (loss_box / normalizer) * box_loss_weight
    total_loss = loss_cls + loss_box

    return {"loss": total_loss, "loss_cls": loss_cls, "loss_box": loss_box}


def plot_training_results(history: dict, save_dir: Path) -> None:
    # 🔒 CHỐNG LỖI MÔI TRƯỜNG LINUX/WSL: Ép matplotlib chạy chế độ lưu file ẩn không giao diện
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    if not epochs:
        return

    # Chỉ vẽ 1 ô duy nhất cho Train Loss (Đơn giản, trực quan và siêu nhẹ)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(epochs, history["train_loss"], 'r-', marker='o', markersize=3, label='Train Loss')
    ax.set_title('Huấn luyện Loss qua từng Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / "results.png", dpi=300)
    plt.close()


train_history = {"epoch": [], "train_loss": []}


def train_detector(args: argparse.Namespace) -> None:
    global train_history
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("using runtime device:", device)

    custom_save_dir = Path("/mnt/f/QuanPM/data7nhan/dataset/models")
    custom_save_dir.mkdir(parents=True, exist_ok=True)

    dataset = PlateDataset(annotation_file=args.annotations, image_root=args.image_root, image_size=args.size)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, # Huy động tối đa CPU đọc ảnh song song
        collate_fn=collate_plate_batch, 
        drop_last=True,
        pin_memory=True if device.type == "cuda" else False # Tăng tốc đẩy dữ liệu lên GPU
    )

    model = build_scrfd_plate_model().to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.937,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(loader),
        epochs=args.epochs,
        pct_start=0.05, 
        anneal_strategy='cos'
    )

    print(f"Dataset contains {len(dataset)} samples. Batches: {len(loader)}")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(loader, start=1):
            images = batch["images"].to(device)
            boxes = [item.to(device) for item in batch["boxes"]]
            labels = [item.to(device) for item in batch["labels"]]

            optimizer.zero_grad()
            cls_scores, box_preds = model(images)
            
            losses = compute_detection_loss(
                cls_scores, box_preds, boxes, labels, model.head.strides,
                num_classes=model.head.num_classes, center_radius=args.center_radius,
            )

            losses["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            scheduler.step() 

            epoch_loss += losses["loss"].item()

            if step % 50 == 0 or step == len(loader):
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"epoch {epoch:02d}/{args.epochs:02d} | step {step:03d}/{len(loader):03d} | "
                    f"loss={losses['loss'].item():.4f} | lr={current_lr:.6f}"
                )

        epoch_loss /= len(loader)
        print(f"--> epoch {epoch} finished! average loss = {epoch_loss:.4f}")
        
        # 💾 1. LUÔN CẬP NHẬT FILE TỔNG HỢP CUỐI CÙNG
        torch.save(model.state_dict(), custom_save_dir / "last.pt")
        
        # 💾 2. LƯU RIÊNG BIỆT THEO TỪNG SỐ THỨ TỰ EPOCH (Ví dụ: epoch_1.pt, epoch_2.pt)
        # Sử dụng định dạng f-string với `:02d` hoặc `:03d` để tên file sắp xếp thẳng hàng, đẹp mắt
        epoch_filename = f"epoch_{epoch:02d}.pt"
        torch.save(model.state_dict(), custom_save_dir / epoch_filename)

        # 📈 VẼ ĐỒ THỊ LOSS NGAY TRÊN LUỒNG CHÍNH (Siêu nhanh)
        train_history["epoch"].append(epoch)
        train_history["train_loss"].append(epoch_loss)
        try:
            plot_training_results(train_history, custom_save_dir)
            print(f"📊 [ĐÃ LƯU] Đã xuất {epoch_filename}, cập nhật 'last.pt' và đồ thị 'results.png'")
        except Exception as e:
            print(f"⚠️ Cảnh báo lỗi vẽ đồ thị tại luồng chính: {e}")
            
        print("-" * 75 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SCRFD-style multi-class traffic detector.")
    parser.add_argument("--annotations", required=True, help="JSON annotation file")
    parser.add_argument("--image-root", default=".", help="Root for relative image paths")
    parser.add_argument("--output-dir", default="traffic_runs")
    parser.add_argument("--size", type=int, default=640, help="Square training image size")
    parser.add_argument("--epochs", type=int, default=300) 
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2) 
    parser.add_argument("--weight-decay", type=float, default=5e-4) 
    parser.add_argument("--center-radius", type=float, default=2.5)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    train_detector(parse_args())