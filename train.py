import argparse
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

try:
    from .assigner import build_batch_assignments, decode_flat_boxes, flatten_outputs
    from .dataset import TrafficDataset, collate_traffic_batch
    from .losses import ciou_loss, reduce_loss_dict, sigmoid_focal_loss
    from .main import build_scrfd_traffic_model
except ImportError:
    from assigner import build_batch_assignments, decode_flat_boxes, flatten_outputs
    from dataset import TrafficDataset, collate_traffic_batch
    from losses import ciou_loss, reduce_loss_dict, sigmoid_focal_loss
    from main import build_scrfd_traffic_model


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
    size_ranges: Optional[Mapping[int, Tuple[float, float]]] = None,
) -> Dict[str, torch.Tensor]:
    flat_cls, flat_boxes = flatten_outputs(cls_scores, box_preds)
    
    assignments = build_batch_assignments(
        cls_scores, gt_boxes, gt_labels, strides,
        num_classes=num_classes,
        center_radius=center_radius,
        size_ranges=size_ranges,
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


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-7)


def average_precision(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    if recalls.numel() == 0:
        return 0.0

    recall_levels = torch.linspace(0.0, 1.0, 101)
    ap = recalls.new_zeros(())
    for recall_level in recall_levels:
        valid = recalls >= recall_level
        if valid.any():
            ap += precisions[valid].max()
    return float(ap / len(recall_levels))


def compute_class_ap(
    detections: Sequence[Tuple[int, float, torch.Tensor]],
    gt_by_image: Dict[int, torch.Tensor],
    num_gt: int,
    iou_threshold: float,
) -> float:
    if num_gt == 0:
        return float("nan")
    if not detections:
        return 0.0

    sorted_detections = sorted(detections, key=lambda item: item[1], reverse=True)
    matched = {
        image_idx: torch.zeros(len(boxes), dtype=torch.bool)
        for image_idx, boxes in gt_by_image.items()
    }
    true_positives = torch.zeros(len(sorted_detections))
    false_positives = torch.zeros(len(sorted_detections))

    for det_idx, (image_idx, _, pred_box) in enumerate(sorted_detections):
        gt_boxes = gt_by_image.get(image_idx)
        if gt_boxes is None or gt_boxes.numel() == 0:
            false_positives[det_idx] = 1.0
            continue

        ious = box_iou(pred_box[None, :], gt_boxes).squeeze(0)
        best_iou, best_gt_idx = ious.max(dim=0)
        if best_iou >= iou_threshold and not matched[image_idx][best_gt_idx]:
            true_positives[det_idx] = 1.0
            matched[image_idx][best_gt_idx] = True
        else:
            false_positives[det_idx] = 1.0

    tp_cumsum = true_positives.cumsum(dim=0)
    fp_cumsum = false_positives.cumsum(dim=0)
    recalls = tp_cumsum / max(1, num_gt)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum).clamp(min=1e-7)
    return average_precision(recalls, precisions)


@torch.no_grad()
def evaluate_detector(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    score_threshold: float = 0.05,
    nms_threshold: float = 0.45,
    max_detections: int = 100,
    max_images: int = 0,
) -> Dict[str, object]:
    model.eval()
    iou_thresholds = torch.linspace(0.50, 0.95, 10)
    detections_by_class = [[] for _ in range(num_classes)]
    gt_by_class = [dict() for _ in range(num_classes)]
    num_gt_by_class = [0 for _ in range(num_classes)]
    image_idx = 0

    for batch in loader:
        if max_images > 0 and image_idx >= max_images:
            break

        if max_images > 0:
            remaining = max_images - image_idx
            images = batch["images"][:remaining].to(device)
            batch_boxes = batch["boxes"][:remaining]
            batch_labels = batch["labels"][:remaining]
        else:
            images = batch["images"].to(device)
            batch_boxes = batch["boxes"]
            batch_labels = batch["labels"]

        predictions = model.predict(
            images,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
        )

        for pred, gt_boxes, gt_labels in zip(predictions, batch_boxes, batch_labels):
            gt_boxes = gt_boxes.cpu()
            gt_labels = gt_labels.cpu()
            pred = pred.detach().cpu()

            for class_idx in range(num_classes):
                class_gt = gt_boxes[gt_labels == class_idx]
                if class_gt.numel() > 0:
                    gt_by_class[class_idx][image_idx] = class_gt
                    num_gt_by_class[class_idx] += class_gt.shape[0]

            if pred.numel() > 0:
                pred_labels = pred[:, 5].long()
                for class_idx in range(num_classes):
                    class_pred = pred[pred_labels == class_idx]
                    for row in class_pred:
                        detections_by_class[class_idx].append(
                            (image_idx, float(row[4]), row[:4])
                        )

            image_idx += 1

    per_class_ap = {}
    class_maps = []
    map50_values = []

    for class_idx in range(num_classes):
        threshold_aps = [
            compute_class_ap(
                detections_by_class[class_idx],
                gt_by_class[class_idx],
                num_gt_by_class[class_idx],
                float(iou_threshold),
            )
            for iou_threshold in iou_thresholds
        ]
        valid_aps = [ap for ap in threshold_aps if ap == ap]
        class_map = sum(valid_aps) / len(valid_aps) if valid_aps else float("nan")
        per_class_ap[class_idx] = {
            "ap": class_map,
            "ap50": threshold_aps[0],
            "num_gt": num_gt_by_class[class_idx],
        }
        if class_map == class_map:
            class_maps.append(class_map)
        if threshold_aps[0] == threshold_aps[0]:
            map50_values.append(threshold_aps[0])

    return {
        "map": sum(class_maps) / len(class_maps) if class_maps else 0.0,
        "map50": sum(map50_values) / len(map50_values) if map50_values else 0.0,
        "per_class_ap": per_class_ap,
        "num_images": image_idx,
    }


def plot_training_results(history: dict, save_dir: Path) -> None:
    # 🔒 CHỐNG LỖI MÔI TRƯỜNG LINUX/WSL: Ép matplotlib chạy chế độ lưu file ẩn không giao diện
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    if not epochs:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(epochs, history["train_loss"], 'r-', marker='o', markersize=3, label='Train Loss')
    ax.set_title('Huấn luyện Loss qua từng Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.grid(True)

    val_epochs = [
        epoch for epoch, val_map in zip(history["epoch"], history["val_map"])
        if val_map is not None
    ]
    val_maps = [val_map for val_map in history["val_map"] if val_map is not None]
    if val_maps:
        ax_map = ax.twinx()
        ax_map.plot(val_epochs, val_maps, 'b-', marker='s', markersize=3, label='Val mAP')
        ax_map.set_ylabel('mAP')
        ax_map.set_ylim(0.0, 1.0)

        loss_lines, loss_labels = ax.get_legend_handles_labels()
        map_lines, map_labels = ax_map.get_legend_handles_labels()
        ax.legend(loss_lines + map_lines, loss_labels + map_labels, loc="best")
    else:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / "results.png", dpi=300)
    plt.close()


train_history = {"epoch": [], "train_loss": [], "val_map": [], "val_map50": []}


def train_detector(args: argparse.Namespace) -> None:
    global train_history
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("using runtime device:", device)

    custom_save_dir = Path("/mnt/f/QuanPM/data7nhan/dataset/models")
    custom_save_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 <= args.val_split < 1.0:
        raise ValueError("--val-split must be in the range [0.0, 1.0)")

    full_dataset = TrafficDataset(
        annotation_file=args.annotations,
        image_root=args.image_root,
        image_size=args.size,
    )
    val_count = int(len(full_dataset) * args.val_split)
    if args.val_split > 0 and val_count == 0 and len(full_dataset) > 1:
        val_count = 1
    train_count = len(full_dataset) - val_count
    if train_count <= 0:
        raise ValueError("Validation split leaves no training samples")

    split_generator = torch.Generator().manual_seed(args.split_seed)
    shuffled_indices = torch.randperm(len(full_dataset), generator=split_generator).tolist()
    val_indices = shuffled_indices[:val_count]
    train_indices = shuffled_indices[val_count:]

    train_dataset = TrafficDataset(
        annotation_file=args.annotations,
        image_root=args.image_root,
        image_size=args.size,
        indices=train_indices,
        augment=args.augment,
        hflip_prob=args.hflip_prob,
        color_jitter=args.color_jitter,
        scale_jitter=args.scale_jitter,
    )
    if val_count > 0:
        val_dataset = TrafficDataset(
            annotation_file=args.annotations,
            image_root=args.image_root,
            image_size=args.size,
            indices=val_indices,
            augment=False,
        )
    else:
        val_dataset = None

    pin_memory = args.pin_memory and device.type == "cuda"
    loader_kwargs = {
        "num_workers": args.workers,
        "collate_fn": collate_traffic_batch,
        "pin_memory": pin_memory,
    }
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers

    loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True,
        **loader_kwargs,
    )
    if len(loader) == 0:
        raise ValueError("Training loader has zero batches; reduce --batch-size or --val-split")

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

    model = build_scrfd_traffic_model().to(device)
    
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

    print(
        f"Dataset contains {len(full_dataset)} samples "
        f"({train_count} train / {val_count} val). Batches: {len(loader)}"
    )
    best_map = -1.0
    
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

        val_metrics = None
        should_eval = (
            val_loader is not None
            and args.eval_interval > 0
            and (epoch % args.eval_interval == 0 or epoch == args.epochs)
        )
        if should_eval:
            val_metrics = evaluate_detector(
                model,
                val_loader,
                device,
                num_classes=model.head.num_classes,
                score_threshold=args.eval_score_threshold,
                nms_threshold=args.eval_nms_threshold,
                max_detections=args.eval_max_detections,
                max_images=args.eval_max_images,
            )
            print(
                f"--> validation: mAP={val_metrics['map']:.4f} | "
                f"mAP50={val_metrics['map50']:.4f} | images={val_metrics['num_images']}"
            )
            per_class_text = []
            for class_idx, class_metrics in val_metrics["per_class_ap"].items():
                ap = class_metrics["ap"]
                ap50 = class_metrics["ap50"]
                ap_text = "nan" if ap != ap else f"{ap:.3f}"
                ap50_text = "nan" if ap50 != ap50 else f"{ap50:.3f}"
                per_class_text.append(
                    f"class_{class_idx}: AP={ap_text}, AP50={ap50_text}, n={class_metrics['num_gt']}"
                )
            print("--> per-class: " + " | ".join(per_class_text))
            if val_metrics["map"] > best_map:
                best_map = val_metrics["map"]
                torch.save(model.state_dict(), custom_save_dir / "best.pt")
                print(f"🏆 [BEST] Saved best.pt with val mAP={best_map:.4f}")
        
        # 💾 1. LUÔN CẬP NHẬT FILE TỔNG HỢP CUỐI CÙNG
        torch.save(model.state_dict(), custom_save_dir / "last.pt")
        
        # 💾 2. LƯU RIÊNG BIỆT THEO TỪNG SỐ THỨ TỰ EPOCH (Ví dụ: epoch_1.pt, epoch_2.pt)
        # Sử dụng định dạng f-string với `:02d` hoặc `:03d` để tên file sắp xếp thẳng hàng, đẹp mắt
        epoch_filename = f"epoch_{epoch:02d}.pt"
        torch.save(model.state_dict(), custom_save_dir / epoch_filename)

        # 📈 VẼ ĐỒ THỊ LOSS NGAY TRÊN LUỒNG CHÍNH (Siêu nhanh)
        train_history["epoch"].append(epoch)
        train_history["train_loss"].append(epoch_loss)
        train_history["val_map"].append(None if val_metrics is None else val_metrics["map"])
        train_history["val_map50"].append(None if val_metrics is None else val_metrics["map50"])
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
    parser.add_argument("--pin-memory", action="store_true", help="Enable CUDA pinned-memory DataLoader transfer")
    parser.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive between epochs")
    parser.add_argument("--lr", type=float, default=1e-2) 
    parser.add_argument("--weight-decay", type=float, default=5e-4) 
    parser.add_argument("--center-radius", type=float, default=2.5)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="Enable train-only data augmentation")
    parser.add_argument("--hflip-prob", type=float, default=0.5, help="Random horizontal flip probability")
    parser.add_argument("--color-jitter", type=float, default=0.2, help="Brightness/contrast/saturation jitter strength")
    parser.add_argument("--scale-jitter", type=float, default=0.2, help="Random scale jitter strength before letterbox")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data used for validation")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--eval-interval", type=int, default=5, help="Run validation every N epochs")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Validation batch size")
    parser.add_argument("--eval-score-threshold", type=float, default=0.05, help="Prediction score threshold for mAP")
    parser.add_argument("--eval-nms-threshold", type=float, default=0.45, help="NMS IoU threshold for mAP")
    parser.add_argument("--eval-max-detections", type=int, default=100, help="Max predictions per image for mAP")
    parser.add_argument("--eval-max-images", type=int, default=200, help="Max validation images per eval; 0 means all")
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    train_detector(parse_args())
