import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from src.cnn_training import WildlifeVerifierCNN, get_data_loaders
from src.inference import WildlifeIntrusionPipeline

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    return intersection / union

def evaluate_cnn_only(cnn_model_path, data_dir, output_dir):
    """Evaluates the CNN classifier alone on the test crop split.
    Works for any number of classes — auto-detected from the dataset folder structure.
    """
    print("Evaluating CNN Verifier on test crops...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dir = os.path.join(data_dir, 'test')
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found. Skipping CNN-only evaluation.")
        return

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    num_classes  = len(test_dataset.classes)
    class_names  = test_dataset.classes

    print(f"Detected {num_classes} classes: {class_names}")

    # Auto-infer num_classes from checkpoint to match training
    checkpoint = torch.load(cnn_model_path, map_location=device)
    last_key   = [k for k in checkpoint.keys() if 'classifier' in k and 'weight' in k]
    ckpt_classes = checkpoint[last_key[-1]].shape[0] if last_key else num_classes

    # Check if the checkpoint keys indicate an attention-enhanced model architecture
    is_attention = any('attention_head' in k for k in checkpoint.keys())
    if is_attention:
        print("Auto-detected Attention-Enhanced CNN architecture for evaluation.")
        from src.experimental.attention_heads import AttentionEnhancedVerifierCNN
        model = AttentionEnhancedVerifierCNN(num_classes=ckpt_classes).to(device)
    else:
        print("Auto-detected Baseline CNN architecture for evaluation.")
        model = WildlifeVerifierCNN(num_classes=ckpt_classes).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    print("\n--- CNN Classifier Report ---")
    print(classification_report(all_targets, all_preds, target_names=class_names))

    # Confusion matrix — scales to any number of classes
    cm = confusion_matrix(all_targets, all_preds)
    fig_size = max(6, num_classes * 2)
    plt.figure(figsize=(fig_size, fig_size - 1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('CNN Verifier Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'cnn_confusion_matrix.png'))
    plt.close()
    print(f"CNN confusion matrix saved to {output_dir}")

def evaluate_pipeline(yolo_model_path, cnn_model_path, test_dir_root, output_dir,
                      target_class_id=0, eval_limit=100):
    """
    Evaluates YOLO-only vs the complete YOLO+CNN pipeline on a test set.
    Works for any species — uses IoU matching against ground truth YOLO labels.
    Target class detections (class 0 by default = elephant / primary species)
    are compared against all pipeline-verified detections.

    Args:
        test_dir_root: Directory containing 'images/' and 'labels/' subdirs.
        target_class_id: The YOLO class ID considered as ground-truth positive.
        eval_limit: Max images to evaluate (for speed on CPU).
    """
    print("\nEvaluating integrated pipeline vs YOLO-only...")
    pipeline = WildlifeIntrusionPipeline(yolo_model_path, cnn_model_path)

    img_dir = Path(test_dir_root) / 'images'
    lbl_dir = Path(test_dir_root) / 'labels'

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"Test folders not found at {test_dir_root}. Skipping pipeline evaluation.")
        return

    img_files = list(img_dir.glob('*.jpg'))
    print(f"Evaluating on {min(len(img_files), eval_limit)} / {len(img_files)} test images...")

    yolo_tp, yolo_fp, yolo_fn = 0, 0, 0
    pipe_tp, pipe_fp, pipe_fn = 0, 0, 0

    for img_file in img_files[:eval_limit]:
        lbl_file = lbl_dir / f"{img_file.stem}.txt"
        if not lbl_file.exists():
            continue

        img_temp = cv2.imread(str(img_file))
        if img_temp is None:
            continue
        h, w, _ = img_temp.shape

        # Read ground truth boxes for the target class
        gts = []
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if int(parts[0]) == target_class_id:
                    _, xc, yc, bw, bh = map(float, parts[:5])
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)
                    gts.append([x1, y1, x2, y2])

        try:
            detections, _ = pipeline.predict_image(img_file, yolo_conf=0.25)
        except Exception:
            continue

        yolo_dets = [d["bbox"] for d in detections]
        pipe_dets = [d["bbox"] for d in detections if d["verified"]]

        # ── Match YOLO-only ───────────────────────────────────────────────────
        matched_gts_yolo = set()
        for det_box in yolo_dets:
            matched = False
            for gt_idx, gt_box in enumerate(gts):
                if gt_idx not in matched_gts_yolo and calculate_iou(det_box, gt_box) >= 0.5:
                    yolo_tp += 1
                    matched_gts_yolo.add(gt_idx)
                    matched = True
                    break
            if not matched:
                yolo_fp += 1
        yolo_fn += len(gts) - len(matched_gts_yolo)

        # ── Match Pipeline ────────────────────────────────────────────────────
        matched_gts_pipe = set()
        for det_box in pipe_dets:
            matched = False
            for gt_idx, gt_box in enumerate(gts):
                if gt_idx not in matched_gts_pipe and calculate_iou(det_box, gt_box) >= 0.5:
                    pipe_tp += 1
                    matched_gts_pipe.add(gt_idx)
                    matched = True
                    break
            if not matched:
                pipe_fp += 1
        pipe_fn += len(gts) - len(matched_gts_pipe)

    def prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    y_prec, y_rec, y_f1 = prf(yolo_tp, yolo_fp, yolo_fn)
    p_prec, p_rec, p_f1 = prf(pipe_tp, pipe_fp, pipe_fn)

    print("\n--- Pipeline Evaluation (YOLO vs YOLO+CNN) ---")
    print(f"YOLO Only        — Precision: {y_prec:.4f}, Recall: {y_rec:.4f}, F1: {y_f1:.4f}")
    print(f"YOLO + Verifier  — Precision: {p_prec:.4f}, Recall: {p_rec:.4f}, F1: {p_f1:.4f}")

    metrics     = ['Precision', 'Recall', 'F1-score']
    yolo_scores = [y_prec, y_rec, y_f1]
    pipe_scores = [p_prec, p_rec, p_f1]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width/2, yolo_scores, width, label='YOLOv8 Only',          color='skyblue')
    ax.bar(x + width/2, pipe_scores, width, label='YOLOv8 + CNN Verifier', color='coral')
    ax.set_ylabel('Score')
    ax.set_title('Impact of Verification Stage on Detection Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'pipeline_metric_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Pipeline metric comparison plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", type=str, default="runs/detect/elephant_yolo/weights/best.pt", help="YOLO model path")
    parser.add_argument("--cnn", type=str, default="runs/cnn_verifier.pth", help="CNN model path")
    parser.add_argument("--cnn_data", type=str, default="data/cnn_dataset", help="CNN crops dataset folder")
    parser.add_argument("--yolo_data_test", type=str, default="data/elephant_thermal/Elephant.v2i.yolov8/test", help="YOLO test directory")
    parser.add_argument("--output_dir", type=str, default="runs", help="Output metrics folder")
    args = parser.parse_args()
    
    evaluate_cnn_only(args.cnn, args.cnn_data, args.output_dir)
    evaluate_pipeline(args.yolo, args.cnn, args.yolo_data_test, args.output_dir)
