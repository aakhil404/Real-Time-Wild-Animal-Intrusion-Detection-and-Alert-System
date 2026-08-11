import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torchvision import transforms
from src.cnn_training import WildlifeVerifierCNN

# ── Per-species display configuration ─────────────────────────────────────────
# Maps CNN class name → (BGR color, alert prefix emoji, threat tier)
SPECIES_CONFIG = {
    'elephant': {
        'color':  (0, 0, 255),      # Red (BGR)
        'label':  '🐘 ELEPHANT',
        'tier':   'HIGH',           # Default alert tier for this species
    },
    'wild_boar': {
        'color':  (0, 100, 255),    # Orange (BGR)
        'label':  '🐗 WILD BOAR',
        'tier':   'MEDIUM',
    },
    # 'other' is intentionally absent — suppressed by the pipeline
}

# Alert score thresholds (shared across all species)
ALERT_HIGH_THRESHOLD   = 0.75
ALERT_MEDIUM_THRESHOLD = 0.45


class WildlifeIntrusionPipeline:
    def __init__(self, yolo_model_path, cnn_model_path, device=None, use_temporal=False):
        """
        Initializes the two-stage multi-species detection and verification pipeline.

        The CNN verifier output class ordering is determined alphabetically by
        ImageFolder (torchvision):  elephant=0, other=1, wild_boar=2
        This is resolved automatically by reading the class_names from the CNN
        dataset directory when available, or falling back to alphabetical order.
        """
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Initializing pipeline on device: {self.device}")

        # ── Load YOLOv8 detector ──────────────────────────────────────────────
        print(f"Loading YOLO detector: {yolo_model_path}")
        self.detector = YOLO(yolo_model_path)

        # ── Load CNN verifier ─────────────────────────────────────────────────
        print(f"Loading CNN verifier: {cnn_model_path}")
        checkpoint = torch.load(cnn_model_path, map_location=self.device)

        # Infer number of classes from the checkpoint's classifier head
        # The last Linear layer output = num_classes
        last_key = [k for k in checkpoint.keys() if 'classifier' in k and 'weight' in k]
        num_classes = checkpoint[last_key[-1]].shape[0] if last_key else 2

        # Check if the checkpoint keys indicate an attention-enhanced model architecture
        is_attention = any('attention_head' in k for k in checkpoint.keys())
        if is_attention:
            print("Auto-detected Attention-Enhanced CNN architecture.")
            from src.experimental.attention_heads import AttentionEnhancedVerifierCNN
            self.verifier = AttentionEnhancedVerifierCNN(num_classes=num_classes)
        else:
            print("Auto-detected Baseline CNN architecture.")
            self.verifier = WildlifeVerifierCNN(num_classes=num_classes)

        self.verifier.load_state_dict(checkpoint)
        self.verifier.to(self.device)
        self.verifier.eval()

        # ── Resolve class name mapping (alphabetical — matches ImageFolder) ───
        # 2-class: ['elephant', 'other']         → {0:'elephant', 1:'other'}
        # 3-class: ['elephant', 'other', 'wild_boar'] → {0:'elephant', 1:'other', 2:'wild_boar'}
        if num_classes == 2:
            self.class_names = ['elephant', 'other']
        elif num_classes == 3:
            self.class_names = ['elephant', 'other', 'wild_boar']
        else:
            # Generic fallback for future species additions
            self.class_names = [f'species_{i}' for i in range(num_classes)]
            self.class_names[self.class_names.index('species_1')] = 'other'

        print(f"CNN classes ({num_classes}): {self.class_names}")

        # ── CNN image preprocessing transform ─────────────────────────────────
        self.cnn_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # ── Setup Temporal threat tracking ──
        self.use_temporal = use_temporal
        if self.use_temporal:
            from src.experimental.temporal_modeling import DetectionTracker
            # Trackers for configured target species
            self.trackers = {
                'elephant': DetectionTracker(window_size=10, alert_threshold=4),
                'wild_boar': DetectionTracker(window_size=10, alert_threshold=4)
            }

    def predict_image(self, image_path, yolo_conf=0.25):
        """Runs the complete two-stage pipeline on a single image."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        h, w, _ = img.shape

        # ── Stage 1: YOLO Detection ───────────────────────────────────────────
        yolo_results = self.detector.predict(
            source=img, conf=yolo_conf, verbose=False, device='cpu'
        )

        detections = []

        for res in yolo_results:
            boxes = res.boxes
            for box in boxes:
                xyxy  = box.xyxy[0].tolist()
                y_conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue

                crop = img[y1:y2, x1:x2]

                # ── Stage 2: CNN Verification ─────────────────────────────────
                cnn_prob, cnn_class_idx = self._verify_crop(crop)
                cnn_class_name = self.class_names[cnn_class_idx]

                # Suppress 'other' detections — not a target species
                if cnn_class_name == 'other':
                    # Still record at very low score (useful for analysis)
                    alert_score  = y_conf * (1.0 - cnn_prob)
                    alert_level  = "SUPPRESSED"
                    species      = 'other'
                    verified     = False
                else:
                    species      = cnn_class_name
                    alert_score  = y_conf * cnn_prob
                    if alert_score >= ALERT_HIGH_THRESHOLD:
                        alert_level = "HIGH"
                    elif alert_score >= ALERT_MEDIUM_THRESHOLD:
                        alert_level = "MEDIUM"
                    else:
                        alert_level = "LOW"
                    verified = (alert_score >= ALERT_MEDIUM_THRESHOLD)

                detections.append({
                    "bbox":        [x1, y1, x2, y2],
                    "yolo_conf":   y_conf,
                    "cnn_prob":    cnn_prob,
                    "cnn_class":   cnn_class_name,
                    "species":     species,
                    "alert_score": alert_score,
                    "alert_level": alert_level,
                    "verified":    verified,
                })

        # Update temporal trackers if enabled
        if self.use_temporal and hasattr(self, 'trackers'):
            for species, tracker in self.trackers.items():
                is_detected_and_verified = any(
                    det["species"] == species and det["verified"]
                    for det in detections
                )
                tracker.update(is_detected_and_verified)

        return detections, img

    def _verify_crop(self, crop):
        """Runs the CNN verifier on an image crop. Returns (max_prob, class_idx)."""
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor   = self.cnn_transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs       = self.verifier(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        prob, class_idx = torch.max(probabilities, dim=0)
        return float(prob), int(class_idx)

    def draw_detections(self, img, detections, show_suppressed=False):
        """
        Draws bounding boxes color-coded by species and alert level.
        Suppressed ('other') detections are hidden by default.
        """
        annotated_img = img.copy()

        # Alert-level override colors (used when species not in SPECIES_CONFIG)
        level_colors = {
            "HIGH":       (0, 0, 255),    # Red
            "MEDIUM":     (0, 140, 255),  # Orange
            "LOW":        (0, 255, 255),  # Yellow
            "SUPPRESSED": (128, 128, 128) # Grey
        }

        for det in detections:
            if det["alert_level"] == "SUPPRESSED" and not show_suppressed:
                continue

            x1, y1, x2, y2 = det["bbox"]
            species     = det["species"]
            level       = det["alert_level"]
            score       = det["alert_score"]
            verified    = det["verified"]

            # Species color takes priority; fall back to level color
            cfg   = SPECIES_CONFIG.get(species, {})
            color = cfg.get('color', level_colors.get(level, (0, 255, 0)))
            label_prefix = cfg.get('label', species.upper())

            thickness = 3 if verified else 1
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)

            label = f"{label_prefix}: {score:.2f} [{level}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + tw, y1), color, -1)
            cv2.putText(annotated_img, label,
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

        # Overlay temporal threat levels if enabled
        if self.use_temporal and hasattr(self, 'trackers'):
            y_offset = 35
            for species, tracker in self.trackers.items():
                threat = tracker.get_threat_level()
                label = f"{species.upper()} THREAT: {threat}"
                color = (0, 0, 255) if "HIGH" in threat else ((0, 140, 255) if "LOW" in threat else (128, 128, 128))
                cv2.rectangle(annotated_img, (10, y_offset - 18), (350, y_offset + 5), (0, 0, 0), -1)
                cv2.putText(annotated_img, label, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y_offset += 28

        return annotated_img


def run_pipeline_demo(yolo_path, cnn_path, input_dir, output_dir, use_temporal=False):
    """Processes a directory of images and saves annotated alert frames."""
    pipeline = WildlifeIntrusionPipeline(yolo_path, cnn_path, use_temporal=use_temporal)

    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    img_files = sorted(list(input_path.glob('**/*.jpg')))
    print(f"Running pipeline on {len(img_files)} images...")

    # Dynamic alert counter — works for any number of species
    total_alerts = {}

    for img_file in img_files:
        try:
            detections, img = pipeline.predict_image(img_file)

            # Only save frames that have at least one verified detection
            verified = [d for d in detections if d["verified"]]
            if verified or use_temporal: # Save every frame if using temporal tracking to show rolling changes
                annotated_img = pipeline.draw_detections(img, detections)
                out_name = f"alert_{img_file.name}"
                cv2.imwrite(str(output_path / out_name), annotated_img)

                for det in verified:
                    key = f"{det['species']}_{det['alert_level']}"
                    total_alerts[key] = total_alerts.get(key, 0) + 1

        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")

    print(f"\nPipeline processing finished. Results saved to {output_dir}")
    print(f"Alert summary: {total_alerts}")
    return total_alerts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo",   type=str, required=True, help="YOLO model path")
    parser.add_argument("--cnn",    type=str, required=True, help="CNN model path")
    parser.add_argument("--input",  type=str, required=True, help="Input image directory")
    parser.add_argument("--output", type=str, required=True, help="Output image directory")
    parser.add_argument("--temporal", action="store_true", help="Enable temporal threat tracking")
    args = parser.parse_args()

    run_pipeline_demo(args.yolo, args.cnn, args.input, args.output, use_temporal=args.temporal)
