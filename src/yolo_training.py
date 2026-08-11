import os
import yaml
import argparse
import shutil
from ultralytics import YOLO
from pathlib import Path

def setup_data_yaml(workspace_dir, use_tiny=False):
    """Updates paths in data.yaml or sets up a tiny dataset for quick CPU verification."""
    data_dir = Path(workspace_dir) / 'data' / 'elephant_thermal'
    base_yaml_path = data_dir / 'Elephant.v2i.yolov8' / 'data.yaml'

    if not base_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {base_yaml_path}. Run preprocessing first.")

    if use_tiny:
        # Create a tiny dataset structure to train in seconds
        tiny_dir = data_dir / 'Elephant.v2i.yolov8_tiny'
        os.makedirs(tiny_dir / 'train' / 'images', exist_ok=True)
        os.makedirs(tiny_dir / 'train' / 'labels', exist_ok=True)
        os.makedirs(tiny_dir / 'valid' / 'images', exist_ok=True)
        os.makedirs(tiny_dir / 'valid' / 'labels', exist_ok=True)

        # Copy 2 files from train and valid
        src_train_img = data_dir / 'Elephant.v2i.yolov8' / 'train' / 'images'
        src_train_lbl = data_dir / 'Elephant.v2i.yolov8' / 'train' / 'labels'
        src_val_img   = data_dir / 'Elephant.v2i.yolov8' / 'valid' / 'images'
        src_val_lbl   = data_dir / 'Elephant.v2i.yolov8' / 'valid' / 'labels'

        train_imgs = list(src_train_img.glob('*.jpg'))[:2]
        val_imgs   = list(src_val_img.glob('*.jpg'))[:2]

        for img in train_imgs:
            shutil.copy(img, tiny_dir / 'train' / 'images')
            lbl = src_train_lbl / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy(lbl, tiny_dir / 'train' / 'labels')

        for img in val_imgs:
            shutil.copy(img, tiny_dir / 'valid' / 'images')
            lbl = src_val_lbl / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy(lbl, tiny_dir / 'valid' / 'labels')

        with open(base_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        data['path'] = str(tiny_dir.resolve())
        data['train'] = 'train/images'
        data['val']   = 'valid/images'
        data['test']  = 'valid/images'  # fallback

        tiny_yaml_path = tiny_dir / 'data_tiny.yaml'
        with open(tiny_yaml_path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)

        print(f"Created tiny verification dataset at {tiny_dir}")
        return tiny_yaml_path
    else:
        with open(base_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        dataset_dir = base_yaml_path.parent.resolve()
        data['path']  = str(dataset_dir)
        data['train'] = 'train/images'
        data['val']   = 'valid/images'
        data['test']  = 'test/images'

        with open(base_yaml_path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)

        print(f"Updated data.yaml at {base_yaml_path} with absolute paths.")
        return base_yaml_path


def setup_combined_data_yaml(workspace_dir, use_tiny=False):
    """
    Creates a merged data.yaml that combines the elephant thermal dataset with
    the wild boar dataset. Both share the same annotation space — YOLO is trained
    as a generic detector, and species discrimination is handled by the CNN verifier.

    If the wild boar dataset is not available, falls back to elephant-only yaml.
    """
    data_dir       = Path(workspace_dir) / 'data'
    elephant_dir   = data_dir / 'elephant_thermal' / 'Elephant.v2i.yolov8'
    wildboar_dir   = data_dir / 'wild_boar'
    combined_dir   = data_dir / 'combined_yolo'
    combined_yaml  = combined_dir / 'data_combined.yaml'

    # Fall back to elephant-only if wild boar dataset is missing
    if not wildboar_dir.exists():
        print("Wild boar dataset not found. Using elephant-only YOLO yaml.")
        return setup_data_yaml(workspace_dir, use_tiny=use_tiny)

    # Detect nested Roboflow export structure (single subdir with train/valid/test)
    wb_subdirs = [d for d in wildboar_dir.iterdir() if d.is_dir()
                  and (d / 'train').exists()]
    wb_base = wb_subdirs[0] if wb_subdirs else wildboar_dir

    # Read both yamls to merge class lists
    el_yaml_path = elephant_dir / 'data.yaml'
    with open(el_yaml_path, 'r') as f:
        el_data = yaml.safe_load(f)

    wb_yaml_candidates = list(wb_base.glob('*.yaml'))
    wb_class_names = ['wild_boar']  # default fallback
    if wb_yaml_candidates:
        with open(wb_yaml_candidates[0], 'r') as f:
            wb_data = yaml.safe_load(f)
        wb_class_names = wb_data.get('names', ['wild_boar'])

    # Merge class lists: elephant dataset classes first, then wild boar additions
    el_classes = el_data.get('names', [])
    # Add 'wild_boar' if not already in the merged list
    merged_classes = list(el_classes)
    for cls in wb_class_names:
        if cls not in merged_classes:
            merged_classes.append(cls)

    # Build combined directory with symlinked/copied split folders
    os.makedirs(combined_dir / 'train' / 'images', exist_ok=True)
    os.makedirs(combined_dir / 'train' / 'labels', exist_ok=True)
    os.makedirs(combined_dir / 'valid' / 'images', exist_ok=True)
    os.makedirs(combined_dir / 'valid' / 'labels', exist_ok=True)
    os.makedirs(combined_dir / 'test'  / 'images', exist_ok=True)
    os.makedirs(combined_dir / 'test'  / 'labels', exist_ok=True)

    def copy_split(src_base, split_yolo, split_out, prefix):
        """Copies images and labels from a source dataset split into the combined dir."""
        src_img = src_base / split_yolo / 'images'
        src_lbl = src_base / split_yolo / 'labels'
        dst_img = combined_dir / split_out / 'images'
        dst_lbl = combined_dir / split_out / 'labels'

        copied = 0
        if src_img.exists():
            for f in src_img.glob('*'):
                dst = dst_img / f"{prefix}_{f.name}"
                if not dst.exists():
                    shutil.copy2(f, dst)
                    copied += 1
        if src_lbl.exists():
            for f in src_lbl.glob('*.txt'):
                dst = dst_lbl / f"{prefix}_{f.name}"
                if not dst.exists():
                    shutil.copy2(f, dst)
        return copied

    print("Building combined YOLO dataset (elephant + wild boar)...")
    for (el_split, wb_split, out_split) in [
        ('train', 'train', 'train'),
        ('valid', 'valid', 'valid'),
        ('test',  'test',  'test'),
    ]:
        n_el = copy_split(elephant_dir, el_split, out_split, prefix='el')
        n_wb = copy_split(wb_base, wb_split, out_split, prefix='wb')
        print(f"  {out_split}: copied {n_el} elephant + {n_wb} wild boar images.")

    if use_tiny:
        # Limit to 4 images for quick CPU testing
        for split in ['train', 'valid', 'test']:
            imgs = list((combined_dir / split / 'images').glob('*'))
            for extra in imgs[4:]:
                extra.unlink()
                lbl = combined_dir / split / 'labels' / (extra.stem + '.txt')
                if lbl.exists():
                    lbl.unlink()

    combined_data = {
        'path':  str(combined_dir.resolve()),
        'train': 'train/images',
        'val':   'valid/images',
        'test':  'test/images',
        'nc':    len(merged_classes),
        'names': merged_classes,
    }

    with open(combined_yaml, 'w') as f:
        yaml.safe_dump(combined_data, f, default_flow_style=False)

    print(f"Combined data.yaml saved to {combined_yaml}")
    print(f"  Classes ({len(merged_classes)}): {merged_classes}")
    return combined_yaml


def train_yolo(workspace_dir, epochs=3, batch=8, imgsz=640, use_tiny=False, combined=False):
    """
    Trains/fine-tunes a YOLOv8 model.

    Args:
        combined: If True and wild boar data exists, trains on the merged
                  elephant + wild boar dataset. Otherwise uses elephant only.
    """
    print("Preparing YOLOv8 training...")

    if combined:
        yaml_path = setup_combined_data_yaml(workspace_dir, use_tiny=use_tiny)
        run_name  = "wildlife_yolo_combined"
    else:
        yaml_path = setup_data_yaml(workspace_dir, use_tiny=use_tiny)
        run_name  = "elephant_yolo"

    model = YOLO("yolov8n.pt")

    if use_tiny:
        epochs = 1
        batch  = 2
        print("Using tiny mode: training for 1 epoch on a tiny subset.")

    print(f"Starting YOLOv8 training on {yaml_path}...")
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=os.path.join(workspace_dir, "runs", "detect"),
        name=run_name,
        device="cpu",
        workers=0,
        verbose=True
    )

    print("YOLOv8 training completed.")
    best_weights = os.path.join(
        workspace_dir, "runs", "detect", run_name, "weights", "best.pt"
    )
    return best_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,  default=3,     help="number of training epochs")
    parser.add_argument("--batch",    type=int,  default=8,     help="batch size")
    parser.add_argument("--imgsz",    type=int,  default=640,   help="image size")
    parser.add_argument("--tiny",     action="store_true",      help="run a quick training on tiny subset")
    parser.add_argument("--combined", action="store_true",      help="train on combined elephant+wild_boar dataset")
    args = parser.parse_args()

    ws_dir = r"c:\Users\aakhi\OneDrive\Desktop\DL PROJECT"
    train_yolo(ws_dir, epochs=args.epochs, batch=args.batch,
               imgsz=args.imgsz, use_tiny=args.tiny, combined=args.combined)
