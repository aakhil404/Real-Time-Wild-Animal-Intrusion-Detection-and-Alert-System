import os
import zipfile
import shutil
import random
import cv2
import numpy as np
from pathlib import Path

def extract_zip(zip_path, extract_to):
    """Extracts a zip file to a target directory if it hasn't been extracted yet."""
    if os.path.exists(extract_to) and len(os.listdir(extract_to)) > 0:
        print(f"Dataset already extracted to {extract_to}")
        return

    print(f"Extracting {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction completed.")

def crop_yolo_bbox(image_path, bbox_line, output_size=(128, 128)):
    """Crops a bounding box from an image using YOLO format label line."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w, _ = img.shape
    parts = bbox_line.strip().split()
    if len(parts) < 5:
        return None

    _, x_center, y_center, bbox_w, bbox_h = map(float, parts[:5])

    # Convert normalized coordinates to pixels
    x1 = int((x_center - bbox_w / 2) * w)
    y1 = int((y_center - bbox_h / 2) * h)
    x2 = int((x_center + bbox_w / 2) * w)
    y2 = int((y_center + bbox_h / 2) * h)

    # Clip coordinates to image boundary
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 - x1 < 5 or y2 - y1 < 5:
        return None

    crop = img[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, output_size)
    return crop_resized

def get_random_crop(image_path, crop_size=(128, 128)):
    """Extracts a random square crop from an image to serve as a negative sample."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w, _ = img.shape
    size = min(h, w, 256)  # Keep crops reasonable in size before resizing
    if size < 20:
        return None

    x1 = random.randint(0, w - size)
    y1 = random.randint(0, h - size)

    crop = img[y1:y1+size, x1:x1+size]
    crop_resized = cv2.resize(crop, crop_size)
    return crop_resized

def extract_positive_crops(dataset_dir, split_name, target_out, target_class_id=0,
                           species_tag="species", max_crops=1500):
    """
    Extracts positive crops for a given species from a YOLO-format dataset split.
    Returns the number of crops extracted and a list of non-target (other) crops found.
    """
    yolo_split = 'valid' if split_name == 'val' else split_name
    img_dir = Path(dataset_dir) / yolo_split / 'images'
    lbl_dir = Path(dataset_dir) / yolo_split / 'labels'

    os.makedirs(target_out, exist_ok=True)
    target_count = 0
    other_crops = []  # (img_file, line) pairs for non-target bboxes

    if not lbl_dir.exists():
        print(f"  Labels not found at {lbl_dir}, skipping.")
        return target_count, other_crops

    lbl_files = list(lbl_dir.glob('*.txt'))
    random.shuffle(lbl_files)

    for lbl_file in lbl_files:
        # Support both .jpg and .png source images
        img_file = img_dir / f"{lbl_file.stem}.jpg"
        if not img_file.exists():
            img_file = img_dir / f"{lbl_file.stem}.png"
        if not img_file.exists():
            continue

        with open(lbl_file, 'r') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])

            if class_id == target_class_id:
                if target_count >= max_crops:
                    continue
                crop = crop_yolo_bbox(img_file, line)
                if crop is not None:
                    cv2.imwrite(
                        str(target_out / f"{species_tag}_{lbl_file.stem}_crop_{idx}.jpg"),
                        crop
                    )
                    target_count += 1
            else:
                # Collect non-target bboxes to potentially use as 'other' negatives
                other_crops.append((img_file, line))

    return target_count, other_crops


def create_cnn_dataset(elephant_dir, serengeti_dir, output_dir,
                       wildboar_dir=None, wildboar_class_id=0,
                       max_crops_per_split=1500):
    """
    Creates a balanced multi-species CNN dataset.

    Classes produced:
      - elephant/   → positive crops from elephant thermal dataset (class 0)
      - wild_boar/  → positive crops from wild boar dataset (class 0 by default)
                      [only created if wildboar_dir is provided]
      - other/      → non-target classes + random Serengeti crops

    If wildboar_dir is None, reverts to original 2-class (elephant / other) mode.
    """
    multi_species = wildboar_dir is not None
    mode = "3-class (elephant / wild_boar / other)" if multi_species else "2-class (elephant / other)"
    print(f"Building CNN crop dataset [{mode}]...")

    splits = {
        'train': 'train',
        'val':   'val',
        'test':  'test',
    }

    for split_name in splits:
        print(f"\nProcessing split: {split_name}...")

        # ── Output directories ───────────────────────────────────────────────
        elephant_out  = Path(output_dir) / split_name / 'elephant'
        other_out     = Path(output_dir) / split_name / 'other'
        os.makedirs(elephant_out, exist_ok=True)
        os.makedirs(other_out, exist_ok=True)

        if multi_species:
            wildboar_out = Path(output_dir) / split_name / 'wild_boar'
            os.makedirs(wildboar_out, exist_ok=True)

        # ── 1. Elephant positive crops ────────────────────────────────────────
        elephant_dataset_split = Path(elephant_dir) / 'Elephant.v2i.yolov8'
        el_count, el_other_crops = extract_positive_crops(
            dataset_dir=elephant_dataset_split,
            split_name=split_name,
            target_out=elephant_out,
            target_class_id=0,
            species_tag="el",
            max_crops=max_crops_per_split
        )
        print(f"  Elephants: {el_count} crops extracted.")

        # ── 2. Wild boar positive crops (if dataset provided) ─────────────────
        wb_count = 0
        wb_other_crops = []
        if multi_species:
            # Detect if the wild boar dataset has a subdirectory (Roboflow export style)
            wildboar_path = Path(wildboar_dir)
            subdirs = [d for d in wildboar_path.iterdir() if d.is_dir()]
            # If there's exactly one subdir and it contains train/valid/test, use it
            wb_dataset_dir = wildboar_path
            if len(subdirs) == 1 and (subdirs[0] / 'train').exists():
                wb_dataset_dir = subdirs[0]

            wb_count, wb_other_crops = extract_positive_crops(
                dataset_dir=wb_dataset_dir,
                split_name=split_name,
                target_out=wildboar_out,
                target_class_id=wildboar_class_id,
                species_tag="wb",
                max_crops=max_crops_per_split
            )
            print(f"  Wild Boars: {wb_count} crops extracted.")

        # ── 3. 'Other' negative crops ─────────────────────────────────────────
        # First, use non-target bboxes collected from both datasets
        all_other_bbox_crops = el_other_crops + wb_other_crops
        random.shuffle(all_other_bbox_crops)
        other_bbox_count = 0
        other_bbox_limit = max_crops_per_split // 3

        for img_file, line in all_other_bbox_crops:
            if other_bbox_count >= other_bbox_limit:
                break
            crop = crop_yolo_bbox(img_file, line)
            if crop is not None:
                cv2.imwrite(
                    str(other_out / f"bbox_other_{other_bbox_count}.jpg"),
                    crop
                )
                other_bbox_count += 1

        print(f"  Other (from bbox labels): {other_bbox_count} crops.")

        # Then, top up 'other' with random Serengeti crops
        ser_split = 'train'  # Always use Serengeti train for all splits (large enough)
        ser_blank_dir    = Path(serengeti_dir) / ser_split / 'blank'
        ser_nonblank_dir = Path(serengeti_dir) / ser_split / 'non_blank'

        ser_files = []
        if ser_blank_dir.exists():
            ser_files.extend(list(ser_blank_dir.glob('*.jpg')))
        if ser_nonblank_dir.exists():
            ser_files.extend(list(ser_nonblank_dir.glob('*.jpg')))
        random.shuffle(ser_files)

        # Target: match the number of positive samples (largest species count)
        target_positives = max(el_count, wb_count) if multi_species else el_count
        needed_negatives = max(0, target_positives - other_bbox_count)
        ser_count = 0

        for ser_file in ser_files:
            if ser_count >= needed_negatives:
                break
            crop = get_random_crop(ser_file)
            if crop is not None:
                cv2.imwrite(
                    str(other_out / f"ser_{ser_file.stem}_crop.jpg"),
                    crop
                )
                ser_count += 1

        print(f"  Other (from Serengeti): {ser_count} crops.")
        total_other = other_bbox_count + ser_count
        print(f"  Split summary -> elephant: {el_count}"
              + (f", wild_boar: {wb_count}" if multi_species else "")
              + f", other: {total_other}")


def partition_wildboar_dataset(wildboar_dir, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Partitions wild boar train dataset into train, valid, and test splits
    if valid and test folders do not exist.
    """
    wildboar_path = Path(wildboar_dir)
    train_img_dir = wildboar_path / 'train' / 'images'
    train_lbl_dir = wildboar_path / 'train' / 'labels'
    
    val_img_dir = wildboar_path / 'valid' / 'images'
    val_lbl_dir = wildboar_path / 'valid' / 'labels'
    test_img_dir = wildboar_path / 'test' / 'images'
    test_lbl_dir = wildboar_path / 'test' / 'labels'
    
    if val_img_dir.exists() and test_img_dir.exists():
        print("Wild boar validation and test splits already exist.")
        return

    print("Partitioning wild boar train dataset into train/valid/test splits...")
    
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_lbl_dir, exist_ok=True)
    
    img_files = list(train_img_dir.glob('*.jpg')) + list(train_img_dir.glob('*.png'))
    random.seed(seed)
    random.shuffle(img_files)
    
    num_total = len(img_files)
    num_val = int(num_total * val_ratio)
    num_test = int(num_total * test_ratio)
    
    val_files = img_files[:num_val]
    test_files = img_files[num_val:num_val + num_test]
    
    def move_files(files, dest_img, dest_lbl):
        moved_count = 0
        for img_path in files:
            # Move image
            shutil.move(str(img_path), str(dest_img / img_path.name))
            
            # Find and move label
            lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(dest_lbl / lbl_path.name))
            moved_count += 1
        return moved_count

    n_val = move_files(val_files, val_img_dir, val_lbl_dir)
    n_test = move_files(test_files, test_img_dir, test_lbl_dir)
    n_train = num_total - n_val - n_test
    
    print(f"Wild boar partitioned: {n_train} train, {n_val} valid, {n_test} test images.")


def main(workspace_dir, wildboar_zip=None, wildboar_class_id=0):
    data_dir = os.path.join(workspace_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # ── Extract core datasets ─────────────────────────────────────────────────
    extract_zip(
        os.path.join(workspace_dir, "elephant_thermal.zip"),
        os.path.join(data_dir, "elephant_thermal")
    )
    extract_zip(
        os.path.join(workspace_dir, "serangetti.zip"),
        os.path.join(data_dir, "serengeti")
    )

    # ── Extract wild boar dataset (optional) ──────────────────────────────────
    wildboar_dir = None
    if wildboar_zip and os.path.exists(wildboar_zip):
        wildboar_dir = os.path.join(data_dir, "wild_boar")
        extract_zip(wildboar_zip, wildboar_dir)
        print(f"Wild boar dataset extracted to {wildboar_dir}")
        partition_wildboar_dataset(wildboar_dir)
    elif wildboar_zip:
        print(f"Warning: Wild boar zip not found at '{wildboar_zip}'. "
              f"Proceeding with elephant-only mode.")

    # ── Build CNN dataset ─────────────────────────────────────────────────────
    create_cnn_dataset(
        elephant_dir=os.path.join(data_dir, "elephant_thermal"),
        serengeti_dir=os.path.join(data_dir, "serengeti"),
        output_dir=os.path.join(data_dir, "cnn_dataset"),
        wildboar_dir=wildboar_dir,
        wildboar_class_id=wildboar_class_id
    )
    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    import sys
    ws_dir = r"c:\Users\aakhi\OneDrive\Desktop\DL PROJECT"
    main(ws_dir)
