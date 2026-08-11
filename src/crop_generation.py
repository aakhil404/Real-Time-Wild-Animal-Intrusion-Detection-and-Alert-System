import os
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

def generate_crops(model_path, input_image_dir, output_crop_dir, conf_threshold=0.25):
    """
    Runs YOLOv8 detector on a directory of images, crops each detection,
    and saves the crops to output_crop_dir, along with a metadata CSV.
    """
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    input_path = Path(input_image_dir)
    output_path = Path(output_crop_dir)
    os.makedirs(output_path, exist_ok=True)
    
    crops_metadata = []
    
    # Supported image extensions
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_files = [f for f in input_path.glob('**/*') if f.suffix.lower() in valid_exts]
    print(f"Found {len(img_files)} images to run through YOLO detector.")
    
    crop_counter = 0
    
    for img_idx, img_file in enumerate(img_files):
        # Run inference
        results = model.predict(source=str(img_file), conf=conf_threshold, verbose=False, device='cpu')
        
        # Load image once for cropping
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        h, w, _ = img.shape
        
        for res in results:
            boxes = res.boxes
            for box_idx, box in enumerate(boxes):
                # Get coords
                xyxy = box.xyxy[0].tolist() # x1, y1, x2, y2
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Clip coords
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue
                
                # Crop
                crop = img[y1:y2, x1:x2]
                
                crop_filename = f"crop_{img_idx:05d}_{box_idx:02d}_{cls_name}.jpg"
                crop_filepath = output_path / crop_filename
                cv2.imwrite(str(crop_filepath), crop)
                
                crops_metadata.append({
                    "crop_filepath": str(crop_filepath),
                    "original_image": str(img_file),
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })
                crop_counter += 1
                
        if (img_idx + 1) % 100 == 0:
            print(f"Processed {img_idx + 1}/{len(img_files)} images...")
            
    # Save metadata
    df = pd.DataFrame(crops_metadata)
    csv_path = output_path / "crops_metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"Crop generation completed. Generated {crop_counter} crops.")
    print(f"Metadata saved to {csv_path}")
    return csv_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="path to YOLO model weights")
    parser.add_argument("--input_dir", type=str, required=True, help="input directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory to save crops")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    args = parser.parse_args()
    
    generate_crops(args.model, args.input_dir, args.output_dir, args.conf)
