import os
import argparse
import sys
from pathlib import Path

# Add current workspace directory to system path to ensure relative module imports work correctly
workspace_dir = r"c:\Users\aakhi\OneDrive\Desktop\DL PROJECT"
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

def find_latest_yolo_weights(workspace_dir):
    """Dynamically locates the most recently modified YOLO 'best.pt' checkpoint."""
    detect_dir = os.path.join(workspace_dir, "runs", "detect")
    if not os.path.exists(detect_dir):
        return "yolov8n.pt"
        
    candidates = []
    for root, dirs, files in os.walk(detect_dir):
        if "best.pt" in files:
            best_path = os.path.join(root, "best.pt")
            candidates.append((best_path, os.path.getmtime(best_path)))
            
    if candidates:
        # Sort by last modified time descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        print(f"Auto-detected latest YOLO checkpoint: {candidates[0][0]}")
        return candidates[0][0]
        
    return "yolov8n.pt"

def main():
    parser = argparse.ArgumentParser(description="Real-Time Wild Animal Intrusion Detection and Alert System CLI")
    
    # Modes
    parser.add_argument("--preprocess", action="store_true", help="Extract raw datasets and build CNN crop splits")
    parser.add_argument("--train-yolo", action="store_true", help="Fine-tune YOLOv8 detector on elephant images")
    parser.add_argument("--crop-gen", action="store_true", help="Run YOLOv8 to generate crop predictions on an input folder")
    parser.add_argument("--train-cnn", action="store_true", help="Train PyTorch CNN verifier on cropped images")
    parser.add_argument("--demo-pipeline", action="store_true", help="Run 2-stage YOLO+CNN detection and alerting pipeline")
    parser.add_argument("--evaluate", action="store_true", help="Run CNN and pipeline evaluation (metrics and plots)")
    parser.add_argument("--explain", action="store_true", help="Run Grad-CAM explainability on a specific crop image")
    
    # Experimental Scaffolds
    parser.add_argument("--ssl-pretrain", action="store_true", help="Run SimCLR contrastive pretraining on Serengeti images")
    parser.add_argument("--compress", action="store_true", help="Quantize and prune the CNN verifier model")
    
    # Configuration / Arguments override
    parser.add_argument("--epochs",           type=int,   default=None,  help="override number of epochs for training stages")
    parser.add_argument("--batch",            type=int,   default=None,  help="override batch size")
    parser.add_argument("--input_dir",        type=str,   default=None,  help="input directory path for inference or crop-gen")
    parser.add_argument("--output_dir",       type=str,   default=None,  help="output directory path")
    parser.add_argument("--crop_file",        type=str,   default=None,  help="path to image crop for Grad-CAM explainability")
    parser.add_argument("--tiny",             action="store_true",        help="use a tiny verification dataset/epoch count for fast CPU runs")
    parser.add_argument("--imgsz",            type=int,   default=640,   help="YOLO image size (default: 640)")
    
    # Multi-species arguments
    parser.add_argument("--wildboar_zip",     type=str,   default=None,  help="path to wild boar dataset zip (Roboflow YOLOv8 format)")
    parser.add_argument("--wildboar_class_id",type=int,   default=0,     help="class ID for wild boar in the wild boar dataset (default: 0)")
    parser.add_argument("--combined",         action="store_true",        help="train YOLO on combined elephant+wild_boar dataset")
    
    # Advanced / Experimental extension arguments
    parser.add_argument("--attention",        action="store_true",        help="train/use the spatial self-attention enhanced verifier CNN")
    parser.add_argument("--curriculum",       action="store_true",        help="train CNN using curriculum learning pacing")
    parser.add_argument("--pretrained-ssl",   type=str,   default=None,  help="path to pretrained SSL backbone weights (e.g. runs/simclr_encoder_pretrained.pth)")
    parser.add_argument("--temporal",         action="store_true",        help="enable temporal threat tracking (DetectionTracker) in inference demo")
    parser.add_argument("--yolo-path",        type=str,   default=None,  help="override path to YOLO weights")
    parser.add_argument("--cnn-path",         type=str,   default=None,  help="override path to CNN weights")
    
    args = parser.parse_args()
    
    # If no flags are provided, print help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    print(f"======================================================================")
    print(f"Wild Animal Intrusion Detection System CLI")
    print(f"Working Directory: {workspace_dir}")
    print(f"======================================================================")
    
    # 1. Preprocess
    if args.preprocess:
        print("\n>>> Stage: Preprocessing & Dataset Extraction")
        from src.preprocessing import main as run_prep
        run_prep(workspace_dir,
                 wildboar_zip=args.wildboar_zip,
                 wildboar_class_id=args.wildboar_class_id)
        
    # 2. Train YOLO
    if args.train_yolo:
        print("\n>>> Stage: Training YOLOv8 Detector")
        from src.yolo_training import train_yolo
        epochs = args.epochs if args.epochs else 3
        batch  = args.batch  if args.batch  else 8
        train_yolo(workspace_dir, epochs=epochs, batch=batch, imgsz=args.imgsz,
                   use_tiny=args.tiny, combined=args.combined)
        
    # 3. Crop Generation
    if args.crop_gen:
        print("\n>>> Stage: Generating YOLO Detection Crops")
        from src.crop_generation import generate_crops
        model_path = args.yolo_path if args.yolo_path else find_latest_yolo_weights(workspace_dir)
        # Fallback to pre-trained nano if custom trained weights do not exist yet
        if not os.path.exists(model_path):
            print("Warning: Custom YOLO weights not found. Using pre-trained yolov8n.pt")
            model_path = "yolov8n.pt"
            
        input_dir = args.input_dir if args.input_dir else os.path.join(workspace_dir, "data", "elephant_thermal", "Elephant.v2i.yolov8", "test", "images")
        output_dir = args.output_dir if args.output_dir else os.path.join(workspace_dir, "runs", "generated_crops")
        
        generate_crops(model_path, input_dir, output_dir)
        
    # 4. Train CNN
    if args.train_cnn:
        print("\n>>> Stage: Training CNN Verifier")
        from src.cnn_training import train_cnn
        data_dir = os.path.join(workspace_dir, "data", "cnn_dataset")
        model_path = args.cnn_path if args.cnn_path else os.path.join(workspace_dir, "runs", "cnn_verifier.pth")
        epochs = args.epochs if args.epochs else (1 if args.tiny else 5)
        train_cnn(data_dir, model_path, epochs=epochs,
                  use_attention=args.attention, use_curriculum=args.curriculum,
                  pretrained_ssl_path=args.pretrained_ssl)
        
    # 5. Demo Pipeline
    if args.demo_pipeline:
        print("\n>>> Stage: Combined Pipeline Inference Demo")
        from src.inference import run_pipeline_demo
        yolo_path = args.yolo_path if args.yolo_path else find_latest_yolo_weights(workspace_dir)
        cnn_path = args.cnn_path if args.cnn_path else os.path.join(workspace_dir, "runs", "cnn_verifier.pth")
        
        input_dir = args.input_dir if args.input_dir else os.path.join(workspace_dir, "data", "elephant_thermal", "Elephant.v2i.yolov8", "test", "images")
        output_dir = args.output_dir if args.output_dir else os.path.join(workspace_dir, "runs", "pipeline_alerts")
        
        if not os.path.exists(cnn_path):
            print(f"Error: CNN model {cnn_path} not found. Please train the CNN verifier first using --train-cnn.")
            sys.exit(1)
            
        run_pipeline_demo(yolo_path, cnn_path, input_dir, output_dir, use_temporal=args.temporal)
        
    # 6. Evaluate
    if args.evaluate:
        print("\n>>> Stage: System Evaluation and Metric Reporting")
        from src.evaluation import evaluate_cnn_only, evaluate_pipeline
        yolo_path = args.yolo_path if args.yolo_path else find_latest_yolo_weights(workspace_dir)
        cnn_path = args.cnn_path if args.cnn_path else os.path.join(workspace_dir, "runs", "cnn_verifier.pth")
        
        cnn_data = os.path.join(workspace_dir, "data", "cnn_dataset")
        yolo_test = os.path.join(workspace_dir, "data", "elephant_thermal", "Elephant.v2i.yolov8", "test")
        out_dir = os.path.join(workspace_dir, "runs")
        
        if not os.path.exists(cnn_path):
            print(f"Error: CNN model {cnn_path} not found. Please train the CNN verifier first.")
            sys.exit(1)
            
        evaluate_cnn_only(cnn_path, cnn_data, out_dir)
        evaluate_pipeline(yolo_path, cnn_path, yolo_test, out_dir)
        
    # 7. Explain
    if args.explain:
        print("\n>>> Stage: Visual Explainability (Grad-CAM)")
        from src.explainability import explain_crop
        cnn_path = args.cnn_path if args.cnn_path else os.path.join(workspace_dir, "runs", "cnn_verifier.pth")
        
        if not os.path.exists(cnn_path):
            print(f"Error: CNN model {cnn_path} not found. Please train the CNN verifier first.")
            sys.exit(1)
            
        crop_file = args.crop_file
        if not crop_file or not os.path.exists(crop_file):
            # Try to pick a sample from cnn test dataset
            test_crop_dir = Path(workspace_dir) / "data" / "cnn_dataset" / "test" / "elephant"
            crops = list(test_crop_dir.glob("*.jpg"))
            if crops:
                crop_file = str(crops[0])
                print(f"No valid crop file provided. Selecting default test sample: {crop_file}")
            else:
                print("Error: No test crops available. Run --preprocess first, and ensure crops exist.")
                sys.exit(1)
                
        output_file = args.output_dir if args.output_dir else os.path.join(workspace_dir, "runs", "grad_cam_explanation.jpg")
        explain_crop(cnn_path, crop_file, output_file)
        
    # 8. SSL Pretrain
    if args.ssl_pretrain:
        print("\n>>> Stage: Self-Supervised Pretraining (SimCLR)")
        from src.experimental.ssl_pretraining import train_simclr
        unlabeled_dir = os.path.join(workspace_dir, "data", "serengeti", "train", "blank") # Use blank camera trap images
        epochs = args.epochs if args.epochs else (1 if args.tiny else 3)
        train_simclr(unlabeled_dir, epochs=epochs)
        
    # 9. Compress
    if args.compress:
        print("\n>>> Stage: Model Compression (Pruning & Quantization)")
        from src.experimental.compression import apply_dynamic_quantization, apply_unstructured_pruning
        cnn_path = args.cnn_path if args.cnn_path else os.path.join(workspace_dir, "runs", "cnn_verifier.pth")
        if not os.path.exists(cnn_path):
            print(f"Error: CNN model {cnn_path} not found. Please train the CNN verifier first.")
            sys.exit(1)
            
        quant_out = os.path.join(workspace_dir, "runs", "cnn_verifier_quantized.pth")
        prune_out = os.path.join(workspace_dir, "runs", "cnn_verifier_pruned.pth")
        
        apply_dynamic_quantization(cnn_path, quant_out)
        apply_unstructured_pruning(cnn_path, prune_out, amount=0.3)
        
    print("\nCommand executed successfully.")

if __name__ == "__main__":
    main()
