import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from src.cnn_training import WildlifeVerifierCNN

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; we want the gradient w.r.t. features output
        self.gradients = grad_output[0]
        
    def generate_cam(self, input_tensor, class_idx=None):
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Extract activations and gradients
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global Average Pooling of gradients (weights)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Compute weighted combination of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        # Apply ReLU to only keep features that positively contribute to the target class
        cam = np.maximum(cam, 0)
        
        # Resize CAM to match input image shape (128x128)
        cam = cv2.resize(cam, (128, 128))
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = cam - cam_min
            
        return cam, class_idx
        
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def overlay_heatmap(img_path, cam, output_path, alpha=0.4):
    """Overlays a CAM heatmap onto the original image crop and saves it."""
    img = cv2.imread(str(img_path))
    if img is None:
        return
        
    img = cv2.resize(img, (128, 128))
    
    # Generate heatmap image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    # Overlay semi-transparent heatmap on original crop
    overlay = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    
    # Save output
    cv2.imwrite(str(output_path), overlay)
    print(f"Explainability Grad-CAM saved to {output_path}")

def explain_crop(cnn_model_path, crop_path, output_path):
    """Generates and saves a Grad-CAM visualization for a specific image crop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model with dynamic class inference and architecture auto-detection
    checkpoint = torch.load(cnn_model_path, map_location=device)
    last_key = [k for k in checkpoint.keys() if 'classifier' in k and 'weight' in k]
    num_classes = checkpoint[last_key[-1]].shape[0] if last_key else 2

    is_attention = any('attention_head' in k for k in checkpoint.keys())
    if is_attention:
        print("Auto-detected Attention-Enhanced CNN architecture for Grad-CAM.")
        from src.experimental.attention_heads import AttentionEnhancedVerifierCNN
        model = AttentionEnhancedVerifierCNN(num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint)
        # Use features4_conv Conv2d layer as Grad-CAM target
        target_layer = model.features4_conv[0]
    else:
        print("Auto-detected Baseline CNN architecture for Grad-CAM.")
        model = WildlifeVerifierCNN(num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint)
        target_layer = model.features4[0]
        
    model.eval()
    
    # Preprocess crop
    img = cv2.imread(str(crop_path))
    if img is None:
        print(f"Could not load crop image: {crop_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    # Run Grad-CAM
    cam_generator = GradCAM(model, target_layer)
    cam, pred_class = cam_generator.generate_cam(input_tensor)
    cam_generator.remove_hooks()
    
    # Overlay and save
    overlay_heatmap(crop_path, cam, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn", type=str, default="runs/cnn_verifier.pth", help="CNN model path")
    parser.add_argument("--crop", type=str, required=True, help="Path to input image crop")
    parser.add_argument("--output", type=str, required=True, help="Path to save visual explanation")
    args = parser.parse_args()
    
    explain_crop(args.cnn, args.crop, args.output)
