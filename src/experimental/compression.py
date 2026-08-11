import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from src.cnn_training import WildlifeVerifierCNN

def get_model_size_kb(model_path):
    """Returns the size of a model file in Kilobytes."""
    if not os.path.exists(model_path):
        return 0.0
    return os.path.getsize(model_path) / 1024.0

def apply_dynamic_quantization(model_path, output_path):
    """
    Applies PyTorch post-training dynamic quantization to convert FP32 weights
    in linear layers to INT8.
    """
    print(f"Applying dynamic quantization to {model_path}...")
    
    # Initialize dynamic model based on checkpoint architecture
    checkpoint = torch.load(model_path, map_location='cpu')
    last_key = [k for k in checkpoint.keys() if 'classifier' in k and 'weight' in k]
    num_classes = checkpoint[last_key[-1]].shape[0] if last_key else 2
    
    is_attention = any('attention_head' in k for k in checkpoint.keys())
    if is_attention:
        print("Auto-detected Attention-Enhanced CNN architecture for quantization.")
        from src.experimental.attention_heads import AttentionEnhancedVerifierCNN
        model_fp32 = AttentionEnhancedVerifierCNN(num_classes=num_classes)
    else:
        print("Auto-detected Baseline CNN architecture for quantization.")
        model_fp32 = WildlifeVerifierCNN(num_classes=num_classes)
        
    model_fp32.load_state_dict(checkpoint)
    model_fp32.eval()
    
    # Apply dynamic quantization to Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model_fp32, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), output_path)
    
    size_orig = get_model_size_kb(model_path)
    size_quant = get_model_size_kb(output_path)
    
    print(f"Quantization completed.")
    print(f"  Original size:  {size_orig:.2f} KB")
    print(f"  Quantized size: {size_quant:.2f} KB (Reduction: {(1.0 - size_quant/size_orig)*100:.1f}%)")
    return quantized_model

def apply_unstructured_pruning(model_path, output_path, amount=0.3):
    """
    Prunes a percentage of the connections in each Conv2d layer
    using L1 unstructured weight pruning.
    """
    print(f"Applying L1 unstructured pruning ({amount*100}%) to {model_path}...")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    last_key = [k for k in checkpoint.keys() if 'classifier' in k and 'weight' in k]
    num_classes = checkpoint[last_key[-1]].shape[0] if last_key else 2
    
    is_attention = any('attention_head' in k for k in checkpoint.keys())
    if is_attention:
        print("Auto-detected Attention-Enhanced CNN architecture for pruning.")
        from src.experimental.attention_heads import AttentionEnhancedVerifierCNN
        model = AttentionEnhancedVerifierCNN(num_classes=num_classes)
    else:
        print("Auto-detected Baseline CNN architecture for pruning.")
        model = WildlifeVerifierCNN(num_classes=num_classes)
        
    model.load_state_dict(checkpoint)
    
    # Iterate and apply L1 pruning to Conv2d layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Prune a percentage of weights in the convolutional layer
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Remove pruning handles to make pruning permanent and remove weights structure overhead
            prune.remove(module, 'weight')
            
    # Save pruned model weights
    torch.save(model.state_dict(), output_path)
    
    size_orig = get_model_size_kb(model_path)
    size_pruned = get_model_size_kb(output_path)
    
    print(f"Pruning completed.")
    print(f"  Original size: {size_orig:.2f} KB")
    print(f"  Pruned size:   {size_pruned:.2f} KB")
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runs/cnn_verifier.pth", help="path to CNN model checkpoint")
    parser.add_argument("--quant_out", type=str, default="runs/cnn_verifier_quantized.pth", help="output quantized path")
    parser.add_argument("--prune_out", type=str, default="runs/cnn_verifier_pruned.pth", help="output pruned path")
    parser.add_argument("--prune_amount", type=float, default=0.3, help="percentage of weights to prune")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        # Create a dummy model checkpoint if it doesn't exist to test the script
        print(f"Model checkpoint {args.model} not found. Creating dummy checkpoint to test...")
        os.makedirs(os.path.dirname(args.model), exist_ok=True)
        dummy_model = WildlifeVerifierCNN()
        torch.save(dummy_model.state_dict(), args.model)
        
    apply_dynamic_quantization(args.model, args.quant_out)
    apply_unstructured_pruning(args.model, args.prune_out, amount=args.prune_amount)
