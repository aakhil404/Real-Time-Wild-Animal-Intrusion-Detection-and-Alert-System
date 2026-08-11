import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        """
        Implements a spatial self-attention block for 2D feature maps.
        Allows CNN models to model long-range global context.
        """
        super(SpatialSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        
        # Verify channels is divisible by num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        # Multi-head attention layer
        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        
        # Layer Normalization for stability
        self.norm = nn.LayerNorm(in_channels)
        
        # Projection back to channels
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Input x: Tensor of shape (batch, channels, height, width)
        Output: Tensor of shape (batch, channels, height, width)
        """
        b, c, h, w = x.size()
        
        # Flatten spatial dimensions: (b, c, h, w) -> (b, c, h*w)
        # Transpose for MHA: (b, c, h*w) -> (b, h*w, c)
        flat_x = x.view(b, c, h * w).transpose(1, 2)
        
        # Apply layer norm
        norm_x = self.norm(flat_x)
        
        # Self-attention: query=norm_x, key=norm_x, value=norm_x
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        
        # Residual connection
        flat_out = flat_x + attn_out
        
        # Reshape back to spatial feature map: (b, h*w, c) -> (b, c, h*w) -> (b, c, h, w)
        spatial_out = flat_out.transpose(1, 2).view(b, c, h, w)
        
        # Projection layer
        out = x + self.proj(spatial_out)
        return out

class AttentionEnhancedVerifierCNN(nn.Module):
    def __init__(self, num_classes=2):
        """
        An attention-enhanced version of WildlifeVerifierCNN.
        Replaces the standard final conv block with a hybrid Conv-Attention block.
        """
        super(AttentionEnhancedVerifierCNN, self).__init__()
        
        # Base convolutions (identical to standard verifier)
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Feature block 4 augmented with Spatial self-attention
        self.features4_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 256-channel self-attention head
        self.attention_head = SpatialSelfAttention(in_channels=256, num_heads=4)
        self.features4_pool = nn.MaxPool2d(2, 2)
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        
        # Conv feature extraction + Attention enhancement
        x = self.features4_conv(x)
        x = self.attention_head(x)
        x = self.features4_pool(x)
        
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # Test block shape
    inputs = torch.randn(2, 3, 128, 128)
    model = AttentionEnhancedVerifierCNN()
    outputs = model(inputs)
    print("Self-Attention test output shape:", outputs.shape)
