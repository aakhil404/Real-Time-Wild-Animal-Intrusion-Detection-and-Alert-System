import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Define custom CNN architecture
class WildlifeVerifierCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(WildlifeVerifierCNN, self).__init__()
        # Block 1: Input 3x128x128 -> Out 32x64x64
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Block 2: 32x64x64 -> Out 64x32x32
        self.features2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Block 3: 64x32x32 -> Out 128x16x16
        self.features3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Block 4: 128x16x16 -> Out 256x8x8
        self.features4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
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
        x = self.features4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

def get_data_loaders(data_dir, batch_size=32):
    """Creates PyTorch training and validation loaders with controlled data augmentation."""
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # Controlled Augmentations: brightness, slight shift/rotation, and occlusion mask (RandomErasing)
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # RandomErasing acts as an occlusion mask simulator
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    # Use num_workers=0 on Windows to avoid multiprocessing issues in simple environments
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, train_dataset.classes

def load_ssl_backbone(model, ssl_path, use_attention=False):
    """Loads SimCLR pretrained sequential encoder weights into the verifier model backbone."""
    print(f"Loading pretrained SSL weights from {ssl_path}...")
    try:
        ssl_state_dict = torch.load(ssl_path, map_location='cpu')
        model_state_dict = model.state_dict()
        
        mapped_state_dict = {}
        mismatches = 0
        
        for key, val in ssl_state_dict.items():
            parts = key.split('.')
            if len(parts) < 2:
                continue
                
            block_idx = parts[0]
            sub_key = ".".join(parts[1:])
            
            if block_idx == '0':
                target_key = f"features1.{sub_key}"
            elif block_idx == '1':
                target_key = f"features2.{sub_key}"
            elif block_idx == '2':
                target_key = f"features3.{sub_key}"
            elif block_idx == '3':
                if use_attention:
                    target_key = f"features4_conv.{sub_key}"
                else:
                    target_key = f"features4.{sub_key}"
            else:
                continue
                
            if target_key in model_state_dict:
                mapped_state_dict[target_key] = val
            else:
                mismatches += 1
                
        model_state_dict.update(mapped_state_dict)
        model.load_state_dict(model_state_dict)
        print(f"SSL backbone initialization completed. Loaded {len(mapped_state_dict)} tensors. (Skipped/mismatched: {mismatches})")
    except Exception as e:
        print(f"Warning: Failed to load SSL backbone: {e}")

def train_cnn(data_dir, model_save_path, epochs=10, batch_size=32, lr=0.001,
              use_attention=False, use_curriculum=False, pretrained_ssl_path=None):
    """Main training loop for CNN verifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CNN verifier on device: {device}")
    
    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size)
    print(f"Dataset classes: {class_names}")
    
    curr_dataset = None
    if use_curriculum:
        from src.experimental.curriculum_learning import get_curriculum_dataloader
        train_dataset = train_loader.dataset
        train_loader, curr_dataset = get_curriculum_dataloader(train_dataset, batch_size=batch_size)
    
    if use_attention:
        from src.experimental.attention_heads import AttentionEnhancedVerifierCNN
        model = AttentionEnhancedVerifierCNN(num_classes=len(class_names)).to(device)
    else:
        model = WildlifeVerifierCNN(num_classes=len(class_names)).to(device)
        
    if pretrained_ssl_path and os.path.exists(pretrained_ssl_path):
        load_ssl_backbone(model, pretrained_ssl_path, use_attention=use_attention)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation stage
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # Step curriculum learning difficulty if enabled
        if use_curriculum and curr_dataset is not None:
            curr_dataset.step_epoch()
            
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Saved new best model checkpoint with Val Acc: {best_val_acc:.4f}")
            
    # Save training curves plot
    plot_curves(history, os.path.dirname(model_save_path))
    print("CNN Training completed.")
    return model_save_path

def plot_curves(history, output_dir):
    """Saves plots of training history loss and accuracy."""
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'cnn_training_history.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training curves to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/cnn_dataset", help="path to CNN crops dataset")
    parser.add_argument("--model_path", type=str, default="runs/cnn_verifier.pth", help="where to save the trained model")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    train_cnn(args.data_dir, args.model_path, epochs=args.epochs, lr=args.lr)
