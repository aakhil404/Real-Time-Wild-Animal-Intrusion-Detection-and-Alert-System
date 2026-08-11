import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np

# NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss)
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self._get_correlated_mask().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.similarity_function = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye(2 * self.batch_size, k=self.batch_size)
        l2 = np.eye(2 * self.batch_size, k=-self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask

    def forward(self, zis, zjs):
        device = zis.device
        # Concatenate zis and zjs
        representations = torch.cat([zis, zjs], dim=0)
        
        # Calculate cosine similarity matrix
        similarity_matrix = self.similarity_function(representations.unsqueeze(1), representations.unsqueeze(0))
        
        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Extract positives
        positives = torch.cat([torch.diag(similarity_matrix, self.batch_size), torch.diag(similarity_matrix, -self.batch_size)])
        positives = positives.unsqueeze(1)
        
        # Mask out self-similarity (diagonals and corresponding pairs)
        # Ensure mask size fits the current representation batch
        curr_batch_size = zis.size(0)
        if curr_batch_size != self.batch_size:
            diag = np.eye(2 * curr_batch_size)
            l1 = np.eye(2 * curr_batch_size, k=curr_batch_size)
            l2 = np.eye(2 * curr_batch_size, k=-curr_batch_size)
            mask = torch.from_numpy((diag + l1 + l2)).to(device).type(torch.bool)
            negatives = similarity_matrix[~mask].view(2 * curr_batch_size, -1)
        else:
            negatives = similarity_matrix[self.mask].view(2 * self.batch_size, -1)
            
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * curr_batch_size).to(device).long()
        
        loss = self.criterion(logits, labels)
        return loss / (2 * curr_batch_size)

# Dataset that returns two augmented views of the same image
class ContrastiveDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Apply transform twice to get two different random views
        view1 = self.transform(img)
        view2 = self.transform(img)
        
        return view1, view2

# ResNet-style SimCLR Model
class SimCLRModel(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLRModel, self).__init__()
        # Extract features block from standard base encoder
        self.encoder = base_encoder
        
        # Projection Head (MLP: Linear -> BatchNorm -> ReLU -> Linear)
        self.projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, projection_dim)
        )

    def forward(self, x):
        # Extract encoder features
        h = self.encoder(x)
        # Normalize/project features
        z = self.projector(h)
        return z

def get_ssl_augmentations(img_size=128):
    """Returns typical SimCLR data augmentations for contrastive pretraining."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_simclr(unlabeled_image_dir, epochs=3, batch_size=32, lr=0.0003):
    """Scaffolds SimCLR contrastive training loop on unlabeled images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting SimCLR pretraining on {device}...")
    
    # Collect all image files
    img_dir = Path(unlabeled_image_dir)
    img_files = list(img_dir.glob('**/*.jpg'))
    if not img_files:
        print(f"No images found in {unlabeled_image_dir}. SSL Pretraining cannot run.")
        return
        
    print(f"Found {len(img_files)} unlabeled images for pretraining.")
    
    transform = get_ssl_augmentations()
    dataset = ContrastiveDataset(img_files, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    # Build base encoder using conv features from WildlifeVerifierCNN (up to the global pooling layer)
    from src.cnn_training import WildlifeVerifierCNN
    cnn_base = WildlifeVerifierCNN()
    
    # We define a custom sequential encoder that maps input to raw features before classifier
    base_encoder = nn.Sequential(
        cnn_base.features1,
        cnn_base.features2,
        cnn_base.features3,
        cnn_base.features4,
        cnn_base.global_pool,
        nn.Flatten()
    )
    
    model = SimCLRModel(base_encoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = ContrastiveLoss(batch_size=batch_size)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for idx, (view1, view2) in enumerate(loader):
            view1, view2 = view1.to(device), view2.to(device)
            
            optimizer.zero_grad()
            z1 = model(view1)
            z2 = model(view2)
            
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"SSL Epoch [{epoch+1}/{epochs}] - Contrastive Loss: {epoch_loss/len(loader):.4f}")
        
    # Save encoder weights
    save_path = "runs/simclr_encoder_pretrained.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(base_encoder.state_dict(), save_path)
    print(f"Pretrained encoder weights saved to {save_path}")
    return save_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/serengeti", help="unlabeled image folder")
    parser.add_argument("--epochs", type=int, default=3, help="number of contrastive pretraining epochs")
    args = parser.parse_args()
    
    train_simclr(args.data_dir, epochs=args.epochs)
