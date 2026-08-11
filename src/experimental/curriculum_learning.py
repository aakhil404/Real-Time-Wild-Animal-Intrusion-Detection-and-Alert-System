import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

def estimate_blur(image_path):
    """Calculates laplacian variance as a metric for image sharpness/blur (higher = sharper)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()

def get_crop_area(image_path):
    """Returns the pixel area of the crop image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return 0
    h, w, _ = img.shape
    return h * w

class CurriculumWildlifeDataset(Dataset):
    def __init__(self, cnn_dataset, start_percent=0.3, pacing_rate=0.15):
        """
        Wraps a PyTorch ImageFolder dataset to introduce curriculum learning.
        Ranks samples from easiest to hardest based on crop size and sharpness.
        """
        self.dataset = cnn_dataset
        self.pacing_rate = pacing_rate
        self.active_percent = start_percent
        self.epoch = 0
        
        # Calculate difficulty for each sample
        print("Scoring dataset difficulty for curriculum learning...")
        self.samples_with_difficulty = []
        for idx, (img_path, class_idx) in enumerate(self.dataset.samples):
            # Easiest images: Large, sharp, well-lit crops.
            # Hardest images: Small, blurry, occluded crops.
            area = get_crop_area(img_path)
            sharpness = estimate_blur(img_path)
            
            # Difficulty score: inverse of area and sharpness
            # We add epsilon to prevent division by zero
            difficulty = (1e6 / (area + 1)) + (1e3 / (sharpness + 1e-3))
            self.samples_with_difficulty.append((idx, img_path, class_idx, difficulty))
            
        # Sort samples by difficulty score ascending (easiest first)
        self.samples_with_difficulty.sort(key=lambda x: x[3])
        print(f"Curriculum sorted: Easiest score = {self.samples_with_difficulty[0][3]:.2f}, "
              f"Hardest score = {self.samples_with_difficulty[-1][3]:.2f}")
              
        self.update_active_subset()

    def update_active_subset(self):
        """Updates the active subset of training samples based on current epoch pacing."""
        # Calculate length based on current percent window
        self.active_length = int(len(self.samples_with_difficulty) * self.active_percent)
        self.active_length = min(self.active_length, len(self.samples_with_difficulty))
        self.active_length = max(self.active_length, 1)
        
        # Select active sample slice
        self.active_samples = self.samples_with_difficulty[:self.active_length]
        print(f"Curriculum Epoch {self.epoch}: Active subset size = {self.active_length}/{len(self.samples_with_difficulty)} ({self.active_percent * 100:.1f}%)")

    def step_epoch(self):
        """Advances the curriculum pacing, adding harder samples for the next epoch."""
        self.epoch += 1
        if self.active_percent < 1.0:
            self.active_percent += self.pacing_rate
            self.active_percent = min(self.active_percent, 1.0)
            self.update_active_subset()

    def __len__(self):
        return self.active_length

    def __getitem__(self, idx):
        # Map active subset index back to the underlying dataset sample
        original_idx, img_path, class_idx, _ = self.active_samples[idx]
        # Return the transformed image and label from the base dataset
        return self.dataset[original_idx]

def get_curriculum_dataloader(base_dataset, batch_size=32):
    """Wraps a base ImageFolder dataset into a Curriculum dataset and returns a DataLoader."""
    curr_dataset = CurriculumWildlifeDataset(base_dataset)
    # Shuffle is true within the active easy-to-hard subset
    loader = DataLoader(curr_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader, curr_dataset
