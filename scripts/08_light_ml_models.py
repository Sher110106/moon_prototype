#!/usr/bin/env python3
"""
08_light_ml_models.py
Light ML models training for landslide detection and boulder segmentation.

Implements:
- Landslide U-Net with ResNet18 encoder
- YOLOv8 fine-tuning for boulder detection
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import rasterio
import cv2
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LandslideDataset(Dataset):
    """Dataset for landslide segmentation."""
    
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.float()

def dice_loss(pred, target, smooth=1.):
    """Dice loss for segmentation."""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(pred, target):
    """Combined BCE + Dice loss."""
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice

def create_landslide_model():
    """Create U-Net model for landslide segmentation."""
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,  # TMC_cos, slope, curvature
        classes=1,
        activation=None
    )
    return model

def prepare_landslide_data(data_dir):
    """Prepare landslide training data."""
    images = []
    masks = []
    
    # Load preprocessed tiles and corresponding masks
    image_dir = Path(data_dir) / "landslide_tiles"
    mask_dir = Path(data_dir) / "landslide_masks"
    
    for image_path in image_dir.glob("*.tif"):
        mask_path = mask_dir / f"{image_path.stem}_mask.tif"
        
        if mask_path.exists():
            # Load image (3 channels: TMC_cos, slope, curvature)
            with rasterio.open(image_path) as src:
                image = src.read().transpose(1, 2, 0).astype(np.float32)
                # Normalize to 0-1
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
            # Load mask
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.float32)
                mask = (mask > 0).astype(np.float32)
            
            images.append(image)
            masks.append(mask)
    
    return np.array(images), np.array(masks)

def train_landslide_unet(data_dir, output_dir, epochs=40, batch_size=8):
    """Train U-Net for landslide segmentation."""
    print("Training landslide U-Net...")
    
    # Prepare data
    images, masks = prepare_landslide_data(data_dir)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, masks, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    
    # Data augmentation
    train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([ToTensorV2()])
    
    # Create datasets
    train_dataset = LandslideDataset(X_train, y_train, train_transform)
    val_dataset = LandslideDataset(X_val, y_val, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = create_landslide_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_iou = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = combined_loss(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_iou = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += combined_loss(output, target.unsqueeze(1)).item()
                
                # Calculate IoU
                pred = torch.sigmoid(output) > 0.5
                target_bool = target.unsqueeze(1) > 0.5
                intersection = (pred & target_bool).float().sum()
                union = (pred | target_bool).float().sum()
                iou = intersection / (union + 1e-8)
                val_iou += iou.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val IoU: {avg_val_iou:.4f}")
        
        # Early stopping and best model saving
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), f"{output_dir}/best_landslide_unet.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
    
    print(f"Best validation IoU: {best_val_iou:.4f}")
    return model

def prepare_yolo_data(data_dir):
    """Prepare YOLO dataset configuration."""
    # Create YOLO dataset YAML
    yaml_content = f"""
path: {data_dir}/boulder_dataset
train: images/train
val: images/val
test: images/test

nc: 1
names: ['boulder']
"""
    
    yaml_path = f"{data_dir}/boulder.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path

def train_yolo_boulder(data_dir, output_dir, epochs=15, batch_size=8):
    """Train YOLOv8 for boulder detection."""
    print("Training YOLOv8 for boulder detection...")
    
    # Prepare YOLO data configuration
    yaml_path = prepare_yolo_data(data_dir)
    
    # Load pre-trained YOLOv8 nano segmentation model
    model = YOLO('yolov8n-seg.pt')
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=1024,
        batch=batch_size,
        lr0=1e-4,
        box=7.5,  # box loss gain
        cls=0.5,  # cls loss gain
        dfl=1.5,  # dfl loss gain
        hsv_h=0.02,  # HSV-Hue augmentation
        hsv_s=0.2,   # HSV-Saturation augmentation
        hsv_v=0.2,   # HSV-Value augmentation
        degrees=10.0,  # rotation augmentation
        translate=0.1,  # translation augmentation
        scale=0.2,     # scaling augmentation
        shear=2.0,     # shear augmentation
        flipud=0.5,    # vertical flip augmentation
        fliplr=0.5,    # horizontal flip augmentation
        mosaic=0.5,    # mosaic augmentation
        project=output_dir,
        name='yolo_boulder',
        save_period=5,
        patience=10,
        device='0' if torch.cuda.is_available() else 'cpu'
    )
    
    # Save best model
    best_model_path = f"{output_dir}/yolo_boulder/weights/best.pt"
    print(f"Best YOLO model saved to: {best_model_path}")
    
    return model, results

def evaluate_models(data_dir, model_dir):
    """Evaluate trained models on test set."""
    print("Evaluating models...")
    
    # Load test data for U-Net evaluation
    images, masks = prepare_landslide_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        images, masks, test_size=0.3, random_state=42
    )
    _, X_test, _, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    
    # Load trained U-Net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet_model = create_landslide_model()
    unet_model.load_state_dict(torch.load(f"{model_dir}/best_landslide_unet.pth"))
    unet_model.to(device)
    unet_model.eval()
    
    # Evaluate U-Net
    test_dataset = LandslideDataset(X_test, y_test, A.Compose([ToTensorV2()]))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    total_iou = 0
    total_precision = 0
    total_recall = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = torch.sigmoid(unet_model(data))
            
            pred = (output > 0.5).float()
            target = target.unsqueeze(1)
            
            # IoU
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum() - intersection
            iou = intersection / (union + 1e-8)
            total_iou += iou.item()
            
            # Precision
            tp = (pred * target).sum()
            precision = tp / (pred.sum() + 1e-8)
            total_precision += precision.item()
            
            # Recall
            recall = tp / (target.sum() + 1e-8)
            total_recall += recall.item()
    
    avg_iou = total_iou / len(test_loader)
    avg_precision = total_precision / len(test_loader)
    avg_recall = total_recall / len(test_loader)
    
    print(f"U-Net Results:")
    print(f"  IoU: {avg_iou:.4f}")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall: {avg_recall:.4f}")
    
    # Load and evaluate YOLO
    yolo_model = YOLO(f"{model_dir}/yolo_boulder/weights/best.pt")
    
    # Validate YOLO model
    yolo_results = yolo_model.val()
    
    print(f"YOLO Results:")
    print(f"  mAP50: {yolo_results.box.map50:.4f}")
    print(f"  mAP50-95: {yolo_results.box.map:.4f}")
    
    return {
        'unet_iou': avg_iou,
        'unet_precision': avg_precision,
        'unet_recall': avg_recall,
        'yolo_map50': yolo_results.box.map50,
        'yolo_map': yolo_results.box.map
    }

def main():
    """Main training pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python 08_light_ml_models.py <data_dir> [output_dir]")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs/models"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print("Starting ML model training...")
    
    # Train landslide U-Net
    unet_model = train_landslide_unet(data_dir, output_dir)
    
    # Train boulder YOLO
    yolo_model, yolo_results = train_yolo_boulder(data_dir, output_dir)
    
    # Evaluate models
    metrics = evaluate_models(data_dir, output_dir)
    
    # Save metrics
    import json
    with open(f"{output_dir}/training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("ML model training completed!")
    print(f"Models saved to: {output_dir}")
    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()