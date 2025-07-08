"""
Advanced Training System with Optuna Optimization
Handles dataset preparation, training, and hyperparameter optimization for attribute classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import wandb

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import os
import random
from collections import defaultdict, Counter
import shutil

from attribute_classifier import (
    MultiTaskAttributeNet, Gender, ShirtColor, HairColor,
    PersonAttributes, AttributeClassifier
)


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    # Data paths
    dataset_path: str = "data/attribute_dataset"
    model_save_path: str = "data/models"
    results_save_path: str = "results"
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    
    # Data parameters
    image_size: int = 224
    num_workers: int = 4
    val_split: float = 0.2
    test_split: float = 0.1
    
    # Model parameters
    backbone: str = "efficientnet_b0"
    dropout: float = 0.3
    
    # Loss weighting for multi-task learning
    gender_weight: float = 1.0
    shirt_weight: float = 1.0
    hair_weight: float = 1.0
    adaptive_weights: bool = True
    
    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "plateau"  # plateau, cosine, none
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_strength: float = 0.5
    
    # Hardware and performance
    device: str = "auto"
    mixed_precision: bool = True
    
    # Experiment tracking
    use_wandb: bool = False
    project_name: str = "cctv-attribute-classification"
    experiment_name: Optional[str] = None
    
    # Early stopping and checkpointing
    save_best_only: bool = True
    save_every_n_epochs: int = 10
    monitor_metric: str = "val_loss"  # val_loss, val_accuracy


class PersonAttributeDataset(Dataset):
    """
    Dataset for person attribute classification
    Expected directory structure:
    dataset_path/
    ├── images/
    │   ├── person_001.jpg
    │   ├── person_002.jpg
    │   └── ...
    ├── annotations.json  # or annotations.csv
    """
    
    def __init__(self, 
                 dataset_path: Union[str, Path],
                 split: str = "train",
                 transforms: Optional[A.Compose] = None,
                 image_size: int = 224):
        
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_size = image_size
        self.transforms = transforms
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Filter by split
        if split != "all":
            self.annotations = [ann for ann in self.annotations if ann.get('split', 'train') == split]
        
        if len(self.annotations) == 0:
            raise ValueError(f"No annotations found for split '{split}' in {dataset_path}")
        
        print(f"📊 Loaded {len(self.annotations)} samples for {split} split")
        
        # Create label mappings
        self.gender_to_idx = {g.value: i for i, g in enumerate(Gender) if g != Gender.UNKNOWN}
        self.shirt_to_idx = {s.value: i for i, s in enumerate(ShirtColor)}
        self.hair_to_idx = {h.value: i for i, h in enumerate(HairColor)}
        
        self.idx_to_gender = {v: k for k, v in self.gender_to_idx.items()}
        self.idx_to_shirt = {v: k for k, v in self.shirt_to_idx.items()}
        self.idx_to_hair = {v: k for k, v in self.hair_to_idx.items()}
        
        # Print class distribution
        self._print_class_distribution()
    
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from JSON or CSV file"""
        json_path = self.dataset_path / "annotations.json"
        csv_path = self.dataset_path / "annotations.csv"
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data if isinstance(data, list) else data.get('annotations', [])
        
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
            return df.to_dict('records')
        
        else:
            raise FileNotFoundError(f"No annotations found in {self.dataset_path}. Expected 'annotations.json' or 'annotations.csv'")
    
    def _print_class_distribution(self):
        """Print class distribution for this split"""
        gender_counts = Counter()
        shirt_counts = Counter()
        hair_counts = Counter()
        
        for ann in self.annotations:
            gender_counts[ann.get('gender', 'unknown')] += 1
            shirt_counts[ann.get('shirt_color', 'unknown')] += 1
            hair_counts[ann.get('hair_color', 'unknown')] += 1
        
        print(f"📈 Class distribution for {self.split}:")
        print(f"   Gender: {dict(gender_counts)}")
        print(f"   Shirt: {dict(shirt_counts)}")
        print(f"   Hair: {dict(hair_counts)}")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Load image
        image_name = annotation['image_name']
        image_path = self.dataset_path / "images" / image_name
        
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"⚠️ Error loading image {image_name}: {e}")
            # Create dummy image
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transforms:
            try:
                transformed = self.transforms(image=image)
                image = transformed['image']
            except Exception as e:
                print(f"⚠️ Transform error for {image_name}: {e}")
                # Fallback to basic resize and normalize
                image = cv2.resize(image, (self.image_size, self.image_size))
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            # Basic resize and normalize
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Get labels
        gender_label = self.gender_to_idx.get(annotation.get('gender', 'unknown'), 0)
        shirt_label = self.shirt_to_idx.get(annotation.get('shirt_color', 'unknown'), len(self.shirt_to_idx) - 1)
        hair_label = self.hair_to_idx.get(annotation.get('hair_color', 'unknown'), len(self.hair_to_idx) - 1)
        
        return {
            'image': image,
            'gender': torch.tensor(gender_label, dtype=torch.long),
            'shirt_color': torch.tensor(shirt_label, dtype=torch.long),
            'hair_color': torch.tensor(hair_label, dtype=torch.long),
            'image_name': image_name
        }
    
    def get_class_weights(self):
        """Calculate class weights for balanced training"""
        gender_counts = Counter()
        shirt_counts = Counter()
        hair_counts = Counter()
        
        for ann in self.annotations:
            gender_counts[ann.get('gender', 'unknown')] += 1
            shirt_counts[ann.get('shirt_color', 'unknown')] += 1
            hair_counts[ann.get('hair_color', 'unknown')] += 1
        
        def calculate_weights(counts, total):
            weights = {}
            for cls, count in counts.items():
                weights[cls] = total / (len(counts) * count) if count > 0 else 1.0
            return weights
        
        total_samples = len(self.annotations)
        
        return {
            'gender': calculate_weights(gender_counts, total_samples),
            'shirt_color': calculate_weights(shirt_counts, total_samples),
            'hair_color': calculate_weights(hair_counts, total_samples)
        }


class AugmentationFactory:
    """Factory for creating augmentation pipelines"""
    
    @staticmethod
    def get_train_transforms(image_size: int = 224, strength: float = 0.5) -> A.Compose:
        """Get training augmentations"""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=int(15 * strength), p=0.7),
            A.ColorJitter(
                brightness=0.2 * strength,
                contrast=0.2 * strength,
                saturation=0.2 * strength,
                hue=0.1 * strength,
                p=0.7
            ),
            A.GaussNoise(var_limit=(10.0 * strength, 30.0 * strength), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.CLAHE(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_val_transforms(image_size: int = 224) -> A.Compose:
        """Get validation/test augmentations"""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class MultiTaskLoss(nn.Module):
    """Multi-task loss with adaptive weighting"""
    
    def __init__(self, 
                 gender_weight: float = 1.0,
                 shirt_weight: float = 1.0,
                 hair_weight: float = 1.0,
                 adaptive_weights: bool = True):
        super().__init__()
        
        self.gender_criterion = nn.CrossEntropyLoss()
        self.shirt_criterion = nn.CrossEntropyLoss()
        self.hair_criterion = nn.CrossEntropyLoss()
        
        self.adaptive_weights = adaptive_weights
        
        if adaptive_weights:
            # Learnable task weights (homoscedastic uncertainty)
            self.log_vars = nn.Parameter(torch.zeros(3))
        else:
            # Fixed weights
            self.register_buffer('weights', torch.tensor([gender_weight, shirt_weight, hair_weight]))
    
    def forward(self, outputs, targets):
        gender_loss = self.gender_criterion(outputs['gender'], targets['gender'])
        shirt_loss = self.shirt_criterion(outputs['shirt_color'], targets['shirt_color'])
        hair_loss = self.hair_criterion(outputs['hair_color'], targets['hair_color'])
        
        losses = torch.stack([gender_loss, shirt_loss, hair_loss])
        
        if self.adaptive_weights:
            # Adaptive weighting based on homoscedastic uncertainty
            precision = torch.exp(-self.log_vars)
            weighted_losses = precision * losses + self.log_vars
            total_loss = weighted_losses.sum()
            task_weights = precision
        else:
            # Fixed weighting
            total_loss = (self.weights * losses).sum()
            task_weights = self.weights
        
        return {
            'total_loss': total_loss,
            'gender_loss': gender_loss,
            'shirt_loss': shirt_loss,
            'hair_loss': hair_loss,
            'task_weights': task_weights
        }


class AttributeTrainer:
    """Main trainer class with comprehensive training pipeline"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        print(f"🔧 Using device: {self.device}")
        
        # Create directories
        self.model_save_path = Path(config.model_save_path)
        self.results_save_path = Path(config.results_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.results_save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        self.training_history = defaultdict(list)
        self.current_epoch = 0
        
        # Initialize wandb if requested
        if config.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        try:
            wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=asdict(self.config)
            )
            print("✅ Weights & Biases initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize wandb: {e}")
            self.config.use_wandb = False
    
    def prepare_datasets(self):
        """Prepare train/val/test datasets"""
        print("📊 Preparing datasets...")
        
        # Create augmentation transforms
        if self.config.use_augmentation:
            train_transforms = AugmentationFactory.get_train_transforms(
                self.config.image_size, self.config.augmentation_strength
            )
        else:
            train_transforms = AugmentationFactory.get_val_transforms(self.config.image_size)
        
        val_transforms = AugmentationFactory.get_val_transforms(self.config.image_size)
        
        try:
            # Load datasets
            self.train_dataset = PersonAttributeDataset(
                self.config.dataset_path, "train", train_transforms, self.config.image_size
            )
            
            self.val_dataset = PersonAttributeDataset(
                self.config.dataset_path, "val", val_transforms, self.config.image_size
            )
            
            # Try to load test dataset
            try:
                self.test_dataset = PersonAttributeDataset(
                    self.config.dataset_path, "test", val_transforms, self.config.image_size
                )
            except:
                print("⚠️ No test split found, using validation for final evaluation")
                self.test_dataset = self.val_dataset
        
        except Exception as e:
            print(f"❌ Failed to load datasets: {e}")
            print("💡 Make sure your dataset follows the expected structure:")
            print("   dataset_path/")
            print("   ├── images/")
            print("   │   ├── person_001.jpg")
            print("   │   └── ...")
            print("   └── annotations.json")
            raise
    
    def create_data_loaders(self, batch_size: Optional[int] = None):
        """Create data loaders"""
        batch_size = batch_size or self.config.batch_size
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        print(f"📦 Data loaders created:")
        print(f"   Train: {len(self.train_loader)} batches")
        print(f"   Val: {len(self.val_loader)} batches")
        print(f"   Test: {len(self.test_loader)} batches")
    
    def create_model(self, trial: Optional[optuna.Trial] = None):
        """Create model with optional hyperparameter optimization"""
        
        if trial:
            # Optuna hyperparameter suggestions
            backbone = trial.suggest_categorical('backbone', ['efficientnet_b0', 'resnet50', 'resnet18'])
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
        else:
            backbone = self.config.backbone
            dropout = self.config.dropout
        
        self.model = MultiTaskAttributeNet(
            backbone=backbone,
            num_genders=len(Gender) - 1,  # Exclude UNKNOWN
            num_shirt_colors=len(ShirtColor),
            num_hair_colors=len(HairColor),
            dropout=dropout
        ).to(self.device)
        
        # Multi-task loss
        if trial:
            adaptive_weights = trial.suggest_categorical('adaptive_weights', [True, False])
        else:
            adaptive_weights = self.config.adaptive_weights
        
        self.criterion = MultiTaskLoss(
            gender_weight=self.config.gender_weight,
            shirt_weight=self.config.shirt_weight,
            hair_weight=self.config.hair_weight,
            adaptive_weights=adaptive_weights
        ).to(self.device)
        
        print(f"🏗️ Model created: {backbone}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_optimizer(self, trial: Optional[optuna.Trial] = None):
        """Create optimizer and scheduler"""
        
        if trial:
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            optimizer_name = trial.suggest_categorical('optimizer', ['adamw', 'adam', 'sgd'])
        else:
            lr = self.config.learning_rate
            weight_decay = self.config.weight_decay
            optimizer_name = self.config.optimizer
        
        # Create optimizer
        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        
        # Create scheduler
        if self.config.scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5, verbose=True)
        elif self.config.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        else:
            self.scheduler = None
        
        # Mixed precision scaler
        if self.config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        print(f"⚙️ Optimizer: {optimizer_name}, LR: {lr:.2e}, Weight Decay: {weight_decay:.2e}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        gender_loss_sum = 0.0
        shirt_loss_sum = 0.0
        hair_loss_sum = 0.0
        
        gender_correct = 0
        shirt_correct = 0
        hair_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            targets = {
                'gender': batch['gender'].to(self.device),
                'shirt_color': batch['shirt_color'].to(self.device),
                'hair_color': batch['hair_color'].to(self.device)
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
            
            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(loss_dict['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total_loss'].backward()
                self.optimizer.step()
            
            # Statistics
            batch_size = images.size(0)
            total_loss += loss_dict['total_loss'].item()
            gender_loss_sum += loss_dict['gender_loss'].item()
            shirt_loss_sum += loss_dict['shirt_loss'].item()
            hair_loss_sum += loss_dict['hair_loss'].item()
            
            # Accuracy calculation
            gender_pred = torch.argmax(outputs['gender'], dim=1)
            shirt_pred = torch.argmax(outputs['shirt_color'], dim=1)
            hair_pred = torch.argmax(outputs['hair_color'], dim=1)
            
            gender_correct += (gender_pred == targets['gender']).sum().item()
            shirt_correct += (shirt_pred == targets['shirt_color']).sum().item()
            hair_correct += (hair_pred == targets['hair_color']).sum().item()
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss_dict['total_loss'].item():.4f}",
                'G_Acc': f"{gender_correct/total_samples:.3f}",
                'S_Acc': f"{shirt_correct/total_samples:.3f}",
                'H_Acc': f"{hair_correct/total_samples:.3f}"
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        gender_acc = gender_correct / total_samples
        shirt_acc = shirt_correct / total_samples
        hair_acc = hair_correct / total_samples
        overall_acc = (gender_correct + shirt_correct + hair_correct) / (3 * total_samples)
        
        return {
            'train_loss': avg_loss,
            'train_gender_loss': gender_loss_sum / len(self.train_loader),
            'train_shirt_loss': shirt_loss_sum / len(self.train_loader),
            'train_hair_loss': hair_loss_sum / len(self.train_loader),
            'train_gender_acc': gender_acc,
            'train_shirt_acc': shirt_acc,
            'train_hair_acc': hair_acc,
            'train_overall_acc': overall_acc
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        gender_loss_sum = 0.0
        shirt_loss_sum = 0.0
        hair_loss_sum = 0.0
        
        gender_correct = 0
        shirt_correct = 0
        hair_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                targets = {
                    'gender': batch['gender'].to(self.device),
                    'shirt_color': batch['shirt_color'].to(self.device),
                    'hair_color': batch['hair_color'].to(self.device)
                }
                
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                
                # Statistics
                batch_size = images.size(0)
                total_loss += loss_dict['total_loss'].item()
                gender_loss_sum += loss_dict['gender_loss'].item()
                shirt_loss_sum += loss_dict['shirt_loss'].item()
                hair_loss_sum += loss_dict['hair_loss'].item()
                
                # Accuracy calculation
                gender_pred = torch.argmax(outputs['gender'], dim=1)
                shirt_pred = torch.argmax(outputs['shirt_color'], dim=1)
                hair_pred = torch.argmax(outputs['hair_color'], dim=1)
                
                gender_correct += (gender_pred == targets['gender']).sum().item()
                shirt_correct += (shirt_pred == targets['shirt_color']).sum().item()
                hair_correct += (hair_pred == targets['hair_color']).sum().item()
                total_samples += batch_size
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        gender_acc = gender_correct / total_samples
        shirt_acc = shirt_correct / total_samples
        hair_acc = hair_correct / total_samples
        overall_acc = (gender_correct + shirt_correct + hair_correct) / (3 * total_samples)
        
        return {
            'val_loss': avg_loss,
            'val_gender_loss': gender_loss_sum / len(self.val_loader),
            'val_shirt_loss': shirt_loss_sum / len(self.val_loader),
            'val_hair_loss': hair_loss_sum / len(self.val_loader),
            'val_gender_acc': gender_acc,
            'val_shirt_acc': shirt_acc,
            'val_hair_acc': hair_acc,
            'val_overall_acc': overall_acc
        }
    
    def train(self, trial: Optional[optuna.Trial] = None) -> float:
        """Main training loop"""
        
        # Prepare everything
        if self.train_dataset is None:
            self.prepare_datasets()
        
        # Create data loaders
        batch_size = trial.suggest_int('batch_size', 16, 64, step=16) if trial else None
        self.create_data_loaders(batch_size)
        
        # Create model and optimizer
        self.create_model(trial)
        self.create_optimizer(trial)
        
        print(f"🚀 Starting training for {self.config.epochs} epochs...")
        
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Check for best model
            monitor_value = val_metrics[self.config.monitor_metric.replace('val_', 'val_')]
            is_best = False
            
            if self.config.monitor_metric == 'val_loss':
                if monitor_value < self.best_val_loss:
                    self.best_val_loss = monitor_value
                    is_best = True
            else:  # accuracy metric
                if monitor_value > self.best_val_accuracy:
                    self.best_val_accuracy = monitor_value
                    is_best = True
            
            if is_best:
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if self.config.save_every_n_epochs > 0 and (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, is_best=False)
            
            # Log metrics
            if self.config.use_wandb:
                wandb.log(epoch_metrics)
            
            # Store history
            for key, value in epoch_metrics.items():
                self.training_history[key].append(value)
            
            # Print epoch summary
            print(f"Epoch {epoch+1:3d}: "
                  f"Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Val Acc: G={val_metrics['val_gender_acc']:.3f} "
                  f"S={val_metrics['val_shirt_acc']:.3f} "
                  f"H={val_metrics['val_hair_acc']:.3f} "
                  f"Overall={val_metrics['val_overall_acc']:.3f}")
            
            # Optuna pruning
            if trial:
                trial.report(val_metrics['val_loss'], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Early stopping
            if patience_counter >= self.config.patience:
                print(f"⏰ Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Save final model
        self._save_checkpoint(self.current_epoch, is_best=True, is_final=True)
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f}s")
        
        return self.best_val_loss
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': asdict(self.config),
            'training_history': dict(self.training_history),
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy
        }
        
        if is_final:
            filename = f'final_model_epoch_{epoch+1}.pth'
        elif is_best:
            filename = 'best_model.pth'
        else:
            filename = f'checkpoint_epoch_{epoch+1}.pth'
        
        save_path = self.model_save_path / filename
        torch.save(checkpoint, save_path)
        
        if is_best or is_final:
            print(f"💾 {'Best' if is_best else 'Final'} model saved: {save_path}")
    
    def evaluate_on_test(self) -> Dict[str, Any]:
        """Evaluate model on test set"""
        print("🧪 Evaluating on test set...")
        
        self.model.eval()
        
        all_gender_preds = []
        all_gender_targets = []
        all_shirt_preds = []
        all_shirt_targets = []
        all_hair_preds = []
        all_hair_targets = []
        
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                images = batch['image'].to(self.device)
                targets = {
                    'gender': batch['gender'].to(self.device),
                    'shirt_color': batch['shirt_color'].to(self.device),
                    'hair_color': batch['hair_color'].to(self.device)
                }
                
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                test_loss += loss_dict['total_loss'].item()
                
                # Predictions
                gender_pred = torch.argmax(outputs['gender'], dim=1)
                shirt_pred = torch.argmax(outputs['shirt_color'], dim=1)
                hair_pred = torch.argmax(outputs['hair_color'], dim=1)
                
                # Store for metrics calculation
                all_gender_preds.extend(gender_pred.cpu().numpy())
                all_gender_targets.extend(targets['gender'].cpu().numpy())
                all_shirt_preds.extend(shirt_pred.cpu().numpy())
                all_shirt_targets.extend(targets['shirt_color'].cpu().numpy())
                all_hair_preds.extend(hair_pred.cpu().numpy())
                all_hair_targets.extend(targets['hair_color'].cpu().numpy())
        
        # Calculate metrics
        test_results = {
            'test_loss': test_loss / len(self.test_loader),
            'gender_accuracy': accuracy_score(all_gender_targets, all_gender_preds),
            'shirt_accuracy': accuracy_score(all_shirt_targets, all_shirt_preds),
            'hair_accuracy': accuracy_score(all_hair_targets, all_hair_preds),
            'overall_accuracy': (
                accuracy_score(all_gender_targets, all_gender_preds) +
                accuracy_score(all_shirt_targets, all_shirt_preds) +
                accuracy_score(all_hair_targets, all_hair_preds)
            ) / 3
        }
        
        print("📊 Test Results:")
        for key, value in test_results.items():
            print(f"   {key}: {value:.4f}")
        
        return test_results
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.training_history:
            print("⚠️ No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.training_history['train_loss'], label='Train')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Overall accuracy plot
        axes[0, 1].plot(self.training_history['train_overall_acc'], label='Train')
        axes[0, 1].plot(self.training_history['val_overall_acc'], label='Validation')
        axes[0, 1].set_title('Overall Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Gender accuracy plot
        axes[1, 0].plot(self.training_history['train_gender_acc'], label='Train')
        axes[1, 0].plot(self.training_history['val_gender_acc'], label='Validation')
        axes[1, 0].set_title('Gender Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Shirt accuracy plot
        axes[1, 1].plot(self.training_history['train_shirt_acc'], label='Train')
        axes[1, 1].plot(self.training_history['val_shirt_acc'], label='Validation')
        axes[1, 1].set_title('Shirt Color Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_save_path / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Training history plot saved: {plot_path}")


class OptunaOptimizer:
    """Optuna-based hyperparameter optimization"""
    
    def __init__(self, config: TrainingConfig, n_trials: int = 50):
        self.config = config
        self.n_trials = n_trials
        
        # Create study
        self.study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function"""
        
        # Create trainer with trial
        trainer = AttributeTrainer(self.config)
        
        try:
            # Train and return validation loss
            val_loss = trainer.train(trial)
            return val_loss
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"❌ Trial failed: {e}")
            return float('inf')
    
    def optimize(self) -> Dict[str, Any]:
        """Run optimization"""
        print(f"🔍 Starting Optuna optimization with {self.n_trials} trials...")
        
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        print("✅ Optimization complete!")
        print(f"🎯 Best trial: {self.study.best_trial.number}")
        print(f"🏆 Best value: {self.study.best_value:.4f}")
        print("📊 Best parameters:")
        for key, value in self.study.best_params.items():
            print(f"   {key}: {value}")
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'study': self.study
        }


# Dataset creation helper
class DatasetCreator:
    """Helper class to create sample datasets for testing"""
    
    @staticmethod
    def create_sample_dataset(output_dir: str = "data/attribute_dataset", num_samples: int = 1000):
        """Create a sample dataset for testing"""
        output_path = Path(output_dir)
        images_path = output_path / "images"
        images_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🎨 Creating sample dataset with {num_samples} images...")
        
        annotations = []
        
        for i in range(num_samples):
            # Create synthetic person image
            img = np.random.randint(50, 255, (224, 224, 3), dtype=np.uint8)
            
            # Add person-like features
            # Head area (hair color region)
            hair_color = random.choice([color.value for color in HairColor if color != HairColor.UNKNOWN])
            hair_rgb = DatasetCreator._get_color_rgb(hair_color)
            cv2.rectangle(img, (80, 20), (144, 80), hair_rgb, -1)
            
            # Body area (shirt color region)
            shirt_color = random.choice([color.value for color in ShirtColor if color != ShirtColor.UNKNOWN])
            shirt_rgb = DatasetCreator._get_color_rgb(shirt_color)
            cv2.rectangle(img, (70, 80), (154, 180), shirt_rgb, -1)
            
            # Random gender
            gender = random.choice([g.value for g in Gender if g != Gender.UNKNOWN])
            
            # Determine split
            if i < num_samples * 0.7:
                split = "train"
            elif i < num_samples * 0.9:
                split = "val"
            else:
                split = "test"
            
            # Save image
            image_name = f"person_{i:04d}.jpg"
            cv2.imwrite(str(images_path / image_name), img)
            
            # Add annotation
            annotations.append({
                'image_name': image_name,
                'gender': gender,
                'shirt_color': shirt_color,
                'hair_color': hair_color,
                'split': split
            })
            
            if (i + 1) % 100 == 0:
                print(f"   Created {i + 1}/{num_samples} images...")
        
        # Save annotations
        annotations_path = output_path / "annotations.json"
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"✅ Sample dataset created at {output_path}")
        print(f"   Images: {len(annotations)}")
        print(f"   Annotations: {annotations_path}")
        
        # Print distribution
        splits = Counter(ann['split'] for ann in annotations)
        print(f"   Split distribution: {dict(splits)}")
    
    @staticmethod
    def _get_color_rgb(color_name: str) -> Tuple[int, int, int]:
        """Get RGB values for color name"""
        color_map = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'yellow': (0, 255, 255),
            'pink': (147, 20, 255),
            'purple': (128, 0, 128),
            'orange': (0, 165, 255),
            'brown': (19, 69, 139),
            'gray': (128, 128, 128),
            'blonde': (0, 255, 255),
        }
        return color_map.get(color_name, (128, 128, 128))


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Attribute Training System")
    print("=" * 50)
    
    # Test 1: Create sample dataset
    print("\n1️⃣ Creating sample dataset...")
    try:
        DatasetCreator.create_sample_dataset(
            output_dir="data/test_attribute_dataset",
            num_samples=200  # Small dataset for testing
        )
        print("✅ Sample dataset created successfully")
    except Exception as e:
        print(f"❌ Sample dataset creation failed: {e}")
        exit(1)
    
    # Test 2: Test dataset loading
    print("\n2️⃣ Testing dataset loading...")
    try:
        config = TrainingConfig(
            dataset_path="data/test_attribute_dataset",
            batch_size=16,
            epochs=2,  # Short test
            use_wandb=False
        )
        trainer = AttributeTrainer(config)
        trainer.prepare_datasets()
        trainer.create_data_loaders()
        print("✅ Dataset loading successful")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        exit(1)
    
    # Test 3: Quick training test
    print("\n3️⃣ Testing quick training...")
    try:
        trainer.create_model()
        trainer.create_optimizer()
        
        # Run one epoch
        train_metrics = trainer.train_epoch(0)
        val_metrics = trainer.validate_epoch()
        
        print("✅ Quick training test successful")
        print(f"   Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"   Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"   Val Accuracy: {val_metrics['val_overall_acc']:.4f}")
        
    except Exception as e:
        print(f"❌ Quick training test failed: {e}")
    
    print("\n🎉 All training system tests completed!")
    print("\nNext steps:")
    print("1. Create your real dataset with proper annotations")
    print("2. Run full training: python training_system.py")
    print("3. Use trained model in AttributeClassifier")
    print("4. Integrate with complete pipeline")
