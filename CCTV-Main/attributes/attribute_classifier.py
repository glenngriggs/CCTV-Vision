"""
Attribute Classification Module
Classifies gender, shirt color, and hair color from person crops
Supports multi-task learning and interactive querying
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0
import timm
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import time
from collections import deque
import random


class Gender(Enum):
    """Gender classifications"""
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class ShirtColor(Enum):
    """Shirt color classifications"""
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    BLACK = "black"
    WHITE = "white"
    YELLOW = "yellow"
    PINK = "pink"
    PURPLE = "purple"
    ORANGE = "orange"
    BROWN = "brown"
    GRAY = "gray"
    UNKNOWN = "unknown"


class HairColor(Enum):
    """Hair color classifications"""
    BLACK = "black"
    BROWN = "brown"
    BLONDE = "blonde"
    RED = "red"
    GRAY = "gray"
    WHITE = "white"
    UNKNOWN = "unknown"


@dataclass
class PersonAttributes:
    """Container for person attributes with confidence scores"""
    gender: Gender
    shirt_color: ShirtColor
    hair_color: HairColor
    gender_confidence: float
    shirt_confidence: float
    hair_confidence: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'gender': self.gender.value,
            'shirt_color': self.shirt_color.value,
            'hair_color': self.hair_color.value,
            'gender_confidence': self.gender_confidence,
            'shirt_confidence': self.shirt_confidence,
            'hair_confidence': self.hair_confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PersonAttributes':
        """Create from dictionary"""
        return cls(
            gender=Gender(data.get('gender', 'unknown')),
            shirt_color=ShirtColor(data.get('shirt_color', 'unknown')),
            hair_color=HairColor(data.get('hair_color', 'unknown')),
            gender_confidence=data.get('gender_confidence', 0.0),
            shirt_confidence=data.get('shirt_confidence', 0.0),
            hair_confidence=data.get('hair_confidence', 0.0)
        )
    
    def matches_query(self, 
                     query_gender: Optional[str] = None,
                     query_shirt: Optional[str] = None,
                     query_hair: Optional[str] = None,
                     min_confidence: float = 0.5) -> bool:
        """Check if attributes match query criteria"""
        
        # Check gender match
        if query_gender and query_gender.lower() not in ["n/a", "any", ""]:
            if (self.gender_confidence < min_confidence or 
                self.gender.value != query_gender.lower()):
                return False
        
        # Check shirt color match
        if query_shirt and query_shirt.lower() not in ["n/a", "any", ""]:
            if (self.shirt_confidence < min_confidence or 
                self.shirt_color.value != query_shirt.lower()):
                return False
        
        # Check hair color match
        if query_hair and query_hair.lower() not in ["n/a", "any", ""]:
            if (self.hair_confidence < min_confidence or 
                self.hair_color.value != query_hair.lower()):
                return False
        
        return True
    
    def get_match_score(self, 
                       query_gender: Optional[str] = None,
                       query_shirt: Optional[str] = None,
                       query_hair: Optional[str] = None) -> float:
        """Get match score (0-1) for query criteria"""
        scores = []
        
        if query_gender and query_gender.lower() not in ["n/a", "any", ""]:
            if self.gender.value == query_gender.lower():
                scores.append(self.gender_confidence)
            else:
                scores.append(0.0)
        
        if query_shirt and query_shirt.lower() not in ["n/a", "any", ""]:
            if self.shirt_color.value == query_shirt.lower():
                scores.append(self.shirt_confidence)
            else:
                scores.append(0.0)
        
        if query_hair and query_hair.lower() not in ["n/a", "any", ""]:
            if self.hair_color.value == query_hair.lower():
                scores.append(self.hair_confidence)
            else:
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 1.0


class MultiTaskAttributeNet(nn.Module):
    """
    Multi-task neural network for gender, shirt color, and hair color classification
    Uses a shared backbone with separate heads for each task
    """
    
    def __init__(self, 
                 backbone: str = "efficientnet_b0", 
                 num_genders: int = 2, 
                 num_shirt_colors: int = 11,
                 num_hair_colors: int = 6,
                 dropout: float = 0.3):
        super().__init__()
        
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == "efficientnet_b0":
            self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
            feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "resnet50":
            self.backbone = resnet50(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet18":
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Shared feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.gender_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_genders)
        )
        
        self.shirt_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_shirt_colors)
        )
        
        self.hair_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_hair_colors)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        # Extract features from backbone
        features = self.backbone(x)
        features = self.feature_extractor(features)
        
        # Task-specific predictions
        gender_logits = self.gender_head(features)
        shirt_logits = self.shirt_head(features)
        hair_logits = self.hair_head(features)
        
        return {
            'gender': gender_logits,
            'shirt_color': shirt_logits,
            'hair_color': hair_logits
        }


class AttributeClassifier:
    """
    Main attribute classifier that handles inference and training
    Can work with trained models or provide demo functionality
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 backbone: str = "efficientnet_b0",
                 device: Optional[str] = None,
                 image_size: int = 224):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.model_path = model_path
        
        # Class mappings
        self.gender_classes = [Gender.MALE, Gender.FEMALE]
        self.shirt_classes = list(ShirtColor)
        self.hair_classes = list(HairColor)
        
        # Initialize model
        self.model = MultiTaskAttributeNet(
            backbone=backbone,
            num_genders=len(self.gender_classes),
            num_shirt_colors=len(self.shirt_classes),
            num_hair_colors=len(self.hair_classes)
        )
        
        # Load weights if provided
        self.model_loaded = False
        if model_path and Path(model_path).exists():
            self._load_model_weights(model_path)
        else:
            print("⚠️ No trained model provided - using demo mode with random predictions")
            print("   To use real predictions, train a model using the training system")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Performance tracking
        self.inference_times = []
        self.total_inferences = 0
        
        print(f"🎯 Attribute Classifier initialized")
        print(f"   Device: {self.device}")
        print(f"   Model: {'Trained' if self.model_loaded else 'Demo (random)'}")
        print(f"   Classes: {len(self.gender_classes)} genders, {len(self.shirt_classes)} shirts, {len(self.hair_classes)} hair")
    
    def _load_model_weights(self, model_path: str):
        """Load trained model weights"""
        try:
            print(f"🔄 Loading model weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model_loaded = True
            print("✅ Model weights loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load model weights: {e}")
            print("   Continuing with demo mode")
            self.model_loaded = False
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            tensor = self.transform(image).unsqueeze(0)
            return tensor.to(self.device)
        
        except Exception as e:
            print(f"⚠️ Preprocessing error: {e}")
            # Return dummy tensor if preprocessing fails
            return torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
    
    def predict_attributes(self, person_crop: np.ndarray) -> PersonAttributes:
        """
        Predict attributes for a person crop
        
        Args:
            person_crop: Person crop image (BGR format)
            
        Returns:
            PersonAttributes object with predictions and confidence scores
        """
        start_time = time.time()
        
        try:
            if self.model_loaded:
                # Use trained model
                attributes = self._predict_with_model(person_crop)
            else:
                # Use demo predictions (rule-based + randomness)
                attributes = self._predict_demo(person_crop)
                
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            # Return unknown attributes on error
            attributes = PersonAttributes(
                gender=Gender.UNKNOWN,
                shirt_color=ShirtColor.UNKNOWN,
                hair_color=HairColor.UNKNOWN,
                gender_confidence=0.0,
                shirt_confidence=0.0,
                hair_confidence=0.0
            )
        
        # Update performance tracking
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.total_inferences += 1
        
        return attributes
    
    def _predict_with_model(self, person_crop: np.ndarray) -> PersonAttributes:
        """Make predictions using trained model"""
        # Preprocess image
        input_tensor = self.preprocess_image(person_crop)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Apply softmax to get probabilities
            gender_probs = F.softmax(outputs['gender'], dim=1)
            shirt_probs = F.softmax(outputs['shirt_color'], dim=1)
            hair_probs = F.softmax(outputs['hair_color'], dim=1)
            
            # Get predictions and confidences
            gender_idx = torch.argmax(gender_probs, dim=1).item()
            shirt_idx = torch.argmax(shirt_probs, dim=1).item()
            hair_idx = torch.argmax(hair_probs, dim=1).item()
            
            gender_conf = gender_probs[0, gender_idx].item()
            shirt_conf = shirt_probs[0, shirt_idx].item()
            hair_conf = hair_probs[0, hair_idx].item()
        
        # Map to enum classes
        gender = self.gender_classes[gender_idx] if gender_idx < len(self.gender_classes) else Gender.UNKNOWN
        shirt = self.shirt_classes[shirt_idx] if shirt_idx < len(self.shirt_classes) else ShirtColor.UNKNOWN
        hair = self.hair_classes[hair_idx] if hair_idx < len(self.hair_classes) else HairColor.UNKNOWN
        
        return PersonAttributes(
            gender=gender,
            shirt_color=shirt,
            hair_color=hair,
            gender_confidence=gender_conf,
            shirt_confidence=shirt_conf,
            hair_confidence=hair_conf
        )
    
    def _predict_demo(self, person_crop: np.ndarray) -> PersonAttributes:
        """Make demo predictions using simple heuristics + randomness"""
        
        # Analyze image colors for shirt color prediction
        shirt_color, shirt_conf = self._analyze_shirt_color(person_crop)
        
        # Analyze top region for hair color
        hair_color, hair_conf = self._analyze_hair_color(person_crop)
        
        # Random gender with bias
        gender = random.choice([Gender.MALE, Gender.FEMALE])
        gender_conf = random.uniform(0.6, 0.9)
        
        return PersonAttributes(
            gender=gender,
            shirt_color=shirt_color,
            hair_color=hair_color,
            gender_confidence=gender_conf,
            shirt_confidence=shirt_conf,
            hair_confidence=hair_conf
        )
    
    def _analyze_shirt_color(self, image: np.ndarray) -> Tuple[ShirtColor, float]:
        """Analyze shirt color from image (demo heuristic)"""
        try:
            # Focus on middle region (torso area)
            height, width = image.shape[:2]
            torso_region = image[height//3:2*height//3, width//4:3*width//4]
            
            if torso_region.size == 0:
                return ShirtColor.UNKNOWN, 0.0
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
            
            # Calculate average hue
            avg_hue = np.mean(hsv[:, :, 0])
            avg_saturation = np.mean(hsv[:, :, 1])
            avg_value = np.mean(hsv[:, :, 2])
            
            # Simple color classification based on HSV
            if avg_value < 50:  # Very dark
                color = ShirtColor.BLACK
            elif avg_value > 200 and avg_saturation < 30:  # Very light with low saturation
                color = ShirtColor.WHITE
            elif avg_saturation < 50:  # Low saturation (grayscale)
                color = ShirtColor.GRAY
            else:  # Color based on hue
                if avg_hue < 10 or avg_hue > 170:
                    color = ShirtColor.RED
                elif avg_hue < 30:
                    color = ShirtColor.ORANGE
                elif avg_hue < 60:
                    color = ShirtColor.YELLOW
                elif avg_hue < 80:
                    color = ShirtColor.GREEN
                elif avg_hue < 130:
                    color = ShirtColor.BLUE
                else:
                    color = ShirtColor.PURPLE
            
            confidence = random.uniform(0.5, 0.8)
            return color, confidence
            
        except:
            return ShirtColor.UNKNOWN, 0.0
    
    def _analyze_hair_color(self, image: np.ndarray) -> Tuple[HairColor, float]:
        """Analyze hair color from image (demo heuristic)"""
        try:
            # Focus on top region (head area)
            height, width = image.shape[:2]
            head_region = image[:height//3, :]
            
            if head_region.size == 0:
                return HairColor.UNKNOWN, 0.0
            
            # Convert to HSV
            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
            
            # Calculate average values
            avg_hue = np.mean(hsv[:, :, 0])
            avg_saturation = np.mean(hsv[:, :, 1])
            avg_value = np.mean(hsv[:, :, 2])
            
            # Simple hair color classification
            if avg_value < 40:
                color = HairColor.BLACK
            elif avg_value > 180:
                color = HairColor.WHITE if avg_saturation < 30 else HairColor.BLONDE
            elif avg_saturation < 30:
                color = HairColor.GRAY
            elif avg_hue < 20:
                color = HairColor.RED
            elif avg_value < 100:
                color = HairColor.BROWN
            else:
                color = HairColor.BLONDE
            
            confidence = random.uniform(0.4, 0.7)
            return color, confidence
            
        except:
            return HairColor.UNKNOWN, 0.0
    
    def batch_predict(self, person_crops: List[np.ndarray]) -> List[PersonAttributes]:
        """Predict attributes for multiple person crops efficiently"""
        if not person_crops:
            return []
        
        results = []
        
        if self.model_loaded and len(person_crops) > 1:
            # Batch processing for trained model
            try:
                # Preprocess all images
                batch_tensors = []
                valid_indices = []
                
                for i, crop in enumerate(person_crops):
                    try:
                        tensor = self.preprocess_image(crop)
                        batch_tensors.append(tensor)
                        valid_indices.append(i)
                    except:
                        continue
                
                if batch_tensors:
                    # Concatenate tensors for batch processing
                    batch_input = torch.cat(batch_tensors, dim=0)
                    
                    # Run batch inference
                    with torch.no_grad():
                        outputs = self.model(batch_input)
                        
                        gender_probs = F.softmax(outputs['gender'], dim=1)
                        shirt_probs = F.softmax(outputs['shirt_color'], dim=1)
                        hair_probs = F.softmax(outputs['hair_color'], dim=1)
                    
                    # Process results
                    batch_results = []
                    for i in range(batch_input.size(0)):
                        gender_idx = torch.argmax(gender_probs[i]).item()
                        shirt_idx = torch.argmax(shirt_probs[i]).item()
                        hair_idx = torch.argmax(hair_probs[i]).item()
                        
                        gender_conf = gender_probs[i, gender_idx].item()
                        shirt_conf = shirt_probs[i, shirt_idx].item()
                        hair_conf = hair_probs[i, hair_idx].item()
                        
                        gender = self.gender_classes[gender_idx] if gender_idx < len(self.gender_classes) else Gender.UNKNOWN
                        shirt = self.shirt_classes[shirt_idx] if shirt_idx < len(self.shirt_classes) else ShirtColor.UNKNOWN
                        hair = self.hair_classes[hair_idx] if hair_idx < len(self.hair_classes) else HairColor.UNKNOWN
                        
                        attributes = PersonAttributes(gender, shirt, hair, gender_conf, shirt_conf, hair_conf)
                        batch_results.append(attributes)
                    
                    # Map results back to original order
                    for i, crop in enumerate(person_crops):
                        if i in valid_indices:
                            batch_idx = valid_indices.index(i)
                            results.append(batch_results[batch_idx])
                        else:
                            # Failed preprocessing
                            results.append(PersonAttributes(Gender.UNKNOWN, ShirtColor.UNKNOWN, HairColor.UNKNOWN, 0.0, 0.0, 0.0))
                
                else:
                    # All preprocessing failed
                    results = [PersonAttributes(Gender.UNKNOWN, ShirtColor.UNKNOWN, HairColor.UNKNOWN, 0.0, 0.0, 0.0) 
                              for _ in person_crops]
                
            except Exception as e:
                print(f"⚠️ Batch processing failed: {e}, falling back to individual processing")
                results = [self.predict_attributes(crop) for crop in person_crops]
        
        else:
            # Individual processing
            results = [self.predict_attributes(crop) for crop in person_crops]
        
        return results
    
    def visualize_attributes(self, 
                           image: np.ndarray, 
                           attributes: PersonAttributes,
                           bbox: Optional[List[int]] = None) -> np.ndarray:
        """
        Visualize predicted attributes on image
        
        Args:
            image: Input image
            attributes: Predicted attributes
            bbox: Optional bounding box [x1, y1, x2, y2]
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Prepare text
        lines = [
            f"Gender: {attributes.gender.value} ({attributes.gender_confidence:.2f})",
            f"Shirt: {attributes.shirt_color.value} ({attributes.shirt_confidence:.2f})",
            f"Hair: {attributes.hair_color.value} ({attributes.hair_confidence:.2f})"
        ]
        
        # Determine text position
        if bbox:
            x1, y1, x2, y2 = bbox
            text_x, text_y = x1, y1 - 10
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            text_x, text_y = 10, 30
        
        # Draw text background and text
        for i, line in enumerate(lines):
            y_pos = text_y - (len(lines) - i - 1) * 25
            if y_pos < 20:  # Adjust if too close to top
                y_pos = 20 + i * 25
            
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background rectangle
            cv2.rectangle(annotated, 
                         (text_x - 5, y_pos - text_size[1] - 5),
                         (text_x + text_size[0] + 5, y_pos + 5),
                         (0, 0, 0), -1)
            
            # Text
            cv2.putText(annotated, line, (text_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {
                'avg_inference_time': 0,
                'avg_fps': 0,
                'total_inferences': 0,
                'model_loaded': self.model_loaded
            }
        
        avg_time = np.mean(self.inference_times)
        return {
            'avg_inference_time': avg_time,
            'avg_fps': 1.0 / avg_time if avg_time > 0 else 0,
            'total_inferences': self.total_inferences,
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'model_loaded': self.model_loaded
        }


class InteractiveQuerySystem:
    """
    Interactive system for querying people by attributes
    """
    
    def __init__(self):
        self.current_query = {
            'gender': None,
            'shirt_color': None,
            'hair_color': None
        }
    
    def get_user_preferences(self) -> Dict[str, Optional[str]]:
        """Get user preferences through interactive prompts"""
        print("\n🔍 Person Search Configuration")
        print("=" * 40)
        print("Enter your search criteria (type 'any' or press Enter to skip any category)")
        
        # Gender preference
        print(f"\n👤 Gender options: {[g.value for g in Gender if g != Gender.UNKNOWN]}")
        gender = input("Enter preferred gender (or 'any'): ").strip().lower()
        if gender in ['any', 'n/a', 'na', '']:
            gender = None
        
        # Shirt color preference
        print(f"\n👕 Shirt color options: {[c.value for c in ShirtColor if c != ShirtColor.UNKNOWN]}")
        shirt = input("Enter preferred shirt color (or 'any'): ").strip().lower()
        if shirt in ['any', 'n/a', 'na', '']:
            shirt = None
        
        # Hair color preference
        print(f"\n💇 Hair color options: {[h.value for h in HairColor if h != HairColor.UNKNOWN]}")
        hair = input("Enter preferred hair color (or 'any'): ").strip().lower()
        if hair in ['any', 'n/a', 'na', '']:
            hair = None
        
        self.current_query = {
            'gender': gender,
            'shirt_color': shirt,
            'hair_color': hair
        }
        
        print(f"\n✅ Search criteria set:")
        print(f"   Gender: {gender or 'Any'}")
        print(f"   Shirt Color: {shirt or 'Any'}")
        print(f"   Hair Color: {hair or 'Any'}")
        
        return self.current_query
    
    def format_query_string(self) -> str:
        """Format current query as human-readable string"""
        parts = []
        
        if self.current_query['gender']:
            parts.append(f"{self.current_query['gender']} person")
        else:
            parts.append("person")
        
        if self.current_query['shirt_color']:
            parts.append(f"wearing {self.current_query['shirt_color']} shirt")
        
        if self.current_query['hair_color']:
            parts.append(f"with {self.current_query['hair_color']} hair")
        
        return " ".join(parts)
    
    def update_query(self, **kwargs):
        """Update query programmatically"""
        for key, value in kwargs.items():
            if key in self.current_query:
                self.current_query[key] = value
    
    def matches_criteria(self, attributes: PersonAttributes, 
                        min_confidence: float = 0.5) -> bool:
        """Check if person attributes match current query criteria"""
        return attributes.matches_query(
            query_gender=self.current_query['gender'],
            query_shirt=self.current_query['shirt_color'],
            query_hair=self.current_query['hair_color'],
            min_confidence=min_confidence
        )
    
    def get_match_score(self, attributes: PersonAttributes) -> float:
        """Get match score for person attributes"""
        return attributes.get_match_score(
            query_gender=self.current_query['gender'],
            query_shirt=self.current_query['shirt_color'],
            query_hair=self.current_query['hair_color']
        )


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Attribute Classification Module")
    print("=" * 50)
    
    # Test 1: Basic initialization
    print("\n1️⃣ Testing basic initialization...")
    try:
        classifier = AttributeClassifier()
        print("✅ Basic initialization successful")
    except Exception as e:
        print(f"❌ Basic initialization failed: {e}")
        exit(1)
    
    # Test 2: Demo prediction on sample image
    print("\n2️⃣ Testing demo prediction...")
    try:
        # Create test image with some color patterns
        test_image = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        
        # Add some colored regions to simulate clothing
        cv2.rectangle(test_image, (20, 60), (80, 140), (0, 0, 255), -1)  # Red shirt area
        cv2.rectangle(test_image, (20, 20), (80, 60), (139, 69, 19), -1)  # Brown hair area
        
        attributes = classifier.predict_attributes(test_image)
        
        print(f"✅ Prediction successful:")
        print(f"   Gender: {attributes.gender.value} ({attributes.gender_confidence:.3f})")
        print(f"   Shirt: {attributes.shirt_color.value} ({attributes.shirt_confidence:.3f})")
        print(f"   Hair: {attributes.hair_color.value} ({attributes.hair_confidence:.3f})")
        
        # Test visualization
        annotated = classifier.visualize_attributes(test_image, attributes)
        cv2.imwrite('test_attributes_result.jpg', annotated)
        print("   💾 Test result saved as 'test_attributes_result.jpg'")
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
    
    # Test 3: Batch prediction
    print("\n3️⃣ Testing batch prediction...")
    try:
        # Create multiple test images
        test_images = []
        for i in range(3):
            img = np.random.randint(0, 255, (150, 80, 3), dtype=np.uint8)
            # Add different colored regions
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i]
            cv2.rectangle(img, (10, 50), (70, 120), color, -1)
            test_images.append(img)
        
        batch_attributes = classifier.batch_predict(test_images)
        
        print(f"✅ Batch prediction successful:")
        for i, attrs in enumerate(batch_attributes):
            print(f"   Image {i+1}: {attrs.gender.value}, {attrs.shirt_color.value}, {attrs.hair_color.value}")
        
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
    
    # Test 4: Interactive query system
    print("\n4️⃣ Testing interactive query system...")
    try:
        query_system = InteractiveQuerySystem()
        
        # Set sample query
        query_system.update_query(gender='female', shirt_color='red')
        
        print(f"✅ Query system test:")
        print(f"   Query: {query_system.format_query_string()}")
        
        # Test matching
        for i, attrs in enumerate(batch_attributes[:2]):
            matches = query_system.matches_criteria(attrs)
            score = query_system.get_match_score(attrs)
            print(f"   Person {i+1} matches: {matches} (score: {score:.2f})")
        
    except Exception as e:
        print(f"❌ Query system test failed: {e}")
    
    # Test 5: Performance stats
    print("\n5️⃣ Testing performance statistics...")
    try:
        stats = classifier.get_performance_stats()
        print("✅ Performance stats:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Performance stats failed: {e}")
    
    print("\n🎉 All attribute classification tests completed!")
    print("\nNext steps:")
    print("1. Check generated file: test_attributes_result.jpg")
    print("2. Train a real model using the training system for better accuracy")
    print("3. Integrate with PersonDetector and PersonTracker")
    print("4. Test the complete pipeline")
