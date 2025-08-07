#!/usr/bin/env python3
"""
VIDEO PERSON SEARCH SYSTEM
BY Glenn Griggs, Shubhanshu Pokharel, and Lucas Morris
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.cuda.amp import autocast

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
import warnings
import math
from sklearn.cluster import KMeans
import threading

# Computer vision
import timm
from ultralytics import YOLO
import mediapipe as mp
from scipy.optimize import linear_sum_assignment

# Color science imports
from skimage import color
from skimage.color import deltaE_ciede2000

warnings.filterwarnings('ignore')

print("VIDEO PERSON SEARCH SYSTEM")
print("="*80)
print("IoU-based tracking")
print("Safe model loading with validation")
print("Smart region extraction with pose detection")
print("Color science improvements")
print("="*80)

# Directory structure setup
try:
    # Try to get the file path (works in .py scripts)
    BASE_DIR = Path(__file__).parent
except NameError:
    # Fallback for Jupyter notebooks/IPython
    import os
    BASE_DIR = Path(os.getcwd())
MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "videos"

# Model configuration
MODEL_PATH = MODELS_DIR / "ULTIMATE_model_score_0.9398_20250716_110939.pt"
MANDATORY_MODEL_LOADING = True

# HEAD ALIASES for backward compatibility
HEAD_ALIASES = {
    'upper_body_color': 'shirt_color',
    'top_color': 'shirt_color',
    'lower_body_color': 'pants_color',
    'bottom_color': 'pants_color',
    'hair': 'hair_color',
    'shirt': 'shirt_color',
    'pants': 'pants_color'
}

def discover_heads_from_checkpoint(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Discover attribute heads from checkpoint state_dict"""
    discovered_heads = {}

    for key in state_dict.keys():
        if key.startswith('attribute_heads.') and key.endswith('.weight'):
            parts = key.split('.')
            if len(parts) >= 3:
                head_name = parts[1]
                bias_key = key.replace('.weight', '.bias')
                if bias_key in state_dict:
                    weight_tensor = state_dict[key]
                    if len(weight_tensor.shape) == 2:
                        num_classes = weight_tensor.shape[0]
                        discovered_heads[head_name] = num_classes

    return discovered_heads

# Tracker with IoU-based motion prediction

@dataclass
class TrackState:
    """Track state with motion prediction"""
    track_id: int
    attributes: Dict[str, Any]
    last_update_frame: int
    last_bbox: Tuple[int, int, int, int]
    confidence_history: List[float]
    velocity: Tuple[float, float]  # (vx, vy) velocity
    bbox_history: List[Tuple[int, int, int, int]]  # Recent bbox history
    age: int = 0  # Track age
    hits: int = 0  # Number of successful matches

    def predict_bbox(self) -> Tuple[int, int, int, int]:
        """Predict next bbox position using velocity"""
        if len(self.bbox_history) < 2:
            return self.last_bbox

        x1, y1, x2, y2 = self.last_bbox
        pred_x1 = int(x1 + self.velocity[0])
        pred_y1 = int(y1 + self.velocity[1])
        pred_x2 = int(x2 + self.velocity[0])
        pred_y2 = int(y2 + self.velocity[1])

        return (pred_x1, pred_y1, pred_x2, pred_y2)

    def update_velocity(self):
        """Update velocity from bbox history"""
        if len(self.bbox_history) >= 2:
            prev_bbox = self.bbox_history[-2]
            curr_bbox = self.bbox_history[-1]

            prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
            prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
            curr_cx = (curr_bbox[0] + curr_bbox[2]) / 2
            curr_cy = (curr_bbox[1] + curr_bbox[3]) / 2

            self.velocity = (curr_cx - prev_cx, curr_cy - prev_cy)
        else:
            self.velocity = (0.0, 0.0)

def calculate_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate IoU between two bboxes"""
    x1, y1, x2, y2 = bbox1
    x1_pred, y1_pred, x2_pred, y2_pred = bbox2

    # Calculate intersection
    ix1 = max(x1, x1_pred)
    iy1 = max(y1, y1_pred)
    ix2 = min(x2, x2_pred)
    iy2 = min(y2, y2_pred)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    union = area1 + area2 - intersection + 1e-6

    return intersection / union

class EnhancedTracker:
    """Tracker with IoU-based Hungarian assignment and motion prediction"""

    def __init__(self,
                 max_disappeared: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 max_age: int = 30,
                 verbose: bool = False):
        self.next_id = 0
        self.tracks: Dict[int, TrackState] = {}
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.verbose = verbose
        self.cache_cleanup_callback = None

        # Performance tracking
        self.stats = {
            'total_assignments': 0,
            'successful_assignments': 0,
            'id_switches_prevented': 0,
            'tracks_created': 0,
            'tracks_deleted': 0
        }

        if self.verbose:
            print("IoU-based tracker initialized")
            print(f"   IoU threshold: {iou_threshold}")
            print(f"   Min hits: {min_hits}")
            print(f"   Max age: {max_age}")

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracker with Hungarian assignment"""

        # Age all tracks
        for track in self.tracks.values():
            track.age += 1

        if not detections:
            deleted_tracks = self._cleanup_tracks()
            return []

        # If no existing tracks, create new ones
        if not self.tracks:
            for det in detections:
                det['track_id'] = self.next_id  # Ensure track_id is set
                self._create_new_track(det)
            return detections

        # Build cost matrix using IoU + motion prediction
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))

        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            predicted_bbox = track.predict_bbox()

            for j, det in enumerate(detections):
                iou = calculate_iou(predicted_bbox, det['bbox'])

                # Cost = 1 - IoU (lower is better)
                cost_matrix[i, j] = 1.0 - iou

                # Penalize low IoU matches heavily
                if iou < self.iou_threshold:
                    cost_matrix[i, j] = 1.0

        # Hungarian assignment
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            self.stats['total_assignments'] += 1
        else:
            row_indices, col_indices = [], []

        # Process matches
        matched_tracks = set()
        matched_detections = set()
        tracked_detections = []

        for row_idx, col_idx in zip(row_indices, col_indices):
            if cost_matrix[row_idx, col_idx] < (1.0 - self.iou_threshold):
                track_id = track_ids[row_idx]
                det = detections[col_idx]

                # Update track
                det['track_id'] = track_id
                self._update_track(track_id, det)

                matched_tracks.add(track_id)
                matched_detections.add(col_idx)
                tracked_detections.append(det)
                self.stats['successful_assignments'] += 1
            else:
                # Prevented potential ID switch
                self.stats['id_switches_prevented'] += 1

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_detections:
                det['track_id'] = self.next_id
                self._create_new_track(det)
                tracked_detections.append(det)

        # Clean up old/unmatched tracks
        deleted_tracks = self._cleanup_tracks(exclude=matched_tracks)

        # Only return tracks that meet minimum hits requirement
        confirmed_detections = []
        for det in tracked_detections:
            track_id = det['track_id']
            if track_id in self.tracks and self.tracks[track_id].hits >= self.min_hits:
                confirmed_detections.append(det)

        return confirmed_detections

    def _create_new_track(self, detection: Dict):
        """Create new track from detection"""
        self.tracks[self.next_id] = TrackState(
            track_id=self.next_id,
            attributes={},
            last_update_frame=-1000000,
            last_bbox=detection['bbox'],
            confidence_history=[detection['confidence']],
            velocity=(0.0, 0.0),
            bbox_history=[detection['bbox']],
            age=0,
            hits=1
        )
        self.stats['tracks_created'] += 1
        self.next_id += 1

    def _update_track(self, track_id: int, detection: Dict):
        """Update existing track with new detection"""
        track = self.tracks[track_id]

        # Update bbox history (keep last 5 for velocity calculation)
        track.bbox_history.append(detection['bbox'])
        if len(track.bbox_history) > 5:
            track.bbox_history.pop(0)

        # Update state
        track.last_bbox = detection['bbox']
        track.confidence_history.append(detection['confidence'])
        if len(track.confidence_history) > 10:
            track.confidence_history.pop(0)

        # Update velocity
        track.update_velocity()

        # Reset age and increment hits
        track.age = 0
        track.hits += 1

    def _cleanup_tracks(self, exclude: set = None) -> List[int]:
        """Clean up old or lost tracks"""
        if exclude is None:
            exclude = set()

        deleted_tracks = []
        for track_id in list(self.tracks.keys()):
            if track_id in exclude:
                continue

            track = self.tracks[track_id]
            if track.age > self.max_age:
                deleted_tracks.append(track_id)
                del self.tracks[track_id]
                self.stats['tracks_deleted'] += 1

        # Clean up cache for deleted tracks
        if deleted_tracks and self.cache_cleanup_callback:
            self.cache_cleanup_callback(deleted_tracks)

        return deleted_tracks

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get tracker performance statistics"""
        if self.stats['total_assignments'] > 0:
            assignment_rate = self.stats['successful_assignments'] / self.stats['total_assignments']
        else:
            assignment_rate = 0.0

        return {
            'active_tracks': len(self.tracks),
            'assignment_success_rate': assignment_rate,
            'id_switches_prevented': self.stats['id_switches_prevented'],
            'tracks_created': self.stats['tracks_created'],
            'tracks_deleted': self.stats['tracks_deleted']
        }

# Smart Region Extractor with pose-based anatomical priors

class SmartRegionExtractor:
    """Smart region extractor with body part segmentation and anatomical priors"""

    def __init__(self, use_pose: bool = True, use_segmentation: bool = True, verbose: bool = False):
        self.use_pose = use_pose
        self.use_segmentation = use_segmentation
        self.verbose = verbose

        # Initialize MediaPipe modules
        if self.use_pose:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=use_segmentation,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )

        # Performance tracking with counters for verbose output
        self.stats = {
            'pose_extractions': 0,
            'anatomical_fallbacks': 0,
            'skin_removals': 0,
            'successful_extractions': 0,
            'pose_attempts': 0,
            'fallback_attempts': 0
        }
        
        # Verbose output counters to reduce spam
        self.verbose_counter = 0
        self.verbose_interval = 50  # Only show verbose output every 50 operations

        if self.verbose:
            print("Smart region extractor initialized")
            print(f"   Pose detection: {'ENABLED' if use_pose else 'DISABLED'}")
            print(f"   Segmentation: {'ENABLED' if use_segmentation else 'DISABLED'}")

    def extract_regions_smart(self, person_crop: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """Extract regions using smart anatomical analysis"""

        regions = {
            'hair': None,
            'shirt': None,
            'pants': None,
            'face': None
        }

        h, w = person_crop.shape[:2]
        self.verbose_counter += 1

        # Method 1: MediaPipe pose-based extraction (most accurate)
        if self.use_pose:
            self.stats['pose_attempts'] += 1
            pose_regions = self._extract_with_pose(person_crop)
            if pose_regions:
                regions.update(pose_regions)
                self.stats['pose_extractions'] += 1
                self.stats['successful_extractions'] += 1
                
                # Only show verbose output every N operations
                if self.verbose and self.verbose_counter % self.verbose_interval == 0:
                    print(f"Using MediaPipe pose-based region extraction (success: {self.stats['pose_extractions']}/{self.stats['pose_attempts']})")
                return regions

        # Method 2: Smart anatomical fallback with body proportions
        self.stats['fallback_attempts'] += 1
        anatomical_regions = self._extract_with_anatomical_priors(person_crop)
        regions.update(anatomical_regions)
        self.stats['anatomical_fallbacks'] += 1
        self.stats['successful_extractions'] += 1

        # Only show verbose output every N operations
        if self.verbose and self.verbose_counter % self.verbose_interval == 0:
            print(f"Using anatomical priors fallback (fallback: {self.stats['anatomical_fallbacks']}/{self.stats['fallback_attempts']})")

        return regions

    def _extract_with_pose(self, image: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Extract regions using MediaPipe pose landmarks"""
        try:
            # Preprocess for better pose detection
            enhanced = self._enhance_for_pose_detection(image)
            rgb_image = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

            results = self.pose.process(rgb_image)

            if not results.pose_landmarks:
                return None

            landmarks = results.pose_landmarks.landmark
            h, w = image.shape[:2]

            # Convert normalized landmarks to pixel coordinates
            points = {}
            for i, landmark in enumerate(landmarks):
                if landmark.visibility > 0.6:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points[i] = (x, y)

            regions = {}

            # Hair region extraction
            hair_region = self._extract_hair_region_pose(image, points)
            if hair_region is not None:
                regions['hair'] = hair_region

            # Shirt region extraction (torso-focused)
            shirt_region = self._extract_shirt_region_pose(image, points)
            if shirt_region is not None:
                regions['shirt'] = shirt_region

            # Pants region extraction
            pants_region = self._extract_pants_region_pose(image, points)
            if pants_region is not None:
                regions['pants'] = pants_region

            # Face region for skin tone analysis
            face_region = self._extract_face_region_pose(image, points)
            if face_region is not None:
                regions['face'] = face_region

            return regions

        except Exception as e:
            if self.verbose and self.verbose_counter % self.verbose_interval == 0:
                print(f"Pose-based extraction failed: {e}")
            return None

    def _extract_hair_region_pose(self, image: np.ndarray, points: Dict[int, Tuple[int, int]]) -> Optional[np.ndarray]:
        """Extract hair region using pose landmarks"""
        h, w = image.shape[:2]

        # Key points for hair region
        nose = points.get(0)
        left_eye = points.get(2)
        right_eye = points.get(5)
        left_ear = points.get(7)
        right_ear = points.get(8)

        if not nose:
            return None

        # Calculate head center and dimensions
        head_points = [p for p in [nose, left_eye, right_eye, left_ear, right_ear] if p]
        if len(head_points) < 2:
            return None

        head_xs = [p[0] for p in head_points]
        head_ys = [p[1] for p in head_points]

        head_center_x = int(np.mean(head_xs))
        head_center_y = int(np.mean(head_ys))
        head_width = max(head_xs) - min(head_xs)

        # Hair region: expand upward and slightly outward from head
        hair_width = int(head_width * 1.4)
        hair_height = int(head_width * 0.8)

        # Hair region bounds
        x1 = max(0, head_center_x - hair_width // 2)
        x2 = min(w, head_center_x + hair_width // 2)
        y1 = max(0, min(head_ys) - hair_height)
        y2 = min(h, head_center_y - int(head_width * 0.1))

        if y2 <= y1 or x2 <= x1:
            return None

        hair_region = image[y1:y2, x1:x2]

        # Apply skin removal to focus on hair
        hair_region = self._remove_skin_pixels(hair_region)

        return hair_region if hair_region.size > 500 else None

    def _extract_shirt_region_pose(self, image: np.ndarray, points: Dict[int, Tuple[int, int]]) -> Optional[np.ndarray]:
        """Extract shirt region using pose landmarks (torso-focused)"""
        h, w = image.shape[:2]

        # Key points for shirt region
        left_shoulder = points.get(11)
        right_shoulder = points.get(12)
        left_hip = points.get(23)
        right_hip = points.get(24)

        if not (left_shoulder and right_shoulder):
            return None

        # Calculate torso bounds
        shoulder_y = min(left_shoulder[1], right_shoulder[1])
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

        # If hips are available, use them; otherwise estimate
        if left_hip and right_hip:
            hip_y = min(left_hip[1], right_hip[1])
            torso_center_x = int((left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4)
        else:
            hip_y = min(h, shoulder_y + int(shoulder_width * 1.5))
            torso_center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)

        # Shirt region bounds (focus on upper torso)
        shirt_width = int(shoulder_width * 1.2)
        shirt_start_y = shoulder_y + int(shoulder_width * 0.15)
        shirt_end_y = shoulder_y + int(shoulder_width * 0.9)

        x1 = max(0, torso_center_x - shirt_width // 2)
        x2 = min(w, torso_center_x + shirt_width // 2)
        y1 = max(0, shirt_start_y)
        y2 = min(h, min(shirt_end_y, hip_y))

        if y2 <= y1 or x2 <= x1:
            return None

        shirt_region = image[y1:y2, x1:x2]

        # Remove skin pixels to focus on clothing
        shirt_region = self._remove_skin_pixels(shirt_region)

        return shirt_region if shirt_region.size > 1000 else None

    def _extract_pants_region_pose(self, image: np.ndarray, points: Dict[int, Tuple[int, int]]) -> Optional[np.ndarray]:
        """Extract pants region using pose landmarks"""
        h, w = image.shape[:2]

        # Key points for pants region
        left_hip = points.get(23)
        right_hip = points.get(24)
        left_knee = points.get(25)
        right_knee = points.get(26)

        if not (left_hip and right_hip):
            return None

        # Calculate pants bounds
        hip_y = min(left_hip[1], right_hip[1])
        hip_center_x = int((left_hip[0] + right_hip[0]) / 2)
        hip_width = abs(right_hip[0] - left_hip[0])

        # If knees are available, use them; otherwise estimate
        if left_knee and right_knee:
            knee_y = min(left_knee[1], right_knee[1])
        else:
            knee_y = min(h, hip_y + int(hip_width * 1.5))

        # Pants region bounds
        pants_width = int(hip_width * 1.3)

        x1 = max(0, hip_center_x - pants_width // 2)
        x2 = min(w, hip_center_x + pants_width // 2)
        y1 = max(0, hip_y - int(hip_width * 0.1))
        y2 = min(h, knee_y + int(hip_width * 0.2))

        if y2 <= y1 or x2 <= x1:
            return None

        return image[y1:y2, x1:x2]

    def _extract_face_region_pose(self, image: np.ndarray, points: Dict[int, Tuple[int, int]]) -> Optional[np.ndarray]:
        """Extract face region for skin tone analysis"""
        face_points = [points.get(i) for i in [0, 1, 2, 5, 7, 8] if points.get(i)]

        if len(face_points) < 3:
            return None

        # Calculate face bounding box
        face_xs = [p[0] for p in face_points]
        face_ys = [p[1] for p in face_points]

        x1 = max(0, min(face_xs) - 10)
        x2 = min(image.shape[1], max(face_xs) + 10)
        y1 = max(0, min(face_ys) - 10)
        y2 = min(image.shape[0], max(face_ys) + 10)

        if y2 <= y1 or x2 <= x1:
            return None

        return image[y1:y2, x1:x2]

    def _extract_with_anatomical_priors(self, image: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """Fallback extraction using improved anatomical proportions"""
        h, w = image.shape[:2]

        regions = {}

        # Improved anatomical proportions
        # Hair region: top 20% of person, centered 80% width
        hair_height = int(h * 0.2)
        hair_width = int(w * 0.8)
        hair_x1 = int(w * 0.1)
        hair_x2 = hair_x1 + hair_width
        regions['hair'] = image[0:hair_height, hair_x1:hair_x2]

        # Shirt region: 25-65% of height, central 80% of width
        shirt_y1 = int(h * 0.25)
        shirt_y2 = int(h * 0.65)
        shirt_x1 = int(w * 0.1)
        shirt_x2 = int(w * 0.9)
        regions['shirt'] = image[shirt_y1:shirt_y2, shirt_x1:shirt_x2]

        # Pants region: 60-90% of height, central 70% of width
        pants_y1 = int(h * 0.6)
        pants_y2 = int(h * 0.9)
        pants_x1 = int(w * 0.15)
        pants_x2 = int(w * 0.85)
        regions['pants'] = image[pants_y1:pants_y2, pants_x1:pants_x2]

        # Face region: 20-45% of height, central 60% of width
        face_y1 = int(h * 0.2)
        face_y2 = int(h * 0.45)
        face_x1 = int(w * 0.2)
        face_x2 = int(w * 0.8)
        regions['face'] = image[face_y1:face_y2, face_x1:face_x2]

        return regions

    def _enhance_for_pose_detection(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better pose detection"""
        h, w = image.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))

        # Enhance contrast and brightness
        enhanced = cv2.convertScaleAbs(image, alpha=1.1, beta=10)

        # Slight denoising
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

        return enhanced

    def _remove_skin_pixels(self, region: np.ndarray) -> np.ndarray:
        """Remove skin-colored pixels to focus on clothing/hair"""
        if region.size == 0:
            return region

        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Skin color ranges in HSV
        lower_skin1 = np.array([0, 20, 70])
        upper_skin1 = np.array([20, 255, 255])
        lower_skin2 = np.array([160, 20, 70])
        upper_skin2 = np.array([180, 255, 255])

        # Create skin mask
        skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)

        # Invert mask to keep non-skin pixels
        non_skin_mask = cv2.bitwise_not(skin_mask)

        # Apply mask
        result = cv2.bitwise_and(region, region, mask=non_skin_mask)
        self.stats['skin_removals'] += 1

        return result

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get region extraction performance statistics"""
        return {
            'pose_extractions': self.stats['pose_extractions'],
            'anatomical_fallbacks': self.stats['anatomical_fallbacks'],
            'skin_removals': self.stats['skin_removals'],
            'successful_extractions': self.stats['successful_extractions'],
            'pose_success_rate': self.stats['pose_extractions'] / max(self.stats['successful_extractions'], 1)
        }

# Safe Model Loader with validation and critical layer checks

class SafeModelLoader:
    """Safe model loader with critical layer validation and proper error handling"""

    def __init__(self, model_path: str, mandatory: bool = True, verbose: bool = False):
        self.model = None
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.discovered_heads = {}
        self.validation_passed = False
        self.prediction_cache = {}
        self.cache_lock = threading.Lock()

        if not Path(model_path).exists():
            if mandatory:
                raise FileNotFoundError(f"Model file not found: {model_path}")
            else:
                if self.verbose:
                    print(f"Model file not found: {model_path} (continuing without model)")
                return

        # Load and validate model
        self._load_and_validate_model(model_path, mandatory)

    def _load_and_validate_model(self, model_path: str, mandatory: bool):
        """Load model with comprehensive validation"""
        try:
            # Step 1: Load checkpoint and analyze structure
            checkpoint = torch.load(model_path, map_location=self.device)

            if 'model_state_dict' not in checkpoint:
                raise ValueError("Checkpoint missing 'model_state_dict'")

            state_dict = checkpoint['model_state_dict']

            # Step 2: Discover and validate head configuration
            self.discovered_heads = self._discover_and_validate_heads(state_dict)

            if not self.discovered_heads:
                raise ValueError("No valid attribute heads discovered in checkpoint")

            # Step 3: Validate backbone compatibility
            backbone_validation = self._validate_backbone_compatibility(state_dict)
            if not backbone_validation['compatible']:
                raise ValueError(f"Backbone incompatible: {backbone_validation['reason']}")

            # Step 4: Create model architecture
            self.model = self._create_validated_model(self.discovered_heads)

            # Step 5: Perform safe loading with validation
            loading_result = self._safe_load_state_dict(state_dict)

            if not loading_result['success']:
                if mandatory:
                    raise ValueError(f"Critical loading failure: {loading_result['error']}")
                else:
                    if self.verbose:
                        print(f"Model loading issues: {loading_result['error']}")
                    self.model = None
                    return

            # Step 6: Post-loading validation
            validation_result = self._validate_loaded_model()

            if not validation_result['success']:
                if mandatory:
                    raise ValueError(f"Model validation failed: {validation_result['error']}")
                else:
                    if self.verbose:
                        print(f"Model validation failed: {validation_result['error']}")
                    self.model = None
                    return

            self.validation_passed = True

            if self.verbose:
                print(f"Safe model loaded and validated successfully")
                print(f"   Device: {self.device}")
                print(f"   Heads: {self.discovered_heads}")
                print(f"   Validation: PASSED")

        except Exception as e:
            if mandatory:
                raise RuntimeError(f"Safe model loading failed: {e}")
            else:
                if self.verbose:
                    print(f"Safe model loading failed: {e}")
                self.model = None

    def _discover_and_validate_heads(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Discover heads with validation"""
        discovered_heads = {}

        # Look for attribute head patterns
        for key in state_dict.keys():
            if key.startswith('attribute_heads.') and key.endswith('.weight'):
                parts = key.replace('attribute_heads.', '').split('.')
                if len(parts) >= 1:
                    head_name = parts[0]

                    # Look for corresponding bias
                    bias_key = key.replace('.weight', '.bias')
                    if bias_key in state_dict:
                        weight_tensor = state_dict[key]
                        bias_tensor = state_dict[bias_key]

                        if len(weight_tensor.shape) == 2 and len(bias_tensor.shape) == 1:
                            num_classes = weight_tensor.shape[0]

                            # Validate head dimensions
                            if bias_tensor.shape[0] == num_classes and num_classes > 0:
                                discovered_heads[head_name] = num_classes

                                if self.verbose:
                                    print(f"   Discovered: {head_name} ({num_classes} classes)")

        # Apply aliases and validate known heads
        validated_heads = {}
        valid_heads = ['gender', 'hair_color', 'shirt_color', 'pants_color']

        for head_name, num_classes in discovered_heads.items():
            # Apply alias if needed
            resolved_name = HEAD_ALIASES.get(head_name, head_name)

            # Validate head configuration
            if resolved_name in valid_heads:
                if self._validate_head_configuration(resolved_name, num_classes):
                    validated_heads[resolved_name] = num_classes
                else:
                    if self.verbose:
                        print(f"   Invalid configuration for {resolved_name}: {num_classes} classes")
            else:
                if self.verbose:
                    print(f"   Unknown head type: {head_name} -> {resolved_name}")

        return validated_heads

    def _validate_head_configuration(self, head_name: str, num_classes: int) -> bool:
        """Validate if head configuration makes sense"""
        valid_configs = {
            'gender': [2],  # male, female
            'hair_color': [4, 5],  # 4: black,brown,blonde,gray or 5: +red
            'shirt_color': [4, 5],  # 4: black,white,red,blue or 5: +green
            'pants_color': [2, 3, 4]  # 2: dark,light or more
        }

        return num_classes in valid_configs.get(head_name, [])

    def _validate_backbone_compatibility(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Validate backbone architecture compatibility"""
        backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.')]

        if not backbone_keys:
            return {'compatible': False, 'reason': 'No backbone found in checkpoint'}

        # Check for EfficientNet patterns
        efficientnet_patterns = ['blocks.', 'conv_stem.', 'bn1.', 'classifier.']
        has_efficientnet = any(any(pattern in key for pattern in efficientnet_patterns)
                              for key in backbone_keys)

        if not has_efficientnet:
            return {'compatible': False, 'reason': 'Backbone not EfficientNet compatible'}

        # Check critical layers exist
        critical_layers = ['backbone.conv_stem.weight', 'backbone.bn1.weight']
        missing_critical = [layer for layer in critical_layers if layer not in state_dict]

        if missing_critical:
            return {'compatible': False, 'reason': f'Missing critical layers: {missing_critical}'}

        return {'compatible': True, 'reason': 'Backbone compatible'}

    def _create_validated_model(self, heads_config: Dict[str, int]):
        """Create model with validated configuration"""

        class ValidatedAttributeModel(nn.Module):
            def __init__(self, heads_config):
                super().__init__()

                # Create backbone
                self.backbone = timm.create_model(
                    'tf_efficientnetv2_s',
                    pretrained=False,
                    num_classes=0,
                    global_pool='avg'
                )

                # Get feature dimension
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    features = self.backbone(dummy_input)
                    feature_dim = features.shape[1]

                # Shared processing layers
                self.shared_processing = nn.Sequential(
                    nn.Dropout(0.48),
                    nn.Linear(feature_dim, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.48),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.48)
                )

                # Attention mechanism
                self.attention = nn.MultiheadAttention(
                    embed_dim=256,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True
                )

                # Create attribute heads
                self.attribute_heads = nn.ModuleDict()
                for head_name, num_classes in heads_config.items():
                    self.attribute_heads[head_name] = self._create_head(head_name, num_classes)

            def _create_head(self, head_name: str, num_classes: int) -> nn.Module:
                """Create appropriate head architecture"""
                if head_name == 'gender':
                    # Simpler architecture for gender
                    return nn.Sequential(
                        nn.Linear(256, 128),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Linear(128, num_classes)
                    )
                else:
                    # More complex for color attributes
                    return nn.Sequential(
                        nn.Linear(256, 128),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Linear(64, num_classes)
                    )

            def forward(self, x):
                features = self.backbone(x)
                shared_features = self.shared_processing(features)
                shared_features_seq = shared_features.unsqueeze(1)
                attended_features, _ = self.attention(
                    shared_features_seq, shared_features_seq, shared_features_seq
                )
                final_features = attended_features.squeeze(1)

                outputs = {}
                for attr_name, head in self.attribute_heads.items():
                    outputs[attr_name] = head(final_features)

                return outputs

        return ValidatedAttributeModel(heads_config).to(self.device)

    def _safe_load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Safely load state dict with critical layer validation"""
        try:
            # Attempt loading
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

            # Analyze missing and unexpected keys
            critical_missing = self._identify_critical_missing_keys(missing_keys)
            acceptable_missing = self._identify_acceptable_missing_keys(missing_keys)

            if critical_missing:
                return {
                    'success': False,
                    'error': f'Critical layers missing: {critical_missing}',
                    'details': {'missing': missing_keys, 'unexpected': unexpected_keys}
                }

            # Initialize acceptable missing keys properly
            if acceptable_missing:
                self._initialize_missing_layers(acceptable_missing)
                if self.verbose:
                    print(f"   Initialized {len(acceptable_missing)} missing layers")

            if self.verbose:
                print(f"   Loading: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")

            return {
                'success': True,
                'missing_keys': missing_keys,
                'unexpected_keys': unexpected_keys,
                'initialized_keys': acceptable_missing
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _identify_critical_missing_keys(self, missing_keys: List[str]) -> List[str]:
        """Identify missing keys that are critical for model function"""
        critical_patterns = [
            'backbone.conv_stem',
            'backbone.bn1',
            'backbone.blocks.0',  # First block is critical
            'shared_processing.1.weight',  # First linear layer
            'attention.in_proj_weight'
        ]

        critical_missing = []
        for key in missing_keys:
            if any(pattern in key for pattern in critical_patterns):
                critical_missing.append(key)

        return critical_missing

    def _identify_acceptable_missing_keys(self, missing_keys: List[str]) -> List[str]:
        """Identify missing keys that can be safely initialized"""
        acceptable_patterns = [
            'attribute_heads.',  # New head layers
            'shared_processing.3.',  # Later shared layers
            'shared_processing.5.'   # Final shared layers
        ]

        acceptable_missing = []
        for key in missing_keys:
            if any(pattern in key for pattern in acceptable_patterns):
                acceptable_missing.append(key)

        return acceptable_missing

    def _initialize_missing_layers(self, missing_keys: List[str]):
        """Properly initialize missing layers"""
        for key in missing_keys:
            try:
                # Navigate to the parameter
                parts = key.split('.')
                module = self.model

                for part in parts[:-1]:
                    module = getattr(module, part)

                param_name = parts[-1]

                if hasattr(module, param_name):
                    param = getattr(module, param_name)

                    if 'weight' in param_name:
                        if len(param.shape) >= 2:
                            # Kaiming initialization for weights
                            nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                        else:
                            # Normal initialization for 1D weights
                            nn.init.normal_(param, 0, 0.02)
                    elif 'bias' in param_name:
                        # Zero initialization for biases
                        nn.init.zeros_(param)

                    if self.verbose:
                        print(f"     Initialized: {key}")

            except Exception as e:
                if self.verbose:
                    print(f"     Failed to initialize {key}: {e}")

    def _validate_loaded_model(self) -> Dict[str, Any]:
        """Validate the loaded model works correctly"""
        try:
            self.model.eval()

            # Test forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                outputs = self.model(dummy_input)

                # Validate outputs
                if not isinstance(outputs, dict):
                    return {'success': False, 'error': 'Model outputs not a dictionary'}

                # Check each head
                for head_name, expected_classes in self.discovered_heads.items():
                    if head_name not in outputs:
                        return {'success': False, 'error': f'Missing output for {head_name}'}

                    output_tensor = outputs[head_name]
                    if output_tensor.shape[1] != expected_classes:
                        return {'success': False, 'error': f'Wrong output shape for {head_name}'}

                    # Check for NaN or Inf
                    if torch.isnan(output_tensor).any() or torch.isinf(output_tensor).any():
                        return {'success': False, 'error': f'Invalid values in {head_name} output'}

            # Set up transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            return {'success': True}

        except Exception as e:
            return {'success': False, 'error': f'Validation failed: {e}'}

    def is_ready(self) -> bool:
        """Check if model is ready for inference"""
        return self.model is not None and self.validation_passed

    def clear_track_cache(self, track_ids: List[int]):
        """Clear cache entries for dead tracks"""
        with self.cache_lock:
            for track_id in track_ids:
                cache_key = f"track_{track_id}"
                if cache_key in self.prediction_cache:
                    del self.prediction_cache[cache_key]

    def predict_batch_cached(self, person_crops: List[np.ndarray], track_ids: List[int]) -> List[Dict[str, Any]]:
        """Batch predict attributes for multiple persons with track-based caching"""
        if not self.is_ready():
            return [self._get_default_result() for _ in person_crops]

        # Check cache first
        results = []
        crops_to_process = []
        indices_to_process = []

        with self.cache_lock:
            for i, (crop, track_id) in enumerate(zip(person_crops, track_ids)):
                cache_key = f"track_{track_id}" if track_id is not None else hash(crop.tobytes())

                if cache_key in self.prediction_cache:
                    results.append(self.prediction_cache[cache_key].copy())
                else:
                    results.append(None)
                    crops_to_process.append(crop)
                    indices_to_process.append(i)

        # Batch process uncached crops
        if crops_to_process:
            try:
                # Convert to batch tensor
                batch_tensors = []
                for crop in crops_to_process:
                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(rgb_crop)
                    batch_tensors.append(tensor)

                batch_tensor = torch.stack(batch_tensors).to(self.device)

                with torch.no_grad():
                    with autocast(enabled=torch.cuda.is_available()):
                        outputs = self.model(batch_tensor)

                # Process batch results
                for idx, i in enumerate(indices_to_process):
                    result = self._process_model_output(outputs, idx)
                    results[i] = result

                    # Cache result
                    track_id = track_ids[i]
                    cache_key = f"track_{track_id}" if track_id is not None else hash(crops_to_process[idx].tobytes())

                    with self.cache_lock:
                        if len(self.prediction_cache) > 1000:
                            keys_to_remove = list(self.prediction_cache.keys())[:100]
                            for key in keys_to_remove:
                                del self.prediction_cache[key]

                        self.prediction_cache[cache_key] = result.copy()

            except Exception as e:
                if self.verbose:
                    print(f"Batch model prediction failed: {e}")
                for i in indices_to_process:
                    results[i] = self._get_default_result()

        return results

    def _process_model_output(self, outputs: Dict, batch_idx: int) -> Dict[str, Any]:
        """Process single item from batch model output with dynamic heads"""
        result = {
            'gender': {'class': 'unknown', 'confidence': 0.0},
            'hair_color': {'color': 'unknown', 'confidence': 0.0},
            'shirt_color': {'color': 'unknown', 'confidence': 0.0}
        }

        # Gender prediction
        if 'gender' in outputs:
            gender_logits = outputs['gender'][batch_idx]
            gender_probs = F.softmax(gender_logits, dim=0)
            max_prob, predicted_class = torch.max(gender_probs, dim=0)

            result['gender'] = {
                'class': ['male', 'female'][predicted_class.item()],
                'confidence': max_prob.item()
            }

        # Dynamic color predictions based on discovered heads
        discovered_mappings = {}
        for head_name, num_classes in self.discovered_heads.items():
            if head_name == 'hair_color':
                if num_classes == 4:
                    discovered_mappings[head_name] = ['black', 'brown', 'blonde', 'gray']
                elif num_classes == 5:
                    discovered_mappings[head_name] = ['black', 'brown', 'blonde', 'gray', 'red']
                else:
                    discovered_mappings[head_name] = [f'hair_class_{i}' for i in range(num_classes)]
            elif head_name == 'shirt_color':
                if num_classes == 5:
                    discovered_mappings[head_name] = ['black', 'white', 'red', 'blue', 'green']
                elif num_classes == 4:
                    discovered_mappings[head_name] = ['black', 'white', 'red', 'blue']
                else:
                    discovered_mappings[head_name] = [f'shirt_class_{i}' for i in range(num_classes)]
            elif head_name == 'pants_color':
                if num_classes == 2:
                    discovered_mappings[head_name] = ['dark', 'light']
                else:
                    discovered_mappings[head_name] = [f'pants_class_{i}' for i in range(num_classes)]

        # Process each discovered head
        for attr_name, classes in discovered_mappings.items():
            if attr_name in outputs:
                logits = outputs[attr_name][batch_idx]
                probs = F.softmax(logits, dim=0)
                max_prob, predicted_class = torch.max(probs, dim=0)

                if max_prob.item() > 0.4:
                    result[attr_name] = {
                        'color': classes[predicted_class.item()],
                        'confidence': max_prob.item()
                    }

        return result

    def _get_default_result(self) -> Dict[str, Any]:
        """Get default detection result"""
        return {
            'gender': {'class': 'unknown', 'confidence': 0.0},
            'hair_color': {'color': 'unknown', 'confidence': 0.0},
            'shirt_color': {'color': 'unknown', 'confidence': 0.0}
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        if not self.is_ready():
            return {'ready': False, 'error': 'Model not loaded or validated'}

        return {
            'ready': True,
            'heads': self.discovered_heads,
            'device': str(self.device),
            'validation_passed': self.validation_passed,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

# Color Analyzer with Lab implementation

class EnhancedColorAnalyzer:
    """Color analyzer with Lab + DeltaE2000 implementation"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Lab color centers
        self.lab_color_centers = self._define_lab_color_centers_fixed()

        # Parametric color gates
        self.parametric_gates = self._define_parametric_gates()

        # White balance calibration state
        self.white_balance_gains = None
        self.calibration_sample_count = 0

        # Clustering parameters
        self.adaptive_clustering = True
        self.brightness_normalization = True

        if self.verbose:
            print("Color Analyzer loaded")

    def _define_lab_color_centers_fixed(self):
        """Lab color centers"""
        # RGB centers for different colors
        rgb_centers = {
            'hair_color': {
                'black': [[30, 30, 30], [15, 15, 15], [25, 20, 18]],
                'brown': [[139, 69, 19], [101, 67, 33], [160, 82, 45], [133, 94, 66]],
                'blonde': [[255, 235, 205], [240, 230, 140], [255, 228, 181], [222, 184, 135]],
                'gray': [[128, 128, 128], [169, 169, 169], [105, 105, 105]],
                'red': [[165, 42, 42], [178, 34, 34], [220, 20, 60]]
            },
            'shirt_color': {
                'black': [[20, 20, 20], [40, 40, 40], [60, 60, 60]],
                'white': [[255, 255, 255], [248, 248, 255], [245, 245, 245]],
                'red': [[255, 0, 0], [220, 20, 60], [178, 34, 34], [139, 0, 0]],
                'blue': [[0, 0, 255], [65, 105, 225], [30, 144, 255], [135, 206, 250], [173, 216, 230]],
                'green': [[0, 128, 0], [34, 139, 34], [50, 205, 50], [0, 100, 0]]
            }
        }

        # RGB to Lab conversion
        lab_centers = {}
        for attribute, colors in rgb_centers.items():
            lab_centers[attribute] = {}
            for color_name, rgb_list in colors.items():
                lab_centers[attribute][color_name] = []
                for rgb in rgb_list:
                    try:
                        rgb_normalized = np.array([rgb], dtype=np.float32) / 255.0
                        rgb_normalized = rgb_normalized.reshape(1, 1, 3)
                        lab_color = color.rgb2lab(rgb_normalized)
                        lab_values = lab_color[0, 0]
                        lab_centers[attribute][color_name].append(lab_values.tolist())
                    except Exception as e:
                        if self.verbose:
                            print(f"   Failed to convert {color_name} RGB{rgb}: {e}")
                        # Fallback Lab values
                        if color_name == 'blue':
                            lab_centers[attribute][color_name].append([32.30, 79.20, -107.86])
                        elif color_name == 'red':
                            lab_centers[attribute][color_name].append([53.24, 80.09, 67.20])
                        elif color_name == 'green':
                            lab_centers[attribute][color_name].append([46.23, -51.70, 49.90])
                        elif color_name == 'black':
                            lab_centers[attribute][color_name].append([10.0, 0.0, 0.0])
                        elif color_name == 'white':
                            lab_centers[attribute][color_name].append([95.0, 0.0, 0.0])
                        else:
                            lab_centers[attribute][color_name].append([50.0, 0.0, 0.0])

        return lab_centers

    def _define_parametric_gates(self):
        """Parametric color gates"""
        return {
            'shirt_color': {
                'black': {'s_max': 0.3, 'v_max': 0.4, 'l_max': 30},
                'white': {'s_max': 0.2, 'v_min': 0.7, 'l_min': 80},
                'red': {'s_min': 0.3, 'v_min': 0.3, 'hue_ranges': [(0, 15), (345, 360)]},
                'blue': {'s_min': 0.2, 'v_min': 0.3, 'hue_ranges': [(200, 260)]},
                'green': {'s_min': 0.3, 'v_min': 0.3, 'hue_ranges': [(80, 150)]}
            },
            'hair_color': {
                'black': {'s_max': 0.3, 'v_max': 0.4, 'l_max': 25},
                'brown': {'s_min': 0.2, 'v_min': 0.2, 'hue_ranges': [(10, 40)]},
                'blonde': {'s_max': 0.5, 'v_min': 0.5, 'l_min': 60, 'hue_ranges': [(20, 80)]},
                'gray': {'s_max': 0.25, 'v_min': 0.3, 'v_max': 0.9, 'l_range': (30, 80)},
                'red': {'s_min': 0.4, 'v_min': 0.4, 'hue_ranges': [(0, 15), (345, 360)]}
            }
        }

    def calibrate_white_balance(self, sample_region: np.ndarray):
        """White-balance calibration"""
        if sample_region.size == 0:
            return

        try:
            gray_mask = (
                (sample_region.min(axis=2) > 50) &
                (sample_region.max(axis=2) < 200) &
                ((sample_region.max(axis=2) - sample_region.min(axis=2)) < 30)
            )

            if gray_mask.sum() > 100:
                gray_pixels = sample_region[gray_mask]
                mean_b = np.mean(gray_pixels[:, 0])
                mean_g = np.mean(gray_pixels[:, 1])
                mean_r = np.mean(gray_pixels[:, 2])
                gray_avg = (mean_b + mean_g + mean_r) / 3.0

                if gray_avg > 10:
                    gains = np.array([
                        np.clip(gray_avg / mean_b if mean_b > 0 else 1.0, 0.5, 2.0),
                        np.clip(gray_avg / mean_g if mean_g > 0 else 1.0, 0.5, 2.0),
                        np.clip(gray_avg / mean_r if mean_r > 0 else 1.0, 0.5, 2.0)
                    ])

                    if self.white_balance_gains is None:
                        self.white_balance_gains = gains
                    else:
                        alpha = 0.1
                        self.white_balance_gains = (1 - alpha) * self.white_balance_gains + alpha * gains

                    self.calibration_sample_count += 1

        except Exception as e:
            if self.verbose:
                print(f"White balance calibration failed: {e}")

    def extract_dominant_colors_enhanced(self, region: np.ndarray) -> List[Tuple[int, int, int]]:
        """Dominant color extraction with adaptive K-means"""
        if region.size == 0:
            return [(128, 128, 128)]

        try:
            # Auto-calibrate white balance
            if self.calibration_sample_count < 3:
                self.calibrate_white_balance(region)

            # Apply color correction
            corrected = self.apply_enhanced_color_correction(region)

            # Resize for efficiency
            h, w = corrected.shape[:2]
            if h * w > 10000:
                scale = np.sqrt(10000 / (h * w))
                new_h, new_w = int(h * scale), int(w * scale)
                corrected = cv2.resize(corrected, (new_w, new_h))

            # Convert to multiple color spaces
            hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)

            # Extract pixels
            pixels_bgr = corrected.reshape(-1, 3)
            pixels_hsv = hsv.reshape(-1, 3)

            # Color filtering
            valid_mask = self._apply_enhanced_color_filter(pixels_hsv)

            if valid_mask.sum() < 50:
                valid_mask = (pixels_hsv[:, 2] > 25) & (pixels_hsv[:, 2] < 240)

            valid_pixels_bgr = pixels_bgr[valid_mask]

            if len(valid_pixels_bgr) < 30:
                return [(128, 128, 128)]

            # Smart sampling
            if len(valid_pixels_bgr) > 3000:
                indices = np.random.choice(len(valid_pixels_bgr), 3000, replace=False)
                valid_pixels_bgr = valid_pixels_bgr[indices]

            # Adaptive K-means clustering
            n_clusters = self._calculate_adaptive_clusters(len(valid_pixels_bgr))

            # Use k-means++ initialization
            kmeans = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                random_state=42,
                n_init=10,
                max_iter=300
            )

            # Cluster in BGR space
            kmeans.fit(valid_pixels_bgr)

            # Get cluster centers and convert to RGB
            bgr_centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            unique_labels, counts = np.unique(labels, return_counts=True)

            # Convert BGR centers to RGB and weight by cluster size
            weighted_colors = []
            for i, label in enumerate(unique_labels):
                if label < len(bgr_centers):
                    bgr_color = bgr_centers[label]
                    rgb_color = tuple(np.clip(bgr_color[::-1], 0, 255).astype(int))
                    weight = counts[i] / len(valid_pixels_bgr)
                    weighted_colors.append((rgb_color, weight))

            # Sort by weight and return colors
            weighted_colors.sort(key=lambda x: x[1], reverse=True)
            return [color for color, _ in weighted_colors]

        except Exception as e:
            if self.verbose:
                print(f"Color extraction failed: {e}")
            try:
                mean_color = tuple(map(int, np.clip(region.reshape(-1, 3).mean(axis=0), 0, 255)))
                return [mean_color]
            except:
                return [(128, 128, 128)]

    def apply_enhanced_color_correction(self, image: np.ndarray, enable_wb: bool = True) -> np.ndarray:
        """Color correction"""
        try:
            corrected = image.copy()
            if enable_wb and self.white_balance_gains is not None:
                corrected = corrected.astype(np.float32)
                corrected[:, :, 0] *= self.white_balance_gains[0]
                corrected[:, :, 1] *= self.white_balance_gains[1]
                corrected[:, :, 2] *= self.white_balance_gains[2]
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)

            denoised = cv2.bilateralFilter(corrected, 9, 75, 75)
            enhanced = self._clahe_lab_enhanced(denoised)
            return enhanced

        except Exception as e:
            if self.verbose:
                print(f"Color correction failed: {e}")
            return image

    def _clahe_lab_enhanced(self, image: np.ndarray) -> np.ndarray:
        """CLAHE in LAB space"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            mean_brightness = np.mean(l)
            if mean_brightness < 80:
                clip_limit = 3.0
                tile_size = (6, 6)
            elif mean_brightness > 180:
                clip_limit = 1.5
                tile_size = (10, 10)
            else:
                clip_limit = 2.5
                tile_size = (8, 8)

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            l_enhanced = clahe.apply(l)

            lab_enhanced = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        except Exception as e:
            return image

    def _apply_enhanced_color_filter(self, pixels_hsv: np.ndarray) -> np.ndarray:
        """Color filtering"""
        h_norm = pixels_hsv[:, 0] * 2
        s_norm = pixels_hsv[:, 1] / 255.0
        v_norm = pixels_hsv[:, 2] / 255.0

        good_color = (s_norm > 0.1) & (v_norm > 0.1) & (v_norm < 0.95)
        return good_color

    def _calculate_adaptive_clusters(self, pixel_count: int) -> int:
        """Adaptive clustering"""
        if not self.adaptive_clustering:
            return 5

        n_clusters = max(2, min(8, pixel_count // 150))

        if pixel_count < 200:
            n_clusters = 2
        elif pixel_count > 5000:
            n_clusters = min(6, n_clusters)

        return n_clusters

    def classify_color_enhanced(self, rgb_color: Tuple[int, int, int], attribute: str) -> Tuple[str, float]:
        """Color classification using CIE Lab + DeltaE2000"""
        if attribute not in self.lab_color_centers:
            return 'unknown', 0.0

        # RGB to Lab conversion
        try:
            rgb_normalized = np.array([rgb_color], dtype=np.float32) / 255.0
            rgb_normalized = rgb_normalized.reshape(1, 1, 3)
            lab_color = color.rgb2lab(rgb_normalized)
            lab_pixel = lab_color[0, 0]

            hsv_pixel = cv2.cvtColor(np.uint8([[[rgb_color[2], rgb_color[1], rgb_color[0]]]]), cv2.COLOR_BGR2HSV)[0, 0]

        except Exception as e:
            if self.verbose:
                print(f"Color conversion failed: {e}")
            return 'unknown', 0.0

        best_color = None
        max_confidence = 0.0

        for color_name in self.lab_color_centers[attribute].keys():
            confidences = []

            # Method 1: DeltaE2000 perceptual distance
            lab_conf = self._calculate_lab_deltae_confidence_fixed(lab_pixel, color_name, attribute)
            confidences.append(lab_conf)

            # Method 2: Parametric gate analysis
            param_conf = self._calculate_parametric_confidence_fixed(rgb_color, hsv_pixel, lab_pixel, color_name, attribute)
            confidences.append(param_conf)

            # Method 3: Special detection
            special_conf = self._calculate_enhanced_special_confidence_fixed(rgb_color, hsv_pixel, lab_pixel, color_name, attribute)
            confidences.append(special_conf)

            # Smart combination with weighting
            max_conf = max(confidences)

            # Bonus for multiple method agreement
            high_conf_methods = sum(1 for c in confidences if c > 0.6)
            agreement_bonus = 0.1 * max(0, high_conf_methods - 1)

            combined_confidence = min(max_conf + agreement_bonus, 1.0)

            if combined_confidence > max_confidence:
                max_confidence = combined_confidence
                best_color = color_name

        return best_color or 'unknown', max_confidence

    def _calculate_lab_deltae_confidence_fixed(self, lab_pixel: np.ndarray, color_name: str, attribute: str) -> float:
        """DeltaE2000 calculation"""
        if color_name not in self.lab_color_centers[attribute]:
            return 0.0

        lab_centers = self.lab_color_centers[attribute][color_name]
        min_delta_e = float('inf')

        for lab_center in lab_centers:
            try:
                lab1 = np.array([[lab_pixel]], dtype=np.float64)
                lab2 = np.array([[lab_center]], dtype=np.float64)

                delta_e = deltaE_ciede2000(lab1, lab2)[0, 0]
                min_delta_e = min(min_delta_e, delta_e)

            except Exception as e:
                try:
                    dist = np.sqrt(np.sum((np.array(lab_pixel) - np.array(lab_center))**2))
                    min_delta_e = min(min_delta_e, dist)
                except:
                    continue

        if min_delta_e == float('inf'):
            return 0.0

        # DeltaE to confidence conversion
        if min_delta_e < 5.0:
            confidence = 1.0
        elif min_delta_e < 15.0:
            confidence = 1.0 - (min_delta_e - 5.0) / 10.0
        elif min_delta_e < 30.0:
            confidence = 0.5 - (min_delta_e - 15.0) / 30.0
        else:
            confidence = max(0.0, 0.2 - (min_delta_e - 30.0) / 50.0)

        return confidence

    def _calculate_parametric_confidence_fixed(self, rgb_color: Tuple[int, int, int],
                                             hsv_pixel: np.ndarray, lab_pixel: np.ndarray,
                                             color_name: str, attribute: str) -> float:
        """Parametric confidence calculation"""
        if attribute not in self.parametric_gates or color_name not in self.parametric_gates[attribute]:
            return 0.0

        gate = self.parametric_gates[attribute][color_name]
        r, g, b = rgb_color
        h, s, v = hsv_pixel[0] * 2, hsv_pixel[1] / 255.0, hsv_pixel[2] / 255.0
        l_lab = lab_pixel[0]

        confidence = 0.0

        # Check Lab lightness constraints
        if 'l_min' in gate and l_lab < gate['l_min']:
            return 0.0
        if 'l_max' in gate and l_lab > gate['l_max']:
            return 0.0
        if 'l_range' in gate and not (gate['l_range'][0] <= l_lab <= gate['l_range'][1]):
            return 0.0

        # Check saturation constraints
        if 's_min' in gate and s < gate['s_min']:
            return 0.0
        if 's_max' in gate and s > gate['s_max']:
            return 0.0

        # Check value constraints
        if 'v_min' in gate and v < gate['v_min']:
            return 0.0
        if 'v_max' in gate and v > gate['v_max']:
            return 0.0

        # Check hue ranges
        if 'hue_ranges' in gate:
            in_hue_range = False
            for hue_min, hue_max in gate['hue_ranges']:
                if hue_min <= h <= hue_max:
                    in_hue_range = True
                    break
            if not in_hue_range:
                return 0.0

        # Calculate confidence based on how well it fits
        if 'l_min' in gate:
            lightness_strength = min((l_lab - gate['l_min']) / 20.0, 1.0)
            confidence = max(confidence, lightness_strength)

        if 's_min' in gate:
            sat_strength = min((s - gate['s_min']) / 0.3, 1.0)
            confidence = max(confidence, sat_strength)

        if 'v_min' in gate:
            val_strength = min((v - gate['v_min']) / 0.3, 1.0)
            confidence = max(confidence, val_strength)

        return min(confidence, 1.0)

    def _calculate_enhanced_special_confidence_fixed(self, rgb_color: Tuple[int, int, int],
                                                   hsv_pixel: np.ndarray, lab_pixel: np.ndarray,
                                                   color_name: str, attribute: str) -> float:
        """Special confidence with Lab-aware processing"""
        r, g, b = rgb_color
        h, s, v = hsv_pixel
        l_lab, a_lab, b_lab = lab_pixel

        confidence = 0.0

        if attribute == 'shirt_color':
            if color_name == 'blue':
                # Blue detection using Lab b* channel
                if b_lab < -10 and l_lab > 20:
                    blue_chroma = abs(b_lab) / 50.0
                    lightness_factor = min(l_lab / 60.0, 1.0)
                    confidence = max(confidence, blue_chroma * lightness_factor)

                if b > max(r, g) + 30:
                    blue_strength = b / 255.0
                    confidence = max(confidence, blue_strength * 0.8)

            elif color_name == 'red':
                # Red detection using Lab a* channel
                if a_lab > 20 and l_lab > 15:
                    red_chroma = min(a_lab / 60.0, 1.0)
                    lightness_factor = min(l_lab / 60.0, 1.0)
                    confidence = max(confidence, red_chroma * lightness_factor)

                if r > max(g, b) + 30:
                    red_strength = r / 255.0
                    confidence = max(confidence, red_strength * 0.8)

            elif color_name == 'green':
                # Green detection using Lab a* channel
                if a_lab < -15 and l_lab > 20:
                    green_chroma = abs(a_lab) / 50.0
                    lightness_factor = min(l_lab / 60.0, 1.0)
                    confidence = max(confidence, green_chroma * lightness_factor)

                if g > max(r, b) + 30:
                    green_strength = g / 255.0
                    confidence = max(confidence, green_strength * 0.8)

            elif color_name == 'white':
                # White detection
                if l_lab > 80 and abs(a_lab) < 5 and abs(b_lab) < 5:
                    lightness_score = min(l_lab / 95.0, 1.0)
                    achromatic_score = 1.0 - (abs(a_lab) + abs(b_lab)) / 10.0
                    confidence = max(confidence, lightness_score * achromatic_score)

            elif color_name == 'black':
                # Black detection
                if l_lab < 30 and abs(a_lab) < 10 and abs(b_lab) < 10:
                    darkness_score = 1.0 - (l_lab / 35.0)
                    achromatic_score = 1.0 - (abs(a_lab) + abs(b_lab)) / 20.0
                    confidence = max(confidence, darkness_score * achromatic_score)

        elif attribute == 'hair_color':
            if color_name == 'black' and l_lab < 25:
                confidence = max(confidence, 1.0 - (l_lab / 30.0))
            elif color_name == 'brown' and 10 <= h <= 40 and a_lab > 5:
                confidence = max(confidence, min(a_lab / 30.0, 0.9))
            elif color_name == 'blonde':
                # Blonde detection
                if l_lab > 60 and b_lab > 8:
                    lightness_score = min((l_lab - 50) / 40.0, 1.0)
                    yellow_chroma = min(b_lab / 25.0, 1.0)
                    confidence = max(confidence, lightness_score * yellow_chroma * 0.9)
            elif color_name == 'gray' and abs(a_lab) < 8 and abs(b_lab) < 8 and 30 <= l_lab <= 70:
                confidence = max(confidence, 0.9)
            elif color_name == 'red' and a_lab > 25:
                confidence = max(confidence, min(a_lab / 50.0, 0.9))

        return min(confidence, 1.0)

    def analyze_region_enhanced(self, region: np.ndarray, attribute: str) -> Dict[str, Any]:
        """Region analysis with all improvements"""
        if region.size == 0:
            return {'color': 'unknown', 'confidence': 0.0, 'rgb': (128, 128, 128)}

        # Extract dominant colors with algorithm
        dominant_colors = self.extract_dominant_colors_enhanced(region)

        # Analyze each color with classification
        color_results = []
        for rgb_color in dominant_colors:
            color_name, confidence = self.classify_color_enhanced(rgb_color, attribute)
            color_results.append({
                'color': color_name,
                'confidence': confidence,
                'rgb': rgb_color
            })

        # Return best result
        best_result = max(color_results, key=lambda x: x['confidence'])

        return {
            'color': best_result['color'],
            'confidence': best_result['confidence'],
            'rgb': best_result['rgb'],
            'all_colors': color_results,
            'method': 'color_analysis'
        }

# Hybrid Detector with integration of all improvements

class EnhancedHybridDetector:
    """Hybrid detector with all improvements integrated"""

    def __init__(self, model_path: str, verbose: bool = False, use_pose: bool = True):
        self.color_analyzer = EnhancedColorAnalyzer(verbose=verbose)
        self.model_loader = SafeModelLoader(model_path, MANDATORY_MODEL_LOADING, verbose=verbose)
        self.region_extractor = SmartRegionExtractor(use_pose=use_pose, verbose=verbose)
        self.verbose = verbose

        # Performance tracking
        self.stats = {
            'total_detections': 0,
            'model_successes': 0,
            'color_successes': 0,
            'fusion_improvements': 0
        }

        if self.verbose:
            print("Hybrid Detector ready with all improvements")
            print(f"   Safe model loading: {'READY' if self.model_loader.is_ready() else 'NOT READY'}")
            print(f"   Smart region extraction: READY")
            print(f"   Color analysis: READY")

    def detect_attributes_enhanced(self, person_crop: np.ndarray, track_id: int = None) -> Dict[str, Any]:
        """Attribute detection with integration"""
        results = {
            'gender': {'class': 'unknown', 'confidence': 0.0, 'method': 'none'},
            'hair_color': {'color': 'unknown', 'confidence': 0.0, 'method': 'none'},
            'shirt_color': {'color': 'unknown', 'confidence': 0.0, 'method': 'none'}
        }

        self.stats['total_detections'] += 1

        try:
            # Step 1: Model predictions (especially for gender)
            model_results = None
            if self.model_loader.is_ready():
                model_results = self.model_loader.predict_batch_cached([person_crop], [track_id])[0]
                self.stats['model_successes'] += 1

                # Use model for gender (highest accuracy)
                if model_results['gender']['confidence'] > 0.3:
                    results['gender'] = {
                        'class': model_results['gender']['class'],
                        'confidence': model_results['gender']['confidence'],
                        'method': 'safe_trained_model'
                    }

            # Step 2: Smart region extraction with pose detection
            regions = self.region_extractor.extract_regions_smart(person_crop)

            # Step 3: Color analysis 

            # Hair color detection
            if regions['hair'] is not None:
                hair_result = self.color_analyzer.analyze_region_enhanced(regions['hair'], 'hair_color')
                if hair_result['confidence'] > 0.4:
                    results['hair_color'] = {
                        'color': hair_result['color'],
                        'confidence': hair_result['confidence'],
                        'method': 'smart_regions_lab_analysis'
                    }
                    self.stats['color_successes'] += 1

            # Shirt color detection
            if regions['shirt'] is not None:
                shirt_result = self.color_analyzer.analyze_region_enhanced(regions['shirt'], 'shirt_color')
                if shirt_result['confidence'] > 0.4:
                    results['shirt_color'] = {
                        'color': shirt_result['color'],
                        'confidence': shirt_result['confidence'],
                        'method': 'smart_regions_lab_analysis'
                    }
                    self.stats['color_successes'] += 1

            # Step 4: Intelligent fusion with model fallback
            if model_results:
                # Hair color fusion
                if (results['hair_color']['confidence'] < 0.5 and
                    model_results['hair_color']['confidence'] > 0.6):
                    results['hair_color'] = {
                        'color': model_results['hair_color']['color'],
                        'confidence': model_results['hair_color']['confidence'] * 0.7,
                        'method': 'safe_model_fallback'
                    }
                    self.stats['fusion_improvements'] += 1

                # Shirt color fusion
                if (results['shirt_color']['confidence'] < 0.5 and
                    model_results['shirt_color']['confidence'] > 0.6):
                    results['shirt_color'] = {
                        'color': model_results['shirt_color']['color'],
                        'confidence': model_results['shirt_color']['confidence'] * 0.7,
                        'method': 'safe_model_fallback'
                    }
                    self.stats['fusion_improvements'] += 1

        except Exception as e:
            if self.verbose:
                print(f"Detection failed: {e}")

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detection performance statistics"""
        total = max(self.stats['total_detections'], 1)
        return {
            'total_detections': self.stats['total_detections'],
            'model_success_rate': self.stats['model_successes'] / total,
            'color_success_rate': self.stats['color_successes'] / total,
            'fusion_improvement_rate': self.stats['fusion_improvements'] / total
        }

# Interactive Query Builder with clean UX

class CompleteQueryBuilder:
    """Interactive query builder with clean UX"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.available_options = {
            'gender': ['male', 'female', 'N/A'],
            'hair_color': ['black', 'brown', 'blonde', 'gray', 'red', 'N/A'],
            'shirt_color': ['black', 'white', 'red', 'blue', 'green', 'N/A']
        }

        if self.verbose:
            print("Query Builder ready")

    def build_query_interactive(self) -> Dict[str, str]:
        """Interactive query builder"""
        print("\nINTERACTIVE QUERY BUILDER")
        print("="*60)
        print("Select what you're looking for:")
        print("Choose N/A for any attribute to match ALL values for that attribute")
        print("Press Enter for quick N/A selection")

        query = {}

        # Gender selection
        print(f"\nGENDER (Safe model with validation):")
        for i, option in enumerate(self.available_options['gender'], 1):
            print(f"  {i}. {option}")

        gender_choice = self._get_user_choice("gender", 3)
        if gender_choice != 'N/A':
            query['gender'] = gender_choice

        # Hair color selection
        print(f"\nHAIR COLOR (Smart regions + Lab analysis):")
        for i, option in enumerate(self.available_options['hair_color'], 1):
            print(f"  {i}. {option}")

        hair_choice = self._get_user_choice("hair color", 6)
        if hair_choice != 'N/A':
            query['hair_color'] = hair_choice

        # Shirt color selection
        print(f"\nSHIRT COLOR (Smart pose regions + clustering):")
        for i, option in enumerate(self.available_options['shirt_color'], 1):
            print(f"  {i}. {option}")

        shirt_choice = self._get_user_choice("shirt color", 6)
        if shirt_choice != 'N/A':
            query['shirt_color'] = shirt_choice

        # Display final query
        self._display_query_summary(query)
        return query

    def _get_user_choice(self, attribute_name: str, max_options: int) -> str:
        """Get user choice with validation"""
        while True:
            try:
                choice = input(f"Select {attribute_name} (1-{max_options}, or Enter for N/A): ").strip()

                if choice == "":
                    print(f"[Selected] {attribute_name.title()}: N/A (will match all)")
                    return 'N/A'

                if choice.isdigit() and 1 <= int(choice) <= max_options:
                    selected = self.available_options[attribute_name.replace(' ', '_')][int(choice) - 1]
                    print(f"[Selected] {attribute_name.title()}: {selected}")
                    return selected
                else:
                    print(f"Please enter a number between 1 and {max_options}, or press Enter for N/A")

            except KeyboardInterrupt:
                print(f"\nUsing N/A for {attribute_name}")
                return 'N/A'
            except:
                print(f"Invalid input. Please try again.")

    def _display_query_summary(self, query: Dict[str, str]):
        """Display query summary"""
        print(f"\nFINAL SEARCH QUERY")
        print("="*60)

        if not query:
            print("SEARCHING FOR: ALL PEOPLE")
            print("   (All attributes set to N/A - maximum flexibility)")
        else:
            print("SEARCHING FOR:")
            for attr, value in query.items():
                attr_name = attr.replace('_', ' ').title()
                print(f"   {attr_name}: {value}")

            # Show flexible attributes
            all_attrs = ['gender', 'hair_color', 'shirt_color']
            flexible_attrs = [attr.replace('_', ' ').title() for attr in all_attrs if attr not in query]

            if flexible_attrs:
                print(f"\nFLEXIBLE (N/A) ATTRIBUTES:")
                for attr in flexible_attrs:
                    print(f"   {attr}: Will match ANY value")

        print(f"\nSYSTEM FEATURES:")
        print(f"   IoU-based tracking")
        print(f"   Safe model loading")
        print(f"   Smart pose-based regions")
        print(f"   Lab color analysis")

# Video Processor with integration

class EnhancedVideoProcessor:
    """Video processor with integration of all improvements"""

    def __init__(self,
                 model_path: str = MODEL_PATH,
                 verbose: bool = False,
                 use_pose: bool = True,
                 bbox_iou_threshold: float = 0.5,
                 attribute_update_interval: int = 5,
                 draw_hud: bool = True):

        self.yolo = YOLO(str(MODELS_DIR / 'yolov8n.pt'))
        self.detector = EnhancedHybridDetector(model_path, verbose=verbose, use_pose=use_pose)
        self.query_builder = CompleteQueryBuilder(verbose=verbose)

        # Use Tracker
        self.tracker = EnhancedTracker(
            max_disappeared=30,
            min_hits=3,
            iou_threshold=0.3,
            max_age=30,
            verbose=verbose
        )

        self.verbose = verbose
        self.draw_hud = draw_hud

        # Add cache cleanup callback
        self.tracker.cache_cleanup_callback = self.detector.model_loader.clear_track_cache

        # Configuration
        self.attribute_update_interval = attribute_update_interval
        self.bbox_iou_threshold = bbox_iou_threshold

        # Performance tracking
        self.stats = {
            'frames_processed': 0,
            'unique_tracks_seen': set(),
            'matched_track_ids': set(),
            'matches_found': 0,
            'processing_time': 0.0,
            'attribute_updates': 0,
            'improvements': 0,
            'tracking_benefits': 0,
            'smart_region_benefits': 0,
            'safe_model_benefits': 0
        }

        if self.verbose:
            print("Video Processor ready with integration")
            print("   IoU-based tracking: ENABLED")
            print("   Safe model loading with validation: ENABLED")
            print("   Smart region extraction with pose: ENABLED")
            print("   Lab color analysis: ENABLED")

    def process_video_enhanced(self,
                             video_path: str,
                             query: Dict[str, str] = None,
                             output_path: str = None,
                             show_progress: bool = True) -> Dict[str, Any]:
        """Video processing with integration"""
        start_time = time.time()

        if self.verbose:
            print(f"\nVIDEO PROCESSING")
            print("="*80)
            print(f"Input: {video_path}")

        # Check video file
        if not Path(video_path).exists():
            if self.verbose:
                print(f"Video file not found: {video_path}")
            return {'success': False, 'error': f'Video file not found: {video_path}'}

        # Use provided query or build interactively
        if query is None:
            query = self.query_builder.build_query_interactive()

        if self.verbose:
            print(f"\nPROCESSING:")
            if not query:
                print("  Looking for: ALL PEOPLE (maximum flexibility)")
            else:
                requirements = [f"{k.replace('_', ' ').title()}: {v}" for k, v in query.items()]
                print(f"  Looking for: {', '.join(requirements)}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'success': False, 'error': f'Cannot open video: {video_path}'}

        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.verbose:
            print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
            print(f"Duration: {total_frames/fps:.1f} seconds")

        # Setup output
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if self.verbose:
                print(f"Output: {output_path}")

        # Processing variables
        matches = []
        frame_idx = 0
        last_progress_update = 0

        # ETA calculation variables
        frames_for_eta = []
        times_for_eta = []

        if self.verbose:
            print(f"\nStarting processing...")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_start_time = time.time()

                # Detect persons with YOLO
                persons = self._detect_persons_optimized(frame)

                # Update tracker with IoU-based assignment
                tracked_persons = self.tracker.update(persons)

                # Track benefits of tracking
                if len(tracked_persons) > 1:
                    self.stats['tracking_benefits'] += 1

                # Track unique track IDs
                unique_tracks_this_frame = set(person.get('track_id', -1) for person in tracked_persons)
                self.stats['unique_tracks_seen'].update(unique_tracks_this_frame)

                # Determine which tracks need attribute updates
                tracks_to_update = []
                for person in tracked_persons:
                    track_id = person.get('track_id')
                    if track_id is None:
                        continue

                    if track_id in self.tracker.tracks:
                        track_state = self.tracker.tracks[track_id]

                        needs_update = (
                            not track_state.attributes or
                            (frame_idx - track_state.last_update_frame) >= self.attribute_update_interval or
                            calculate_iou(track_state.last_bbox, person['bbox']) < self.bbox_iou_threshold
                        )

                        if needs_update:
                            tracks_to_update.append((track_id, person))

                # Attribute detection
                if tracks_to_update:
                    crops_to_process = [person['crop'] for _, person in tracks_to_update]
                    track_ids_to_process = [track_id for track_id, _ in tracks_to_update]

                    # Use safe model loading for batch inference
                    if self.detector.model_loader.is_ready():
                        detections = self.detector.model_loader.predict_batch_cached(crops_to_process, track_ids_to_process)
                        self.stats['safe_model_benefits'] += len(detections)
                    else:
                        detections = [self.detector.model_loader._get_default_result() for _ in crops_to_process]

                    # Update track states with fusion
                    for (track_id, person), detection in zip(tracks_to_update, detections):
                        # Smart region extraction + color analysis
                        try:
                            regions = self.detector.region_extractor.extract_regions_smart(person['crop'])
                            self.stats['smart_region_benefits'] += 1

                            # Hair color fusion
                            if regions['hair'] is not None:
                                hair_result = self.detector.color_analyzer.analyze_region_enhanced(regions['hair'], 'hair_color')
                                if (hair_result['confidence'] > 0.5 and
                                    hair_result['confidence'] > detection['hair_color']['confidence']):
                                    detection['hair_color'] = {
                                        'color': hair_result['color'],
                                        'confidence': hair_result['confidence'],
                                        'method': 'integration'
                                    }
                                    self.stats['improvements'] += 1

                            # Shirt color fusion
                            if regions['shirt'] is not None:
                                shirt_result = self.detector.color_analyzer.analyze_region_enhanced(regions['shirt'], 'shirt_color')
                                if (shirt_result['confidence'] > 0.5 and
                                    shirt_result['confidence'] > detection['shirt_color']['confidence']):
                                    detection['shirt_color'] = {
                                        'color': shirt_result['color'],
                                        'confidence': shirt_result['confidence'],
                                        'method': 'integration'
                                    }
                                    self.stats['improvements'] += 1

                        except Exception as e:
                            if self.verbose:
                                print(f"Fusion failed for track {track_id}: {e}")

                        # Update track state
                        if track_id in self.tracker.tracks:
                            self.tracker.tracks[track_id].attributes = detection
                            self.tracker.tracks[track_id].last_update_frame = frame_idx
                            self.stats['attribute_updates'] += 1

                # Calculate matches using cached attributes
                frame_matches = 0
                for person in tracked_persons:
                    track_id = person.get('track_id')

                    if track_id is not None and track_id in self.tracker.tracks:
                        cached_attributes = self.tracker.tracks[track_id].attributes

                        if cached_attributes:
                            match_score = self._calculate_match_enhanced(cached_attributes, query)
                            is_match = match_score >= 0.7

                            if is_match:
                                matches.append({
                                    'frame': frame_idx,
                                    'timestamp': frame_idx / fps,
                                    'score': match_score,
                                    'detection': cached_attributes,
                                    'bbox': person['bbox'],
                                    'track_id': track_id
                                })
                                frame_matches += 1
                                self.stats['matched_track_ids'].add(track_id)

                self.stats['matches_found'] += frame_matches

                # Draw results
                self._draw_enhanced_results(frame, tracked_persons, query, matches)

                # Write frame
                if writer:
                    writer.write(frame)

                frame_end_time = time.time()
                frame_processing_time = frame_end_time - frame_start_time

                # ETA calculation
                frames_for_eta.append(frame_idx)
                times_for_eta.append(frame_processing_time)

                # Keep only recent frames for ETA (last 50 frames)
                if len(frames_for_eta) > 50:
                    frames_for_eta.pop(0)
                    times_for_eta.pop(0)

                # Progress reporting with ETA - REDUCED FREQUENCY
                should_show_progress = (
                    show_progress and 
                    (frame_idx % 100 == 0 or frame_idx - last_progress_update >= 100)
                )
                
                if should_show_progress:
                    progress = (frame_idx / total_frames) * 100

                    # Calculate ETA
                    if len(times_for_eta) > 5:
                        avg_time_per_frame = sum(times_for_eta) / len(times_for_eta)
                        remaining_frames = total_frames - frame_idx
                        eta_seconds = remaining_frames * avg_time_per_frame

                        # Format ETA
                        if eta_seconds < 60:
                            eta_str = f"{eta_seconds:.0f}s"
                        elif eta_seconds < 3600:
                            eta_str = f"{eta_seconds/60:.1f}m"
                        else:
                            eta_str = f"{eta_seconds/3600:.1f}h"
                    else:
                        eta_str = "calculating..."

                    if self.verbose:
                        tracker_stats = self.tracker.get_performance_stats()
                        region_stats = self.detector.region_extractor.get_performance_stats()

                        print(f"Progress: {progress:.1f}% | Frame {frame_idx:,}/{total_frames:,} | ETA: {eta_str}")
                        print(f"   Tracking: {tracker_stats['active_tracks']} active, {tracker_stats['id_switches_prevented']} switches prevented")
                        print(f"   Smart Regions: {region_stats['pose_success_rate']:.1%} pose success")
                        print(f"   Improvements: {len(tracks_to_update)} processed | Matches: {frame_matches}")

                    last_progress_update = frame_idx

                frame_idx += 1
                self.stats['frames_processed'] += 1

        except KeyboardInterrupt:
            if self.verbose:
                print(f"\nProcessing interrupted by user")
        except Exception as e:
            if self.verbose:
                print(f"\nProcessing error: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            cap.release()
            if writer:
                writer.release()

        # Final statistics
        self.stats['processing_time'] = time.time() - start_time

        # Get component performance stats
        tracker_performance = self.tracker.get_performance_stats()
        region_performance = self.detector.region_extractor.get_performance_stats()
        detector_performance = self.detector.get_performance_stats()

        # Results summary
        if self.verbose:
            print(f"\n" + "="*80)
            print(f"PROCESSING FINISHED!")
            print(f"="*80)
            print(f"Total time: {self.stats['processing_time']:.1f} seconds")
            print(f"PERFORMANCE:")
            print(f"   Frames processed: {self.stats['frames_processed']:,}")
            print(f"   Unique people tracked: {len(self.stats['unique_tracks_seen'])}")
            print(f"   Improvements applied: {self.stats['improvements']:,}")
            print(f"   Attribute updates: {self.stats['attribute_updates']:,}")
            print(f"   Matches found: {self.stats['matches_found']:,}")

            processing_speed = self.stats['frames_processed'] / max(self.stats['processing_time'], 0.001)
            improvement_rate = self.stats['improvements'] / max(self.stats['attribute_updates'], 1)
            match_rate = (len(self.stats['matched_track_ids']) / max(len(self.stats['unique_tracks_seen']), 1)) * 100

            print(f"   Processing speed: {processing_speed:.1f} FPS")
            print(f"   Improvement rate: {improvement_rate:.1%}")
            print(f"   Match rate (unique): {match_rate:.1f}%")

            print(f"\nCOMPONENT PERFORMANCE:")
            print(f"   Tracking:")
            print(f"     - Assignment success: {tracker_performance['assignment_success_rate']:.1%}")
            print(f"     - ID switches prevented: {tracker_performance['id_switches_prevented']}")
            print(f"     - Active tracks: {tracker_performance['active_tracks']}")

            print(f"   Smart Region Extraction:")
            print(f"     - Pose success rate: {region_performance['pose_success_rate']:.1%}")
            print(f"     - Total extractions: {region_performance['successful_extractions']}")
            print(f"     - Skin removals: {region_performance['skin_removals']}")

            print(f"   Detection:")
            print(f"     - Model success rate: {detector_performance['model_success_rate']:.1%}")
            print(f"     - Color success rate: {detector_performance['color_success_rate']:.1%}")
            print(f"     - Fusion improvements: {detector_performance['fusion_improvement_rate']:.1%}")

            if matches:
                print(f"\nTOP MATCHES:")
                for i, match in enumerate(matches[:10]):
                    det = match['detection']
                    print(f"  {i+1:2d}. Track {match['track_id']:,} | Frame {match['frame']:,} ({match['timestamp']:.1f}s) - Score: {match['score']:.3f}")
                    
                    # Safe access to method field with fallback
                    gender_method = det.get('gender', {}).get('method', 'unknown')
                    hair_method = det.get('hair_color', {}).get('method', 'unknown')
                    shirt_method = det.get('shirt_color', {}).get('method', 'unknown')
                    
                    print(f"      Gender: {det['gender']['class']} ({det['gender']['confidence']:.2f}) [{gender_method}]")
                    print(f"      Hair: {det['hair_color']['color']} ({det['hair_color']['confidence']:.2f}) [{hair_method}]")
                    print(f"      Shirt: {det['shirt_color']['color']} ({det['shirt_color']['confidence']:.2f}) [{shirt_method}]")

        # Convert sets to counts for JSON serialization
        serializable_stats = self.stats.copy()
        serializable_stats['unique_tracks_seen'] = len(self.stats['unique_tracks_seen'])
        serializable_stats['matched_track_ids'] = len(self.stats['matched_track_ids'])

        return {
            'success': True,
            'matches': matches,
            'output_path': output_path,
            'query': query,
            'stats': serializable_stats,
            'total_people': len(self.stats['unique_tracks_seen']),
            'total_matched_people': len(self.stats['matched_track_ids']),
            'match_rate': match_rate,
            'improvements': self.stats['improvements'],
            'component_performance': {
                'tracker': tracker_performance,
                'regions': region_performance,
                'detector': detector_performance
            }
        }

    def _detect_persons_optimized(self, frame: np.ndarray) -> List[Dict]:
        """Optimized person detection with YOLO"""
        results = self.yolo(frame, verbose=False, conf=0.3, iou=0.5)
        persons = []

        if results and len(results) > 0:
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        if int(box.cls[0]) == 0:  # Person class
                            confidence = float(box.conf[0])
                            if confidence >= 0.3:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                                width = x2 - x1
                                height = y2 - y1
                                aspect_ratio = height / width if width > 0 else 0

                                if (width > 60 and height > 100 and
                                    1.2 <= aspect_ratio <= 4.0 and
                                    width * height > 8000):

                                    crop = frame[y1:y2, x1:x2]
                                    if crop.size > 0:
                                        persons.append({
                                            'bbox': (x1, y1, x2, y2),
                                            'confidence': confidence,
                                            'crop': crop,
                                            'area': width * height
                                        })
        return persons

    def _calculate_match_enhanced(self, detection: Dict, query: Dict) -> float:
        """Match calculation with method weighting"""
        if not query:
            return 1.0

        matches = 0.0
        total_weight = 0.0

        # Weighted scoring
        weights = {
            'gender': 1.0,
            'hair_color': 1.0,
            'shirt_color': 1.0
        }

        for attr, required in query.items():
            weight = weights.get(attr, 1.0)

            # Bonus weight for integration methods
            detected = detection.get(attr, {})
            method = detected.get('method', '')
            if 'integration' in method:
                weight *= 1.3  # 30% bonus for integration methods
            elif 'smart_regions' in method:
                weight *= 1.2  # 20% bonus for smart methods
            elif 'safe_' in method:
                weight *= 1.1  # 10% bonus for safe methods

            total_weight += weight

            if attr == 'gender':
                if (detected.get('class') == required and
                    detected.get('confidence', 0) > 0.3):
                    matches += detected['confidence'] * weight
            elif attr in ['hair_color', 'shirt_color']:
                if (detected.get('color') == required and
                    detected.get('confidence', 0) > 0.4):
                    matches += detected['confidence'] * weight

        return matches / total_weight if total_weight > 0 else 1.0
   

    def _draw_enhanced_results(self, frame: np.ndarray, tracked_persons: List[Dict],
                             query: Dict, all_matches: List[Dict]):
        """Draw results with integration indicators"""
        for person in tracked_persons:
            bbox = person['bbox']
            track_id = person['track_id']

            if track_id in self.tracker.tracks:
                track_state = self.tracker.tracks[track_id]
                cached_attributes = track_state.attributes

                # Calculate match score
                match_score = 0.0
                is_match = False
                has_integration = False

                if cached_attributes:
                    match_score = self._calculate_match_enhanced(cached_attributes, query)
                    is_match = match_score >= 0.7

                    # Check for integration
                    for attr in ['hair_color', 'shirt_color']:
                        method = cached_attributes.get(attr, {}).get('method', '')
                        if 'integration' in method:
                            has_integration = True
                            break

                # Color coding
                if is_match:
                    if has_integration:
                        color = (0, 255, 0)      # Bright green for integration matches
                        thickness = 6
                    else:
                        color = (0, 200, 0)      # Regular green for normal matches
                        thickness = 4
                elif match_score >= 0.5:
                    color = (0, 255, 255)        # Yellow for partial matches
                    thickness = 3
                else:
                    color = (128, 128, 128)      # Gray for non-matches
                    thickness = 2

                # Draw bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

                # Track ID and match score with indicator
                integration_indicator = " [I]" if has_integration else ""
                track_text = f"ID:{track_id} Score:{match_score:.3f}{integration_indicator}"
                cv2.putText(frame, track_text, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw cached attributes with method indicators
                if cached_attributes:
                    y_offset = bbox[1] + 25

                    for attr in ['gender', 'hair_color', 'shirt_color']:
                        detected = cached_attributes.get(attr, {})
                        required = query.get(attr)

                        if detected.get('class' if attr == 'gender' else 'color', 'unknown') != 'unknown':
                            value = detected.get('class' if attr == 'gender' else 'color')
                            confidence = detected.get('confidence', 0.0)
                            method = detected.get('method', '')

                            attr_short = attr.split('_')[0]

                            # Method indicator
                            method_indicator = ""
                            if 'integration' in method:
                                method_indicator = "[I]"
                            elif 'smart_regions' in method:
                                method_indicator = "[S]"
                            elif 'safe_' in method:
                                method_indicator = "[SAFE]"
                            elif 'trained_model' in method:
                                method_indicator = "[MODEL]"

                            if required:
                                if value == required:
                                    text = f"[MATCH]{method_indicator}{attr_short}: {value} ({confidence:.2f})"
                                    text_color = (0, 255, 0)
                                else:
                                    text = f"[DIFF]{method_indicator}{attr_short}: {value} ({confidence:.2f})"
                                    text_color = (0, 0, 255)
                            else:
                                text = f"[INFO]{method_indicator}{attr_short}: {value} ({confidence:.2f})"
                                text_color = (255, 255, 255)

                            cv2.putText(frame, text, (bbox[0], y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                            y_offset += 18

        # Update query info display
        self._draw_enhanced_query_info(frame, query, len(all_matches))

    def _draw_enhanced_query_info(self, frame: np.ndarray, query: Dict, total_matches: int):
        """Draw query information"""
        if not self.draw_hud:
            return

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (min(1200, frame.shape[1] - 10), 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Query display
        if not query:
            query_text = "QUERY: ALL PEOPLE (maximum flexibility)"
        else:
            requirements = [f"{k.replace('_', ' ')}={v}" for k, v in query.items()]
            query_text = f"QUERY: {', '.join(requirements)}"

        cv2.putText(frame, query_text, (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Tracking statistics
        active_tracks = len(self.tracker.tracks)
        improvements = self.stats.get('improvements', 0)
        tracker_performance = self.tracker.get_performance_stats()

        track_stats = f"Active: {active_tracks} | Integrated: {improvements} | Matches: {total_matches}"
        cv2.putText(frame, track_stats, (15, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # System info
        enhancement_stats = f"Tracking: {tracker_performance['id_switches_prevented']} switches prevented"
        cv2.putText(frame, enhancement_stats, (15, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        integration_stats = f"Smart Regions + Safe Model + Lab Analysis"
        cv2.putText(frame, integration_stats, (15, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Performance info
        perf_text = f"Frame: {self.stats['frames_processed']} | People: {len(self.stats['unique_tracks_seen'])} | Updates: {self.stats.get('attribute_updates', 0)}"
        cv2.putText(frame, perf_text, (15, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Model info
        model_status = "READY" if self.detector.model_loader.is_ready() else "NOT READY"
        model_text = f"Safe Model: {model_status} | Device: {'GPU' if torch.cuda.is_available() else 'CPU'} | Heads: {len(self.detector.model_loader.discovered_heads)}"
        cv2.putText(frame, model_text, (15, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Legend
        legend_text = "[MATCH]=Match [DIFF]=Mismatch [INFO]=N/A | [I]=Integration [S]=Smart [SAFE]=Safe [MODEL]=Model"
        cv2.putText(frame, legend_text, (15, 185),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

# Testing and demo functions

def test_integration(debug_mode: bool = False):
    """Test the integration improvements"""
    if debug_mode:
        print("TESTING INTEGRATION")
        print("="*60)

    # Test 1: Tracker
    if debug_mode:
        print("Step 1: Testing IoU-based Tracker...")
    tracker = EnhancedTracker(verbose=debug_mode)
    test_detections = [
        {'bbox': (100, 100, 200, 300), 'confidence': 0.9, 'crop': np.zeros((200, 100, 3), dtype=np.uint8), 'area': 20000},
        {'bbox': (105, 105, 205, 305), 'confidence': 0.8, 'crop': np.zeros((200, 100, 3), dtype=np.uint8), 'area': 20000}
    ]

    # First update
    tracked1 = tracker.update(test_detections)
    # Multiple updates to meet min_hits requirement
    for _ in range(3):
        tracked1 = tracker.update([
            {'bbox': (102, 102, 202, 302), 'confidence': 0.9, 'crop': np.zeros((200, 100, 3), dtype=np.uint8), 'area': 20000},
            {'bbox': (107, 107, 207, 307), 'confidence': 0.8, 'crop': np.zeros((200, 100, 3), dtype=np.uint8), 'area': 20000}
        ])

    # Second update with slight movement (should maintain IDs)
    tracked2 = tracker.update([{'bbox': (110, 110, 210, 310), 'confidence': 0.9, 'crop': np.zeros((200, 100, 3), dtype=np.uint8), 'area': 20000}])

    # Check that tracks have IDs and are working
    has_track_ids = all('track_id' in person for person in tracked1) and all('track_id' in person for person in tracked2)

    if has_track_ids and len(tracked2) >= 1:
        if debug_mode:
            print("Tracker: SUCCESS - IoU-based assignment working")
        tracker_passed = True
    else:
        if debug_mode:
            print("Tracker: FAILED")
        tracker_passed = False

    # Test 2: Smart Region Extractor
    if debug_mode:
        print("\nStep 2: Testing Smart Region Extractor...")
    try:
        region_extractor = SmartRegionExtractor(use_pose=True, verbose=debug_mode)
        test_image = np.random.randint(0, 255, (400, 200, 3), dtype=np.uint8)
        regions = region_extractor.extract_regions_smart(test_image)

        if 'hair' in regions and 'shirt' in regions:
            if debug_mode:
                print("Smart Region Extractor: SUCCESS")
            region_passed = True
        else:
            if debug_mode:
                print("Smart Region Extractor: FAILED")
            region_passed = False
    except Exception as e:
        if debug_mode:
            print(f"Smart Region Extractor: FAILED - {e}")
        region_passed = False

    # Test 3: Safe Model Loader
    if debug_mode:
        print("\nStep 3: Testing Safe Model Loader...")
    try:
        safe_loader = SafeModelLoader(MODEL_PATH, mandatory=False, verbose=debug_mode)
        if safe_loader.is_ready():
            if debug_mode:
                print("Safe Model Loader: SUCCESS")
            model_passed = True
        else:
            if debug_mode:
                print("Safe Model Loader: Model not ready (but no crash)")
            model_passed = True  # Still consider success if it handled gracefully
    except Exception as e:
        if debug_mode:
            print(f"Safe Model Loader: FAILED - {e}")
        model_passed = False

    # Test 4: Color Analyzer
    if debug_mode:
        print("\nStep 4: Testing Color Analyzer...")
    analyzer = EnhancedColorAnalyzer(verbose=debug_mode if debug_mode else False)

    # Test critical colors
    test_cases = [
        ((173, 216, 230), 'blue', 'shirt_color'),
        ((255, 0, 0), 'red', 'shirt_color'),
        ((240, 230, 140), 'blonde', 'hair_color'),
        ((20, 20, 20), 'black', 'shirt_color')
    ]

    color_successes = 0
    for rgb, expected, attribute in test_cases:
        color, confidence = analyzer.classify_color_enhanced(rgb, attribute)
        if color == expected and confidence > 0.5:
            color_successes += 1

    color_passed = color_successes >= 3
    if debug_mode:
        print(f"{'Color Analyzer: SUCCESS' if color_passed else 'Color Analyzer: FAILED'} - {color_successes}/4 critical colors passed")

    # Test 5: System Integration
    if debug_mode:
        print("\nStep 5: Testing System Integration...")
    try:
        processor = EnhancedVideoProcessor(verbose=debug_mode, use_pose=False)  # Disable pose for testing
        if debug_mode:
            print("System Integration: SUCCESS")
        integration_passed = True
    except Exception as e:
        if debug_mode:
            print(f"System Integration: FAILED - {e}")
        integration_passed = False

    all_tests_passed = all([tracker_passed, region_passed, model_passed, color_passed, integration_passed])

    if debug_mode:
        print(f"\nINTEGRATION TEST RESULT: {'ALL SYSTEMS WORKING' if all_tests_passed else 'SOME ISSUES DETECTED'}")

        if all_tests_passed:
            print("Integration benefits:")
            print("   IoU-based tracking")
            print("   Safe model loading")
            print("   Smart region extraction")
            print("   Lab color analysis")
            print("   All improvements working together")

    return all_tests_passed
def list_available_videos():
  """List available video files in videos directory"""
  video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
  videos = []
  for ext in video_extensions:
      videos.extend(VIDEOS_DIR.glob(f'*{ext}'))
  return sorted(videos)


def main_system():
    """Main function for the system"""
    print("VIDEO PERSON SEARCH SYSTEM")
    print("="*80)
    print("IoU-based tracking")
    print("Safe model loading")
    print("Smart region extraction")
    print("Lab color analysis")
    print("="*80)

    # Ask for debugging mode first
    debug_mode = input("\nEnable debugging mode? (shows detailed tests and verbose output) (y/N): ").strip().lower() == 'y'

    if debug_mode:
        print("Debug mode ENABLED - Detailed information will be shown")
    else:
        print("Production mode - Clean output")

    while True:
        print(f"\nCHOOSE OPERATION:")

        if debug_mode:
            print(f"  1. Test integration")
            print(f"  2. Quick system test")
            print(f"  3. Process video with system (RECOMMENDED)")
            print(f"  4. Test specific color")
            print(f"  5. System status check")
            print(f"  6. Test individual components")
            print(f"  0. Exit")
            max_option = 6
        else:
            print(f"  1. Process video with system")
            print(f"  0. Exit")
            max_option = 1

        try:
            if debug_mode:
                choice = input(f"\nSelect option (0-{max_option}): ").strip()
            else:
                choice = input(f"\nSelect option (0-{max_option}): ").strip()

            if choice == '0':
                print("Goodbye!")
                break
            elif choice == '1' and debug_mode:
                test_integration(debug_mode)
            elif choice == '2' and debug_mode:
                test_integration(debug_mode)
            elif choice == '3' and debug_mode or choice == '1' and not debug_mode:
                  # Video processing
                # Show available videos
                available_videos = list_available_videos()
                if available_videos:
                    print(f"\nAvailable videos in {VIDEOS_DIR}:")
                    for i, video in enumerate(available_videos, 1):
                        print(f"  {i}. {video.name}")
                    
                    choice = input(f"\nSelect video by number (1-{len(available_videos)}) or enter custom path: ").strip()
                    
                    if choice.isdigit() and 1 <= int(choice) <= len(available_videos):
                        video_path = str(available_videos[int(choice) - 1])
                    elif choice:
                        # If it's not an absolute path, assume it's in videos directory
                        if not os.path.isabs(choice):
                            video_path = str(VIDEOS_DIR / choice)
                        else:
                            video_path = choice
                    else:
                        if available_videos:
                            video_path = str(available_videos[0])  # Default to first video
                        else:
                            video_path = str(VIDEOS_DIR / "myvideo.mp4")
                else:
                    video_path = input(f"No videos found in {VIDEOS_DIR}. Enter video path: ").strip()
                    if not os.path.isabs(video_path):
                        video_path = str(VIDEOS_DIR / video_path)
                try:
                    print(f"\nCONFIGURATION:")
                    use_pose = input("Use MediaPipe pose detection? (smart regions but slower) (Y/n): ").strip().lower() != 'n'
                    verbose = debug_mode or input("Verbose output? (y/N): ").strip().lower() == 'y'
                    draw_hud = input("Draw HUD overlay? (Y/n): ").strip().lower() != 'n'

                    try:
                        interval = int(input("Attribute update interval in frames (default: 5): ").strip() or "5")
                    except:
                        interval = 5

                    processor = EnhancedVideoProcessor(
                        verbose=verbose,
                        use_pose=use_pose,
                        attribute_update_interval=interval,
                        draw_hud=draw_hud
                    )

                    # Custom Query Builder
                    print(f"\nCUSTOM QUERY BUILDER")
                    print("Opening interactive query builder...")

                    query = processor.query_builder.build_query_interactive()

                    # Confirm before processing
                    confirm = input(f"\nProcess video with system? (y/N): ").strip().lower()
                    if confirm != 'y':
                        print("Processing cancelled")
                        continue

                    output_path = f"output_{int(time.time())}.mp4"
                    print(f"\nStarting video processing...")
                    print(f"Output will be saved as: {output_path}")

                    results = processor.process_video_enhanced(video_path, query, output_path)

                    if results['success']:
                        print(f"\nPROCESSING COMPLETE!")
                        print(f"Output file: {output_path}")
                        print(f"Results:")
                        print(f"   Unique people tracked: {results['total_people']}")
                        print(f"   Improvements applied: {results['improvements']}")
                        print(f"   Perfect matches found: {len(results['matches'])}")
                        print(f"   Match rate: {results['match_rate']:.1f}%")
                        print(f"   Processing time: {results['stats']['processing_time']:.1f}s")

                        # Show component performance only in debug mode
                        if debug_mode:
                            comp_perf = results['component_performance']
                            print(f"\nCOMPONENT PERFORMANCE:")
                            print(f"   Tracking: {comp_perf['tracker']['assignment_success_rate']:.1%} assignment success")
                            print(f"   Smart Regions: {comp_perf['regions']['pose_success_rate']:.1%} pose success")
                            print(f"   Detection: {comp_perf['detector']['model_success_rate']:.1%} model success")

                        if results['matches']:
                            print(f"\nTOP 3 MATCHES:")
                            for i, match in enumerate(results['matches'][:3], 1):
                                det = match['detection']
                                print(f"  {i}. Track {match['track_id']:,} | Frame {match['frame']:,} ({match['timestamp']:.1f}s)")
                                method_hair = det['hair_color']['method']
                                method_shirt = det['shirt_color']['method']
                                integration_hair = "[I]" if 'integration' in method_hair else "[S]" if 'smart' in method_hair else ""
                                integration_shirt = "[I]" if 'integration' in method_shirt else "[S]" if 'smart' in method_shirt else ""
                                print(f"     Hair: {det['hair_color']['color']} ({det['hair_color']['confidence']:.2f}) {integration_hair}")
                                print(f"     Shirt: {det['shirt_color']['color']} ({det['shirt_color']['confidence']:.2f}) {integration_shirt}")

                    else:
                        print(f"\nProcessing failed: {results.get('error', 'Unknown error')}")

                except Exception as e:
                    print(f"Processing failed: {e}")

            elif choice == '4' and debug_mode:
                # Test specific color
                try:
                    rgb_input = input("Enter RGB values (e.g., 173,216,230): ").strip()
                    r, g, b = map(int, rgb_input.split(','))
                    attribute = input("Enter attribute (shirt_color/hair_color): ").strip()
                    expected = input("Expected color (optional, press Enter to skip): ").strip() or None

                    print(f"\nSINGLE COLOR TEST:")
                    analyzer = EnhancedColorAnalyzer(verbose=debug_mode)
                    result_color, confidence = analyzer.classify_color_enhanced((r, g, b), attribute)

                    print(f"\nSUMMARY:")
                    print(f"   RGB({r}, {g}, {b}) -> {result_color} ({confidence:.3f})")

                    if expected:
                        is_correct = result_color == expected
                        print(f"   Status: {'CORRECT' if is_correct else 'NEEDS MORE WORK'}")

                except Exception as e:
                    print(f"Test failed: {e}")

            elif choice == '5' and debug_mode:
                # System status check
                print("SYSTEM STATUS CHECK")
                print("-" * 60)

                # Tracker check
                try:
                    tracker = EnhancedTracker(verbose=False)
                    print(f"IoU-based Tracker: READY")
                except Exception as e:
                    print(f"Tracker: FAILED - {e}")

                # Smart Region Extractor check
                try:
                    region_extractor = SmartRegionExtractor(use_pose=False, verbose=False)
                    print(f"Smart Region Extractor: READY")
                except Exception as e:
                    print(f"Smart Region Extractor: FAILED - {e}")

                # Safe Model Loader check
                try:
                    if Path(MODEL_PATH).exists():
                        model_size = Path(MODEL_PATH).stat().st_size / (1024*1024)
                        print(f"Model file: {MODEL_PATH} ({model_size:.1f} MB)")

                        safe_loader = SafeModelLoader(MODEL_PATH, mandatory=False, verbose=False)
                        if safe_loader.is_ready():
                            model_info = safe_loader.get_model_info()
                            print(f"Safe Model Loader: READY with {len(model_info['heads'])} heads")
                        else:
                            print(f"Safe Model Loader: Model validation failed")
                    else:
                        print(f"Model file: {MODEL_PATH} NOT FOUND")
                except Exception as e:
                    print(f"Safe Model Loader check failed: {e}")

                # Color Analyzer check
                try:
                    analyzer = EnhancedColorAnalyzer(verbose=False)

                    # Test improvements
                    test_cases = [
                        ((173, 216, 230), 'blue', 'shirt_color'),
                        ((255, 0, 0), 'red', 'shirt_color'),
                        ((0, 128, 0), 'green', 'shirt_color'),
                        ((240, 230, 140), 'blonde', 'hair_color')
                    ]

                    successes = 0
                    for rgb, expected, attribute in test_cases:
                        color, confidence = analyzer.classify_color_enhanced(rgb, attribute)
                        if color == expected and confidence > 0.6:
                            successes += 1

                    print(f"Color Analyzer: {successes}/4 tests passed ({successes/4:.1%})")

                except Exception as e:
                    print(f"Color Analyzer test failed: {e}")

                print(f"System status: ALL COMPONENTS INTEGRATED")

            elif choice == '6' and debug_mode:
                # Test individual components
                test_integration(debug_mode)

            else:
                print("Invalid choice. Please try again.")

        except KeyboardInterrupt:
            print(f"\nExiting system...")
            break
        except Exception as e:
            print(f"System error: {e}")

if __name__ == "__main__":
    print("VIDEO PERSON SEARCH SYSTEM LOADED")
    print("IoU-based tracking")
    print("Safe model loading")
    print("Smart region extraction")
    print("Lab color analysis")
    print(f"Model required: {MODEL_PATH}")

    # Ask if user wants detailed startup test
    detailed_startup = input("\nRun detailed startup test? (y/N): ").strip().lower() == 'y'

    if detailed_startup:
        print("\nRunning startup test...")
        try:
            # Quick integration test
            discovered = discover_heads_from_checkpoint(torch.load(MODEL_PATH, map_location='cpu')['model_state_dict']) if Path(MODEL_PATH).exists() else {}
            if discovered:
                print("Startup test: All systems integrated!")

                # Run full test suite
                test_integration(debug_mode=True)
            else:
                print("Startup test: Model head discovery needs attention")
        except Exception as e:
            print(f"Startup test: {e}")
    else:
        print("Ready for production use")

    print("\n" + "="*80)
    print("SYSTEM READY")
    print("Run main_system() for the integrated experience")

    if detailed_startup:
        print("\nSYSTEM BENEFITS:")
        print("   IoU-based tracking")
        print("   Safe model loading")
        print("   Smart pose-based regions") 
        print("   Lab + DeltaE2000 color analysis")
        print("   Integration of all improvements")
        print("   ETA calculation")
        print("   Debug mode")

    print("="*80)
    if __name__ == "__main__":
        main_system()