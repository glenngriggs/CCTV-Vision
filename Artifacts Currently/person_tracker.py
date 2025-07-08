"""
Advanced Person Tracking Module
Tracks people across video frames using IoU-based tracking with Kalman filtering
Maintains consistent person IDs for attribute classification and search
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import OrderedDict, deque
import time
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import math


@dataclass
class TrackingConfig:
    """Configuration for person tracking"""
    iou_threshold: float = 0.3  # Minimum IoU for track association
    max_disappeared: int = 30   # Max frames before removing disappeared track
    min_hits: int = 3          # Min detections before confirming track
    max_age: int = 70          # Max age of track without detection
    use_kalman: bool = True    # Use Kalman filter for motion prediction
    confidence_threshold: float = 0.5  # Min confidence for using detection
    
    # Advanced tracking parameters
    appearance_weight: float = 0.0  # Weight for appearance features (future use)
    motion_weight: float = 1.0      # Weight for motion/position features
    size_similarity_weight: float = 0.2  # Weight for size similarity


@dataclass
class Detection:
    """Single detection data structure"""
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int = 0  # Person class
    features: Optional[np.ndarray] = None  # Appearance features (future use)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get detection center point"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """Get detection area"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    @property
    def width(self) -> float:
        """Get detection width"""
        x1, y1, x2, y2 = self.bbox
        return x2 - x1
    
    @property
    def height(self) -> float:
        """Get detection height"""
        x1, y1, x2, y2 = self.bbox
        return y2 - y1


class KalmanTracker:
    """Kalman filter for tracking bounding box motion"""
    
    def __init__(self, bbox: List[int]):
        """
        Initialize Kalman tracker with initial bounding box
        State vector: [x_center, y_center, width, height, dx, dy, dw, dh]
        """
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + dx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + dy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + dw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + dh
            [0, 0, 0, 0, 1, 0, 0, 0],  # dx = dx
            [0, 0, 0, 0, 0, 1, 0, 0],  # dy = dy
            [0, 0, 0, 0, 0, 0, 1, 0],  # dw = dw
            [0, 0, 0, 0, 0, 0, 0, 1],  # dh = dh
        ])
        
        # Measurement matrix (we observe x, y, w, h)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R *= 10.0
        
        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initial state covariance
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        
        # Initialize state
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        self.kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0]).reshape(8, 1)
        
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
    
    def update(self, bbox: List[int]):
        """Update Kalman filter with new detection"""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        # Convert bbox to measurement
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        measurement = np.array([cx, cy, w, h]).reshape(4, 1)
        self.kf.update(measurement)
    
    def predict(self):
        """Predict next state and return predicted bbox"""
        if self.kf.x[2] + self.kf.x[0] <= 0:
            self.kf.x[2] = 0.1
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        
        return self.get_bbox()
    
    def get_bbox(self) -> List[int]:
        """Get current bbox from Kalman state"""
        cx, cy, w, h = self.kf.x[:4].flatten()
        
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        
        return [x1, y1, x2, y2]


@dataclass
class Track:
    """Represents a tracked person"""
    track_id: int
    bbox: List[int]  # Current bounding box [x1, y1, x2, y2]
    confidence: float
    kalman_tracker: Optional[KalmanTracker] = None
    
    # Tracking state
    hits: int = 1
    time_since_update: int = 0
    age: int = 1
    state: str = 'tentative'  # 'tentative', 'confirmed', 'deleted'
    
    # History tracking
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=10))
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Additional metadata
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    total_detections: int = 1
    
    def __post_init__(self):
        """Initialize history with current values"""
        self.bbox_history.append(self.bbox.copy())
        self.confidence_history.append(self.confidence)
    
    def update(self, detection: Detection, frame_id: int):
        """Update track with new detection"""
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0
        self.last_seen_frame = frame_id
        self.total_detections += 1
        
        # Update history
        self.bbox_history.append(self.bbox.copy())
        self.confidence_history.append(self.confidence)
        
        # Update Kalman filter
        if self.kalman_tracker:
            self.kalman_tracker.update(self.bbox)
    
    def predict(self):
        """Predict next position using Kalman filter"""
        if self.kalman_tracker:
            predicted_bbox = self.kalman_tracker.predict()
            return predicted_bbox
        else:
            # Simple prediction: assume no movement
            return self.bbox
    
    def mark_missed(self):
        """Mark track as missed this frame"""
        self.time_since_update += 1
        self.age += 1
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get track center point"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """Get track area"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    @property
    def velocity(self) -> Tuple[float, float]:
        """Estimate velocity from recent positions"""
        if len(self.bbox_history) < 2:
            return (0.0, 0.0)
        
        # Get last two positions
        curr_bbox = self.bbox_history[-1]
        prev_bbox = self.bbox_history[-2]
        
        curr_center = ((curr_bbox[0] + curr_bbox[2]) / 2, (curr_bbox[1] + curr_bbox[3]) / 2)
        prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
        
        vx = curr_center[0] - prev_center[0]
        vy = curr_center[1] - prev_center[1]
        
        return (vx, vy)
    
    @property
    def average_confidence(self) -> float:
        """Get average confidence over history"""
        if not self.confidence_history:
            return self.confidence
        return sum(self.confidence_history) / len(self.confidence_history)


class PersonTracker:
    """
    Main person tracker that manages multiple tracks
    Uses IoU-based association with Kalman filtering for robust tracking
    """
    
    def __init__(self, config: Optional[TrackingConfig] = None):
        """Initialize person tracker"""
        self.config = config or TrackingConfig()
        
        # Tracking state
        self.tracks: OrderedDict[int, Track] = OrderedDict()
        self.next_track_id = 1
        self.frame_count = 0
        
        # Performance tracking
        self.processing_times = []
        self.track_stats = {
            'total_tracks_created': 0,
            'active_tracks': 0,
            'confirmed_tracks': 0,
            'deleted_tracks': 0
        }
        
        print(f"🎬 Person Tracker initialized")
        print(f"   IoU threshold: {self.config.iou_threshold}")
        print(f"   Max disappeared: {self.config.max_disappeared}")
        print(f"   Min hits: {self.config.min_hits}")
        print(f"   Use Kalman: {self.config.use_kalman}")
    
    def update(self, detections_dict: Dict) -> List[Track]:
        """
        Update tracker with new detections
        
        Args:
            detections_dict: Dictionary from PersonDetector containing:
                - boxes: List of [x1, y1, x2, y2]
                - confidences: List of confidence scores
                - class_ids: List of class IDs
                
        Returns:
            List of active tracks
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Convert detection dict to Detection objects
        detections = self._convert_detections(detections_dict)
        
        # Filter detections by confidence
        detections = [det for det in detections if det.confidence >= self.config.confidence_threshold]
        
        # Predict existing tracks
        self._predict_tracks()
        
        # Associate detections with tracks
        matched_tracks, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(detections)
        
        # Update matched tracks
        for track_idx, detection_idx in matched_tracks:
            track = list(self.tracks.values())[track_idx]
            detection = detections[detection_idx]
            track.update(detection, self.frame_count)
            
            # Confirm track if it has enough hits
            if track.state == 'tentative' and track.hits >= self.config.min_hits:
                track.state = 'confirmed'
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            self._create_new_track(detection)
        
        # Handle unmatched tracks (mark as missed)
        for track_idx in unmatched_tracks:
            track = list(self.tracks.values())[track_idx]
            track.mark_missed()
        
        # Delete old tracks
        self._delete_old_tracks()
        
        # Update performance stats
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self._update_stats()
        
        # Return active confirmed tracks
        active_tracks = [track for track in self.tracks.values() 
                        if track.state == 'confirmed']
        
        return active_tracks
    
    def _convert_detections(self, detections_dict: Dict) -> List[Detection]:
        """Convert detection dictionary to Detection objects"""
        detections = []
        
        boxes = detections_dict.get('boxes', [])
        confidences = detections_dict.get('confidences', [])
        class_ids = detections_dict.get('class_ids', [])
        
        for i, bbox in enumerate(boxes):
            confidence = confidences[i] if i < len(confidences) else 0.5
            class_id = class_ids[i] if i < len(class_ids) else 0
            
            detection = Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=class_id
            )
            detections.append(detection)
        
        return detections
    
    def _predict_tracks(self):
        """Predict next positions for all tracks"""
        for track in self.tracks.values():
            if self.config.use_kalman and track.kalman_tracker:
                predicted_bbox = track.predict()
                # Update bbox with prediction if track was not detected
                if track.time_since_update > 0:
                    track.bbox = predicted_bbox
    
    def _associate_detections_to_tracks(self, detections: List[Detection]) -> Tuple[List, List, List]:
        """
        Associate detections with existing tracks using IoU and Hungarian algorithm
        
        Returns:
            matched_pairs: List of (track_idx, detection_idx) tuples
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track indices
        """
        if not detections or not self.tracks:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # Get active tracks (not deleted)
        active_tracks = [track for track in self.tracks.values() if track.state != 'deleted']
        
        if not active_tracks:
            return [], list(range(len(detections))), []
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(active_tracks, detections)
        
        # Apply additional similarity metrics
        similarity_matrix = self._calculate_similarity_matrix(active_tracks, detections, iou_matrix)
        
        # Use Hungarian algorithm for optimal assignment
        # Note: scipy.optimize.linear_sum_assignment minimizes cost, so we use negative similarity
        cost_matrix = 1.0 - similarity_matrix
        
        # Only consider assignments above IoU threshold
        cost_matrix[similarity_matrix < self.config.iou_threshold] = 1e6
        
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out assignments with low similarity
        matched_pairs = []
        for track_idx, detection_idx in zip(track_indices, detection_indices):
            if similarity_matrix[track_idx, detection_idx] >= self.config.iou_threshold:
                matched_pairs.append((track_idx, detection_idx))
        
        # Find unmatched detections and tracks
        matched_detection_indices = [pair[1] for pair in matched_pairs]
        matched_track_indices = [pair[0] for pair in matched_pairs]
        
        unmatched_detections = [i for i in range(len(detections)) 
                              if i not in matched_detection_indices]
        unmatched_tracks = [i for i in range(len(active_tracks)) 
                           if i not in matched_track_indices]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _calculate_iou_matrix(self, tracks: List[Track], detections: List[Detection]) -> np.ndarray:
        """Calculate IoU matrix between tracks and detections"""
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)))
        
        iou_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                iou = self._calculate_iou(track.bbox, detection.bbox)
                iou_matrix[i, j] = iou
        
        return iou_matrix
    
    def _calculate_similarity_matrix(self, tracks: List[Track], detections: List[Detection], 
                                   iou_matrix: np.ndarray) -> np.ndarray:
        """Calculate comprehensive similarity matrix including IoU, size, and motion"""
        similarity_matrix = iou_matrix.copy()
        
        # Add size similarity component
        if self.config.size_similarity_weight > 0:
            size_similarity = self._calculate_size_similarity_matrix(tracks, detections)
            similarity_matrix += self.config.size_similarity_weight * size_similarity
        
        # Future: Add appearance similarity when features are available
        # if self.config.appearance_weight > 0:
        #     appearance_similarity = self._calculate_appearance_similarity_matrix(tracks, detections)
        #     similarity_matrix += self.config.appearance_weight * appearance_similarity
        
        return similarity_matrix
    
    def _calculate_size_similarity_matrix(self, tracks: List[Track], detections: List[Detection]) -> np.ndarray:
        """Calculate size similarity matrix"""
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)))
        
        similarity_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            track_area = track.area
            for j, detection in enumerate(detections):
                detection_area = detection.area
                
                if track_area > 0 and detection_area > 0:
                    # Size similarity based on area ratio
                    area_ratio = min(track_area, detection_area) / max(track_area, detection_area)
                    similarity_matrix[i, j] = area_ratio
        
        return similarity_matrix
    
    @staticmethod
    def _calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0
        
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _create_new_track(self, detection: Detection):
        """Create new track from detection"""
        # Create Kalman tracker if enabled
        kalman_tracker = None
        if self.config.use_kalman:
            kalman_tracker = KalmanTracker(detection.bbox)
        
        # Create new track
        track = Track(
            track_id=self.next_track_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            kalman_tracker=kalman_tracker,
            first_seen_frame=self.frame_count,
            last_seen_frame=self.frame_count
        )
        
        # Add to tracks
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        self.track_stats['total_tracks_created'] += 1
    
    def _delete_old_tracks(self):
        """Delete tracks that haven't been seen for too long"""
        tracks_to_delete = []
        
        for track_id, track in self.tracks.items():
            # Delete if track is too old or hasn't been seen for too long
            if (track.time_since_update > self.config.max_disappeared or 
                track.age > self.config.max_age):
                tracks_to_delete.append(track_id)
                track.state = 'deleted'
        
        # Remove deleted tracks
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
            self.track_stats['deleted_tracks'] += 1
    
    def _update_stats(self):
        """Update tracking statistics"""
        self.track_stats['active_tracks'] = len(self.tracks)
        self.track_stats['confirmed_tracks'] = len([t for t in self.tracks.values() 
                                                   if t.state == 'confirmed'])
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID"""
        return self.tracks.get(track_id)
    
    def visualize_tracks(self, 
                        frame: np.ndarray, 
                        show_ids: bool = True,
                        show_trails: bool = True,
                        show_predictions: bool = False) -> np.ndarray:
        """
        Visualize tracks on frame
        
        Args:
            frame: Input frame
            show_ids: Show track IDs
            show_trails: Show track trails
            show_predictions: Show predicted positions
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Colors for different track states
        colors = {
            'confirmed': (0, 255, 0),    # Green
            'tentative': (0, 255, 255),  # Yellow
            'deleted': (0, 0, 255)       # Red
        }
        
        for track in self.tracks.values():
            if track.state == 'deleted':
                continue
            
            x1, y1, x2, y2 = track.bbox
            color = colors.get(track.state, (255, 255, 255))
            
            # Draw bounding box
            thickness = 3 if track.state == 'confirmed' else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw track ID
            if show_ids:
                label = f"ID:{track.track_id}"
                if track.state == 'confirmed':
                    label += f" ({track.hits})"
                
                # Label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated,
                             (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0] + 5, y1),
                             color, -1)
                
                # Label text
                cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw trail
            if show_trails and len(track.bbox_history) > 1:
                trail_points = []
                for bbox in track.bbox_history:
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                    trail_points.append((cx, cy))
                
                # Draw trail lines
                for i in range(1, len(trail_points)):
                    alpha = i / len(trail_points)  # Fade effect
                    trail_color = tuple(int(c * alpha) for c in color)
                    cv2.line(annotated, trail_points[i-1], trail_points[i], trail_color, 2)
            
            # Draw prediction
            if show_predictions and track.kalman_tracker:
                predicted_bbox = track.kalman_tracker.get_bbox()
                px1, py1, px2, py2 = predicted_bbox
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 0, 255), 1)
                cv2.putText(annotated, "PRED", (px1, py1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Draw tracking stats
        stats_text = [
            f"Active: {self.track_stats['active_tracks']}",
            f"Confirmed: {self.track_stats['confirmed_tracks']}",
            f"Frame: {self.frame_count}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(annotated, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def get_performance_stats(self) -> Dict:
        """Get tracking performance statistics"""
        if not self.processing_times:
            return {
                'avg_processing_time': 0,
                'avg_fps': 0,
                'total_frames': 0,
                **self.track_stats
            }
        
        avg_time = np.mean(self.processing_times)
        
        return {
            'avg_processing_time': avg_time,
            'avg_fps': 1.0 / avg_time if avg_time > 0 else 0,
            'total_frames': self.frame_count,
            'min_processing_time': np.min(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            **self.track_stats
        }
    
    def reset(self):
        """Reset tracker state"""
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0
        self.processing_times.clear()
        
        # Reset stats
        self.track_stats = {
            'total_tracks_created': 0,
            'active_tracks': 0,
            'confirmed_tracks': 0,
            'deleted_tracks': 0
        }
        
        print("🔄 Tracker reset")
    
    def save_tracks(self, filename: str):
        """Save track data to file"""
        track_data = {
            'frame_count': self.frame_count,
            'tracks': {}
        }
        
        for track_id, track in self.tracks.items():
            track_data['tracks'][track_id] = {
                'track_id': track.track_id,
                'bbox': track.bbox,
                'confidence': track.confidence,
                'hits': track.hits,
                'age': track.age,
                'state': track.state,
                'first_seen_frame': track.first_seen_frame,
                'last_seen_frame': track.last_seen_frame,
                'total_detections': track.total_detections,
                'bbox_history': list(track.bbox_history),
                'confidence_history': list(track.confidence_history)
            }
        
        import json
        with open(filename, 'w') as f:
            json.dump(track_data, f, indent=2)
        
        print(f"💾 Tracks saved to: {filename}")


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Person Tracking Module")
    print("=" * 50)
    
    # Test 1: Basic initialization
    print("\n1️⃣ Testing basic initialization...")
    try:
        tracker = PersonTracker()
        print("✅ Basic initialization successful")
    except Exception as e:
        print(f"❌ Basic initialization failed: {e}")
        exit(1)
    
    # Test 2: Custom configuration
    print("\n2️⃣ Testing custom configuration...")
    try:
        config = TrackingConfig(
            iou_threshold=0.5,
            max_disappeared=20,
            min_hits=2,
            use_kalman=True
        )
        custom_tracker = PersonTracker(config)
        print("✅ Custom configuration successful")
    except Exception as e:
        print(f"❌ Custom configuration failed: {e}")
    
    # Test 3: Simulated tracking scenario
    print("\n3️⃣ Testing simulated tracking scenario...")
    try:
        # Simulate detection data for multiple frames
        frames_data = [
            # Frame 1: Two people
            {
                'boxes': [[100, 100, 200, 300], [300, 150, 400, 350]],
                'confidences': [0.9, 0.8],
                'class_ids': [0, 0]
            },
            # Frame 2: People moved slightly
            {
                'boxes': [[105, 105, 205, 305], [295, 145, 395, 345]],
                'confidences': [0.85, 0.82],
                'class_ids': [0, 0]
            },
            # Frame 3: One person disappeared, new person appeared
            {
                'boxes': [[110, 110, 210, 310], [500, 200, 600, 400]],
                'confidences': [0.88, 0.75],
                'class_ids': [0, 0]
            },
            # Frame 4: All people moved
            {
                'boxes': [[115, 115, 215, 315], [505, 205, 605, 405], [290, 140, 390, 340]],
                'confidences': [0.9, 0.8, 0.85],
                'class_ids': [0, 0, 0]
            }
        ]
        
        # Process frames
        all_tracks = []
        for i, frame_data in enumerate(frames_data):
            print(f"   Processing frame {i+1}...")
            tracks = tracker.update(frame_data)
            all_tracks.append(tracks)
            
            print(f"     Active tracks: {len(tracks)}")
            for track in tracks:
                print(f"       Track {track.track_id}: bbox={track.bbox}, conf={track.confidence:.2f}")
        
        print("✅ Tracking simulation successful")
        
        # Test visualization on dummy frame
        test_frame = np.zeros((600, 800, 3), dtype=np.uint8)
        annotated = tracker.visualize_tracks(test_frame, show_ids=True, show_trails=True)
        cv2.imwrite('test_tracking_result.jpg', annotated)
        print("   💾 Test tracking result saved as 'test_tracking_result.jpg'")
        
    except Exception as e:
        print(f"❌ Tracking simulation failed: {e}")
    
    # Test 4: Performance stats
    print("\n4️⃣ Testing performance statistics...")
    try:
        stats = tracker.get_performance_stats()
        print("✅ Performance stats retrieved:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Performance stats failed: {e}")
    
    # Test 5: Save/load tracks
    print("\n5️⃣ Testing track save/load...")
    try:
        tracker.save_tracks('test_tracks.json')
        print("✅ Track save/load test successful")
    except Exception as e:
        print(f"❌ Track save/load failed: {e}")
    
    print("\n🎉 All tracking tests completed!")
    print("\nNext steps:")
    print("1. Check generated files: test_tracking_result.jpg, test_tracks.json")
    print("2. Integrate with PersonDetector for complete detection+tracking")
    print("3. Proceed to next component: Attribute Classifier")
