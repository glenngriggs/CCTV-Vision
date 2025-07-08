"""
Enhanced Integrated Pipeline
Combines detection, tracking, and attribute classification with interactive querying
"""

import cv2
import numpy as np
import torch
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import queue
from datetime import datetime
import logging

# Import our modules
from person_detector import PersonDetector, DetectionConfig
from person_tracker import PersonTracker, TrackingConfig, Track
from attribute_classifier import (
    AttributeClassifier, PersonAttributes, Gender, ShirtColor, HairColor,
    InteractiveQuerySystem
)


@dataclass
class EnhancedPipelineConfig:
    """Configuration for the enhanced pipeline"""
    # Detection
    detector_model: str = "yolov8n.pt"
    detection_confidence: float = 0.6
    detection_iou_threshold: float = 0.7
    
    # Tracking
    tracking_iou_threshold: float = 0.3
    max_disappeared: int = 30
    min_hits: int = 3
    use_kalman: bool = True
    
    # Attribute classification
    attribute_model: Optional[str] = None
    attribute_confidence: float = 0.5
    
    # Query system
    enable_interactive_query: bool = True
    auto_update_query: bool = False
    query_update_interval: int = 300  # frames
    
    # Performance
    target_fps: float = 30.0
    batch_processing: bool = True
    max_tracks_to_classify: int = 10
    
    # Visualization
    show_detection_boxes: bool = True
    show_track_ids: bool = True
    show_attributes: bool = True
    show_confidence_scores: bool = False
    highlight_matches: bool = True
    show_trails: bool = True
    
    # Recording and output
    save_output: bool = False
    output_directory: str = "output"
    save_annotations: bool = True
    save_statistics: bool = True
    
    # Hardware
    device: str = "auto"
    
    # Logging
    log_level: str = "INFO"
    save_logs: bool = True


@dataclass 
class TrackedPerson:
    """Enhanced tracked person with attributes and history"""
    track: Track
    attributes: Optional[PersonAttributes] = None
    attribute_history: deque = None
    last_attribute_update: float = 0.0
    matches_query: bool = False
    match_confidence: float = 0.0
    
    # Additional metadata
    first_classification_time: Optional[float] = None
    total_classifications: int = 0
    stable_attributes: Optional[PersonAttributes] = None
    
    def __post_init__(self):
        if self.attribute_history is None:
            self.attribute_history = deque(maxlen=5)
    
    def update_attributes(self, attributes: PersonAttributes):
        """Update person attributes with history tracking"""
        self.attributes = attributes
        self.attribute_history.append(attributes)
        self.last_attribute_update = time.time()
        self.total_classifications += 1
        
        if self.first_classification_time is None:
            self.first_classification_time = time.time()
        
        # Update stable attributes based on consensus
        self._update_stable_attributes()
    
    def _update_stable_attributes(self):
        """Update stable attributes based on history consensus"""
        if len(self.attribute_history) < 2:
            self.stable_attributes = self.attributes
            return
        
        # Simple voting mechanism - use most confident attributes
        best_attrs = None
        best_score = 0.0
        
        for attrs in self.attribute_history:
            score = (attrs.gender_confidence + attrs.shirt_confidence + attrs.hair_confidence) / 3
            if score > best_score:
                best_score = score
                best_attrs = attrs
        
        self.stable_attributes = best_attrs or self.attributes
    
    def get_display_attributes(self) -> Optional[PersonAttributes]:
        """Get attributes to display (stable if available, otherwise current)"""
        return self.stable_attributes or self.attributes


class EnhancedCVPipeline:
    """
    Enhanced computer vision pipeline with attribute classification and interactive querying
    """
    
    def __init__(self, config: EnhancedPipelineConfig):
        self.config = config
        
        # Setup logging
        self._setup_logging()
        
        # Set device
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        
        self.logger.info(f"🚀 Initializing Enhanced CV Pipeline on {self.device}")
        
        # Initialize components
        self._initialize_components()
        
        # Pipeline state
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.frame_count = 0
        self.total_processing_time = 0.0
        self.current_query = {}
        self.session_start_time = time.time()
        
        # Query system
        self.query_system = InteractiveQuerySystem()
        
        # Performance tracking
        self.performance_stats = defaultdict(list)
        self.frame_times = deque(maxlen=100)  # Keep last 100 frame times
        
        # Output setup
        if config.save_output:
            self.output_dir = Path(config.output_directory)
            self.output_dir.mkdir(exist_ok=True)
            self.annotation_log = []
        
        self.logger.info("✅ Enhanced CV Pipeline initialized successfully!")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper())
        
        # Create logger
        self.logger = logging.getLogger('EnhancedPipeline')
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.save_logs:
            log_dir = Path(self.config.output_directory) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(console_format)
            self.logger.addHandler(file_handler)
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        
        # Person detector
        self.logger.info("  📷 Loading person detector...")
        detector_config = DetectionConfig(
            model_path=self.config.detector_model,
            confidence_threshold=self.config.detection_confidence,
            iou_threshold=self.config.detection_iou_threshold,
            device=self.device
        )
        self.detector = PersonDetector(detector_config)
        
        # Person tracker
        self.logger.info("  🎬 Initializing person tracker...")
        tracker_config = TrackingConfig(
            iou_threshold=self.config.tracking_iou_threshold,
            max_disappeared=self.config.max_disappeared,
            min_hits=self.config.min_hits,
            use_kalman=self.config.use_kalman
        )
        self.tracker = PersonTracker(tracker_config)
        
        # Attribute classifier
        self.logger.info("  🎯 Loading attribute classifier...")
        if self.config.attribute_model:
            self.attribute_classifier = AttributeClassifier(
                model_path=self.config.attribute_model,
                device=self.device
            )
        else:
            self.logger.warning("  ⚠️ No attribute model specified, using demo classifier")
            self.attribute_classifier = AttributeClassifier(device=self.device)
    
    def setup_interactive_query(self):
        """Setup interactive query system"""
        if self.config.enable_interactive_query:
            self.logger.info("🔍 Setting up person search criteria...")
            self.current_query = self.query_system.get_user_preferences()
            return True
        return False
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame through the enhanced pipeline
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Processing results with tracks and attributes
        """
        frame_start_time = time.time()
        
        # Step 1: Person Detection
        detection_start = time.time()
        detections = self.detector.detect_persons(frame, return_crops=True)
        detection_time = time.time() - detection_start
        
        # Step 2: Person Tracking
        tracking_start = time.time()
        tracks = self.tracker.update(detections)
        tracking_time = time.time() - tracking_start
        
        # Step 3: Attribute Classification
        attribute_start = time.time()
        self._update_person_attributes(tracks, frame, detections.get('crops', []))
        attribute_time = time.time() - attribute_start
        
        # Step 4: Query Matching
        query_start = time.time()
        if self.current_query:
            self._update_query_matches()
        query_time = time.time() - query_start
        
        # Update performance stats
        total_time = time.time() - frame_start_time
        self.performance_stats['detection_time'].append(detection_time)
        self.performance_stats['tracking_time'].append(tracking_time)
        self.performance_stats['attribute_time'].append(attribute_time)
        self.performance_stats['query_time'].append(query_time)
        self.performance_stats['total_time'].append(total_time)
        self.frame_times.append(total_time)
        
        self.frame_count += 1
        self.total_processing_time += total_time
        
        # Calculate current FPS
        current_fps = 1.0 / total_time if total_time > 0 else 0
        
        # Log performance periodically
        if self.frame_count % 100 == 0:
            avg_fps = len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0
            self.logger.info(f"Frame {self.frame_count}: Current FPS: {current_fps:.1f}, Avg FPS: {avg_fps:.1f}")
        
        # Prepare results
        results = {
            'frame_id': self.frame_count,
            'timestamp': time.time(),
            'detections': detections,
            'tracks': tracks,
            'tracked_persons': self.tracked_persons,
            'processing_times': {
                'detection': detection_time,
                'tracking': tracking_time,
                'attribute': attribute_time,
                'query': query_time,
                'total': total_time
            },
            'query_matches': [tp for tp in self.tracked_persons.values() if tp.matches_query],
            'current_query': self.current_query,
            'performance': {
                'current_fps': current_fps,
                'avg_fps': len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0
            }
        }
        
        # Save annotations if requested
        if self.config.save_annotations:
            self._save_frame_annotations(results)
        
        return results
    
    def _update_person_attributes(self, tracks: List[Track], frame: np.ndarray, crops: List[np.ndarray]):
        """Update attributes for tracked persons with intelligent scheduling"""
        
        # Create mapping from track to crop
        track_to_crop = {}
        if crops and len(crops) == len(tracks):
            track_to_crop = {track.track_id: crop for track, crop in zip(tracks, crops)}
        else:
            # Extract crops from frame using bounding boxes
            for track in tracks:
                x1, y1, x2, y2 = track.bbox
                if x2 > x1 and y2 > y1:
                    # Add some padding
                    h, w = frame.shape[:2]
                    x1 = max(0, x1 - 5)
                    y1 = max(0, y1 - 5)
                    x2 = min(w, x2 + 5)
                    y2 = min(h, y2 + 5)
                    
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        track_to_crop[track.track_id] = crop
        
        # Update tracked persons
        current_track_ids = set()
        
        for track in tracks:
            current_track_ids.add(track.track_id)
            
            # Create or update tracked person
            if track.track_id not in self.tracked_persons:
                self.tracked_persons[track.track_id] = TrackedPerson(track=track)
            else:
                self.tracked_persons[track.track_id].track = track
        
        # Remove old tracked persons
        old_track_ids = set(self.tracked_persons.keys()) - current_track_ids
        for old_id in old_track_ids:
            del self.tracked_persons[old_id]
        
        # Classify attributes for tracks that need updates
        tracks_to_classify = []
        crops_to_classify = []
        
        current_time = time.time()
        
        # Priority-based selection for attribute classification
        candidates = []
        for track_id, tracked_person in self.tracked_persons.items():
            if track_id not in track_to_crop:
                continue
            
            # Calculate priority score
            time_since_update = current_time - tracked_person.last_attribute_update
            confidence_score = 0.0
            
            if tracked_person.attributes:
                confidence_score = (
                    tracked_person.attributes.gender_confidence +
                    tracked_person.attributes.shirt_confidence +
                    tracked_person.attributes.hair_confidence
                ) / 3
            
            # Higher priority for: new tracks, low confidence, long time since update
            priority = time_since_update * (1.0 - confidence_score) * (1.0 if tracked_person.attributes is None else 0.5)
            
            candidates.append((priority, tracked_person, track_to_crop[track_id]))
        
        # Sort by priority and select top candidates
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        for priority, tracked_person, crop in candidates[:self.config.max_tracks_to_classify]:
            # Only update if enough time has passed or never classified
            if (tracked_person.attributes is None or 
                current_time - tracked_person.last_attribute_update > 1.0):
                tracks_to_classify.append(tracked_person)
                crops_to_classify.append(crop)
        
        # Batch attribute classification
        if tracks_to_classify and crops_to_classify:
            if self.config.batch_processing and len(crops_to_classify) > 1:
                # Batch prediction
                try:
                    attributes_list = self.attribute_classifier.batch_predict(crops_to_classify)
                    
                    for tracked_person, attributes in zip(tracks_to_classify, attributes_list):
                        tracked_person.update_attributes(attributes)
                except Exception as e:
                    self.logger.warning(f"Batch prediction failed: {e}, falling back to individual predictions")
                    # Fallback to individual predictions
                    for tracked_person, crop in zip(tracks_to_classify, crops_to_classify):
                        try:
                            attributes = self.attribute_classifier.predict_attributes(crop)
                            tracked_person.update_attributes(attributes)
                        except Exception as e:
                            self.logger.error(f"Individual prediction failed for track {tracked_person.track.track_id}: {e}")
            else:
                # Individual predictions
                for tracked_person, crop in zip(tracks_to_classify, crops_to_classify):
                    try:
                        attributes = self.attribute_classifier.predict_attributes(crop)
                        tracked_person.update_attributes(attributes)
                    except Exception as e:
                        self.logger.error(f"Prediction failed for track {tracked_person.track.track_id}: {e}")
    
    def _update_query_matches(self):
        """Update which tracked persons match current query"""
        for tracked_person in self.tracked_persons.values():
            display_attrs = tracked_person.get_display_attributes()
            
            if display_attrs:
                matches = display_attrs.matches_query(
                    query_gender=self.current_query.get('gender'),
                    query_shirt=self.current_query.get('shirt_color'),
                    query_hair=self.current_query.get('hair_color'),
                    min_confidence=self.config.attribute_confidence
                )
                tracked_person.matches_query = matches
                
                # Calculate match confidence
                if matches:
                    tracked_person.match_confidence = display_attrs.get_match_score(
                        query_gender=self.current_query.get('gender'),
                        query_shirt=self.current_query.get('shirt_color'),
                        query_hair=self.current_query.get('hair_color')
                    )
                else:
                    tracked_person.match_confidence = 0.0
            else:
                tracked_person.matches_query = False
                tracked_person.match_confidence = 0.0
    
    def _save_frame_annotations(self, results: Dict[str, Any]):
        """Save frame annotations for later analysis"""
        frame_annotation = {
            'frame_id': results['frame_id'],
            'timestamp': results['timestamp'],
            'processing_times': results['processing_times'],
            'persons': []
        }
        
        for tracked_person in results['tracked_persons'].values():
            display_attrs = tracked_person.get_display_attributes()
            
            person_data = {
                'track_id': tracked_person.track.track_id,
                'bbox': tracked_person.track.bbox,
                'confidence': tracked_person.track.confidence,
                'attributes': display_attrs.to_dict() if display_attrs else None,
                'matches_query': tracked_person.matches_query,
                'match_confidence': tracked_person.match_confidence,
                'total_classifications': tracked_person.total_classifications
            }
            frame_annotation['persons'].append(person_data)
        
        self.annotation_log.append(frame_annotation)
    
    def visualize_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Visualize pipeline results with enhanced attribute display
        
        Args:
            frame: Input frame
            results: Processing results
            
        Returns:
            Annotated frame with comprehensive visualizations
        """
        annotated_frame = frame.copy()
        
        # Color schemes
        match_color = (0, 255, 255)    # Yellow for matches
        track_color = (0, 255, 0)      # Green for regular tracks
        tentative_color = (0, 165, 255) # Orange for tentative tracks
        text_color = (255, 255, 255)   # White text
        bg_color = (0, 0, 0)           # Black background
        
        # Draw tracked persons
        for tracked_person in results['tracked_persons'].values():
            track = tracked_person.track
            x1, y1, x2, y2 = track.bbox
            
            # Choose color based on track state and query match
            if tracked_person.matches_query and self.config.highlight_matches:
                color = match_color
                thickness = 3
            elif track.state == 'confirmed':
                color = track_color
                thickness = 2
            else:
                color = tentative_color
                thickness = 2
            
            # Draw bounding box
            if self.config.show_detection_boxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare text information
            text_lines = []
            
            # Track ID
            if self.config.show_track_ids:
                if tracked_person.matches_query:
                    text_lines.append(f"ID: {track.track_id} ⭐")
                else:
                    text_lines.append(f"ID: {track.track_id}")
            
            # Attributes
            if self.config.show_attributes:
                display_attrs = tracked_person.get_display_attributes()
                if display_attrs:
                    if self.config.show_confidence_scores:
                        text_lines.append(f"G: {display_attrs.gender.value} ({display_attrs.gender_confidence:.2f})")
                        text_lines.append(f"S: {display_attrs.shirt_color.value} ({display_attrs.shirt_confidence:.2f})")
                        text_lines.append(f"H: {display_attrs.hair_color.value} ({display_attrs.hair_confidence:.2f})")
                    else:
                        text_lines.append(f"G: {display_attrs.gender.value}")
                        text_lines.append(f"S: {display_attrs.shirt_color.value}")
                        text_lines.append(f"H: {display_attrs.hair_color.value}")
                    
                    # Show classification count
                    if tracked_person.total_classifications > 1:
                        text_lines.append(f"Cls: {tracked_person.total_classifications}")
            
            # Query match indicator
            if tracked_person.matches_query:
                text_lines.append(f"MATCH ({tracked_person.match_confidence:.2f})")
            
            # Draw text
            if text_lines:
                y_offset = y1 - 10
                for line in reversed(text_lines):  # Draw from bottom to top
                    text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    
                    # Ensure text stays within frame bounds
                    text_x = min(x1, annotated_frame.shape[1] - text_size[0] - 10)
                    text_y = max(y_offset, text_size[1] + 10)
                    
                    # Background rectangle
                    cv2.rectangle(annotated_frame,
                                 (text_x, text_y - text_size[1] - 5),
                                 (text_x + text_size[0] + 5, text_y + 2),
                                 bg_color, -1)
                    
                    # Text
                    cv2.putText(annotated_frame, line, (text_x + 2, text_y - 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    
                    y_offset -= text_size[1] + 8
            
            # Draw track trail
            if self.config.show_trails and len(track.bbox_history) > 1:
                trail_points = []
                for bbox in track.bbox_history:
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                    trail_points.append((cx, cy))
                
                # Draw trail lines with fading effect
                for i in range(1, len(trail_points)):
                    alpha = i / len(trail_points)
                    trail_color = tuple(int(c * alpha) for c in color)
                    cv2.line(annotated_frame, trail_points[i-1], trail_points[i], trail_color, 2)
        
        # Draw current query information
        if self.current_query:
            query_text = self.query_system.format_query_string()
            
            # Query background
            text_size = cv2.getTextSize(query_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated_frame,
                         (10, 10),
                         (20 + text_size[0], 40 + text_size[1]),
                         bg_color, -1)
            
            # Query text
            cv2.putText(annotated_frame, f"Query: {query_text}", (15, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)
            
            # Match count and statistics
            match_count = len(results['query_matches'])
            total_tracked = len([tp for tp in results['tracked_persons'].values() 
                               if tp.get_display_attributes() is not None])
            
            stats_text = f"Matches: {match_count}/{total_tracked}"
            cv2.putText(annotated_frame, stats_text, (15, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Performance information
        if 'performance' in results:
            perf = results['performance']
            
            perf_text = f"FPS: {perf['current_fps']:.1f} | Avg: {perf['avg_fps']:.1f}"
            cv2.putText(annotated_frame, perf_text,
                       (10, annotated_frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Component breakdown
            times = results['processing_times']
            comp_text = (f"Det: {times['detection']*1000:.1f}ms | "
                        f"Trk: {times['tracking']*1000:.1f}ms | "
                        f"Attr: {times['attribute']*1000:.1f}ms")
            cv2.putText(annotated_frame, comp_text,
                       (10, annotated_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Session information
        session_time = time.time() - self.session_start_time
        session_text = f"Session: {session_time//60:.0f}m {session_time%60:.0f}s | Frame: {self.frame_count}"
        cv2.putText(annotated_frame, session_text,
                   (10, annotated_frame.shape[0] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        return annotated_frame
    
    def run_video(self, video_source: str, 
                 output_path: Optional[str] = None,
                 max_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Run enhanced pipeline on video source
        
        Args:
            video_source: Video file path or camera index
            output_path: Optional output video path
            max_frames: Maximum frames to process
            
        Returns:
            Processing summary
        """
        self.logger.info(f"🎥 Starting enhanced pipeline on: {video_source}")
        
        # Setup interactive query if enabled
        if self.config.enable_interactive_query:
            self.setup_interactive_query()
        
        # Open video source
        if isinstance(video_source, str) and video_source.isdigit():
            cap = cv2.VideoCapture(int(video_source))
        else:
            cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"📹 Video: {width}x{height} @ {fps:.1f}fps")
        if total_frames > 0:
            self.logger.info(f"   Total frames: {total_frames}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            self.logger.info(f"💾 Saving output to: {output_path}")
        
        # Display instructions
        self.logger.info("🎮 Controls:")
        self.logger.info("   'q' - Quit")
        self.logger.info("   'c' - Change search criteria")
        self.logger.info("   's' - Save current frame")
        self.logger.info("   'r' - Reset statistics")
        self.logger.info("   'p' - Pause/Resume")
        self.logger.info("   SPACE - Step frame (when paused)")
        
        paused = False
        
        try:
            while cap.isOpened():
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                
                    # Check frame limit
                    if max_frames and self.frame_count >= max_frames:
                        self.logger.info(f"📊 Reached frame limit: {max_frames}")
                        break
                    
                    # Process frame
                    results = self.process_frame(frame)
                    
                    # Visualize results
                    annotated_frame = self.visualize_results(frame, results)
                else:
                    # Use last annotated frame when paused
                    if 'annotated_frame' in locals():
                        # Add pause indicator
                        pause_text = "PAUSED - Press 'p' to resume or SPACE to step"
                        cv2.putText(annotated_frame, pause_text, (width//2 - 200, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Save frame
                if writer and not paused:
                    writer.write(annotated_frame)
                
                # Display frame
                cv2.imshow('Enhanced CV Pipeline', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    self.logger.info("🛑 Stopping (user requested)")
                    break
                elif key == ord('c'):
                    # Change query
                    self.logger.info("🔄 Updating search criteria...")
                    cv2.destroyWindow('Enhanced CV Pipeline')  # Temporarily close window
                    self.current_query = self.query_system.get_user_preferences()
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = self.output_dir / f"frame_{timestamp}.jpg" if self.config.save_output else f"frame_{timestamp}.jpg"
                    cv2.imwrite(str(save_path), annotated_frame)
                    self.logger.info(f"📸 Frame saved: {save_path}")
                elif key == ord('r'):
                    # Reset statistics
                    self.logger.info("🔄 Resetting statistics...")
                    self.performance_stats.clear()
                    self.frame_times.clear()
                elif key == ord('p'):
                    # Pause/Resume
                    paused = not paused
                    self.logger.info(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
                elif key == ord(' ') and paused:
                    # Step frame when paused
                    ret, frame = cap.read()
                    if ret:
                        results = self.process_frame(frame)
                        annotated_frame = self.visualize_results(frame, results)
                
                # Auto-update query
                if (self.config.auto_update_query and 
                    self.frame_count % self.config.query_update_interval == 0):
                    self.logger.info("🔄 Auto-updating query...")
                    # Could implement automatic query updates here
                
                # Progress update
                if self.frame_count % 500 == 0 and self.frame_count > 0:
                    progress = (self.frame_count / total_frames) * 100 if total_frames > 0 else 0
                    matches = len([tp for tp in self.tracked_persons.values() if tp.matches_query])
                    self.logger.info(f"📈 Progress: Frame {self.frame_count}" + 
                                   (f" ({progress:.1f}%)" if total_frames > 0 else "") + 
                                   f" | Current matches: {matches}")
        
        except KeyboardInterrupt:
            self.logger.info("🛑 Stopping (interrupted)")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Save annotations and statistics
        if self.config.save_annotations and self.annotation_log:
            annotation_file = self.output_dir / f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(annotation_file, 'w') as f:
                json.dump(self.annotation_log, f, indent=2)
            self.logger.info(f"📝 Annotations saved: {annotation_file}")
        
        # Generate and save summary
        summary = self._generate_summary()
        
        if self.config.save_statistics:
            summary_file = self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            self.logger.info(f"📊 Summary saved: {summary_file}")
        
        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive processing summary"""
        
        # Calculate performance statistics
        perf_stats = {}
        for metric, values in self.performance_stats.items():
            if values:
                perf_stats[f"avg_{metric}"] = np.mean(values)
                perf_stats[f"std_{metric}"] = np.std(values)
                perf_stats[f"min_{metric}"] = np.min(values)
                perf_stats[f"max_{metric}"] = np.max(values)
        
        # Track statistics
        total_tracks = len(self.tracked_persons)
        tracks_with_attributes = sum(1 for tp in self.tracked_persons.values() if tp.get_display_attributes())
        current_matches = sum(1 for tp in self.tracked_persons.values() if tp.matches_query)
        
        # Attribute distribution
        attr_stats = self._calculate_attribute_statistics()
        
        # Session information
        session_duration = time.time() - self.session_start_time
        
        summary = {
            'session_info': {
                'start_time': datetime.fromtimestamp(self.session_start_time).isoformat(),
                'duration_seconds': session_duration,
                'frames_processed': self.frame_count,
                'average_fps': self.frame_count / self.total_processing_time if self.total_processing_time > 0 else 0
            },
            'performance_stats': perf_stats,
            'tracking_stats': {
                'total_tracks': total_tracks,
                'tracks_with_attributes': tracks_with_attributes,
                'current_matches': current_matches,
                'attribute_coverage': tracks_with_attributes / total_tracks if total_tracks > 0 else 0
            },
            'attribute_stats': attr_stats,
            'current_query': self.current_query,
            'component_stats': {
                'detector': self.detector.get_performance_stats(),
                'tracker': self.tracker.get_performance_stats(),
                'classifier': self.attribute_classifier.get_performance_stats()
            },
            'configuration': asdict(self.config)
        }
        
        return summary
    
    def _calculate_attribute_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about detected attributes"""
        gender_counts = defaultdict(int)
        shirt_counts = defaultdict(int)
        hair_counts = defaultdict(int)
        
        total_confidence_sum = 0.0
        total_attributes = 0
        
        for tracked_person in self.tracked_persons.values():
            display_attrs = tracked_person.get_display_attributes()
            if display_attrs:
                gender_counts[display_attrs.gender.value] += 1
                shirt_counts[display_attrs.shirt_color.value] += 1
                hair_counts[display_attrs.hair_color.value] += 1
                
                # Calculate average confidence
                avg_conf = (display_attrs.gender_confidence + 
                           display_attrs.shirt_confidence + 
                           display_attrs.hair_confidence) / 3
                total_confidence_sum += avg_conf
                total_attributes += 1
        
        return {
            'gender_distribution': dict(gender_counts),
            'shirt_color_distribution': dict(shirt_counts),
            'hair_color_distribution': dict(hair_counts),
            'total_classified': total_attributes,
            'average_confidence': total_confidence_sum / total_attributes if total_attributes > 0 else 0.0
        }
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print comprehensive summary"""
        print("\n" + "="*70)
        print("📊 ENHANCED PIPELINE SUMMARY")
        print("="*70)
        
        # Session info
        session = summary['session_info']
        print(f"🎬 Session Information:")
        print(f"   Duration: {session['duration_seconds']//60:.0f}m {session['duration_seconds']%60:.0f}s")
        print(f"   Frames processed: {session['frames_processed']}")
        print(f"   Average FPS: {session['average_fps']:.2f}")
        
        # Performance
        print(f"\n⚡ Performance:")
        perf = summary['performance_stats']
        if perf:
            print(f"   Avg total time: {perf.get('avg_total_time', 0)*1000:.1f}ms")
            print(f"   Avg detection: {perf.get('avg_detection_time', 0)*1000:.1f}ms")
            print(f"   Avg tracking: {perf.get('avg_tracking_time', 0)*1000:.1f}ms")
            print(f"   Avg attributes: {perf.get('avg_attribute_time', 0)*1000:.1f}ms")
        
        # Tracking
        tracking = summary['tracking_stats']
        print(f"\n👥 Tracking:")
        print(f"   Total tracks: {tracking['total_tracks']}")
        print(f"   With attributes: {tracking['tracks_with_attributes']}")
        print(f"   Current matches: {tracking['current_matches']}")
        print(f"   Coverage: {tracking['attribute_coverage']:.1%}")
        
        # Attributes
        print(f"\n🎯 Attributes:")
        attrs = summary['attribute_stats']
        print(f"   Average confidence: {attrs['average_confidence']:.3f}")
        print(f"   Gender: {attrs['gender_distribution']}")
        print(f"   Shirts: {attrs['shirt_color_distribution']}")
        print(f"   Hair: {attrs['hair_color_distribution']}")
        
        # Query
        if summary['current_query']:
            print(f"\n🔍 Current Query:")
            for key, value in summary['current_query'].items():
                print(f"   {key.replace('_', ' ').title()}: {value or 'Any'}")
        
        print("="*70)


def main():
    """Main function with enhanced CLI"""
    parser = argparse.ArgumentParser(description="Enhanced CCTV Computer Vision Pipeline")
    
    # Input/Output
    parser.add_argument('--video', type=str, default='0',
                       help='Video source (file path or camera index)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    
    # Model parameters
    parser.add_argument('--detector-model', type=str, default='yolov8n.pt',
                       help='YOLO model path')
    parser.add_argument('--attribute-model', type=str, default=None,
                       help='Attribute classification model path')
    parser.add_argument('--detection-conf', type=float, default=0.6,
                       help='Detection confidence threshold')
    parser.add_argument('--attribute-conf', type=float, default=0.5,
                       help='Attribute confidence threshold')
    
    # Pipeline options
    parser.add_argument('--no-interactive', action='store_true',
                       help='Disable interactive query setup')
    parser.add_argument('--batch-processing', action='store_true', default=True,
                       help='Enable batch processing for attributes')
    parser.add_argument('--save-output', action='store_true',
                       help='Save output and annotations')
    
    # Visualization
    parser.add_argument('--no-trails', action='store_true',
                       help='Disable track trails')
    parser.add_argument('--show-confidence', action='store_true',
                       help='Show confidence scores')
    
    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = EnhancedPipelineConfig(
        detector_model=args.detector_model,
        detection_confidence=args.detection_conf,
        attribute_model=args.attribute_model,
        attribute_confidence=args.attribute_conf,
        enable_interactive_query=not args.no_interactive,
        batch_processing=args.batch_processing,
        save_output=args.save_output,
        show_trails=not args.no_trails,
        show_confidence_scores=args.show_confidence,
        device=args.device
    )
    
    try:
        # Initialize pipeline
        pipeline = EnhancedCVPipeline(config)
        
        # Run processing
        summary = pipeline.run_video(
            video_source=args.video,
            output_path=args.output,
            max_frames=args.max_frames
        )
        
        # Print results
        pipeline.print_summary(summary)
        
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
