"""
Enhanced Person Detection Module with Custom YOLOv8/YOLOv11 Support
Handles person detection in video streams with optimizations for CCTV scenarios
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import json
import requests
import os
from dataclasses import dataclass
import logging


@dataclass
class DetectionConfig:
    """Configuration for person detection"""
    model_path: str = "yolov8n.pt"  # Model file path or name
    confidence_threshold: float = 0.5  # Minimum confidence for detections
    iou_threshold: float = 0.7  # IoU threshold for Non-Maximum Suppression
    max_detections: int = 100  # Maximum number of detections per frame
    device: Optional[str] = None  # Device to run on (cuda/cpu/auto)
    half_precision: bool = False  # Use half precision for faster inference
    filter_person_class: bool = True  # Only detect person class (class 0)
    
    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class ModelManager:
    """Manages YOLO model downloading and caching"""
    
    def __init__(self, cache_dir: str = "data/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard YOLOv8 model URLs
        self.model_urls = {
            'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
            'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
            'yolov8l.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
            'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt'
        }
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path to model, downloading if necessary"""
        model_path = self.cache_dir / model_name
        
        # If model exists locally, use it
        if model_path.exists():
            return model_path
        
        # If it's a standard model name, try to download
        if model_name in self.model_urls:
            return self._download_model(model_name)
        
        # If it's a full path, use as-is
        if Path(model_name).exists():
            return Path(model_name)
        
        # Let YOLO handle it (will download automatically)
        return Path(model_name)
    
    def _download_model(self, model_name: str) -> Path:
        """Download model from URL"""
        model_path = self.cache_dir / model_name
        url = self.model_urls[model_name]
        
        print(f"📥 Downloading {model_name}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded {model_name}")
            return model_path
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            raise


class PersonDetector:
    """
    Main person detector class that handles YOLO inference for person detection
    Optimized for CCTV and surveillance scenarios
    """
    
    def __init__(self, config: Union[DetectionConfig, Dict, str] = None):
        """
        Initialize person detector
        
        Args:
            config: Detection configuration (DetectionConfig object, dict, or model path string)
        """
        # Handle different config input types
        if isinstance(config, str):
            self.config = DetectionConfig(model_path=config)
        elif isinstance(config, dict):
            self.config = DetectionConfig(**config)
        elif config is None:
            self.config = DetectionConfig()
        else:
            self.config = config
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Performance tracking
        self.detection_times = []
        self.frame_count = 0
        
        # Load and initialize model
        self._load_model()
        
        print(f"🚀 Person Detector initialized")
        print(f"   Model: {self.config.model_path}")
        print(f"   Device: {self.config.device}")
        print(f"   Confidence: {self.config.confidence_threshold}")
        print(f"   IoU Threshold: {self.config.iou_threshold}")
    
    def _load_model(self):
        """Load YOLO model with error handling"""
        try:
            # Get model path (download if necessary)
            model_path = self.model_manager.get_model_path(self.config.model_path)
            
            print(f"📂 Loading model: {model_path}")
            
            # Initialize YOLO model
            self.model = YOLO(str(model_path))
            
            # Move to specified device
            self.model.to(self.config.device)
            
            # Enable half precision if requested
            if self.config.half_precision and self.config.device != 'cpu':
                self.model.half()
                print("⚡ Half precision enabled")
            
            # Get model information
            self.model_info = self._get_model_info()
            
            print(f"✅ Model loaded successfully")
            print(f"   Classes: {len(self.model_info['names'])}")
            print(f"   Parameters: {self.model_info['parameters']:,}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def _get_model_info(self) -> Dict:
        """Extract model information"""
        try:
            names = self.model.names if hasattr(self.model, 'names') else {}
            total_params = sum(p.numel() for p in self.model.model.parameters())
            
            return {
                'names': names,
                'parameters': total_params,
                'device': str(next(self.model.model.parameters()).device),
                'input_size': getattr(self.model.model, 'imgsz', 640)
            }
        except:
            return {'names': {}, 'parameters': 0, 'device': 'unknown', 'input_size': 640}
    
    def detect_persons(self, 
                      frame: np.ndarray, 
                      return_crops: bool = False) -> Dict:
        """
        Detect persons in a single frame
        
        Args:
            frame: Input image in BGR format
            return_crops: Whether to return cropped person images
            
        Returns:
            Dictionary containing detection results:
            - boxes: List of bounding boxes [x1, y1, x2, y2]
            - confidences: List of confidence scores
            - class_ids: List of class IDs (all 0 for person if filtered)
            - crops: List of cropped person images (if return_crops=True)
        """
        start_time = time.time()
        
        # Run YOLO inference
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            verbose=False
        )
        
        # Process results
        detections = self._process_results(results[0], frame, return_crops)
        
        # Update performance tracking
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        self.frame_count += 1
        
        # Add metadata
        detections['detection_time'] = detection_time
        detections['frame_id'] = self.frame_count
        
        return detections
    
    def _process_results(self, result, original_frame: np.ndarray, return_crops: bool) -> Dict:
        """Process YOLO detection results"""
        
        detections = {
            'boxes': [],
            'confidences': [],
            'class_ids': [],
            'crops': [] if return_crops else None
        }
        
        # Check if any detections were found
        if result.boxes is None:
            return detections
        
        # Extract detection data
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] format
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Get frame dimensions
        frame_height, frame_width = original_frame.shape[:2]
        
        # Process each detection
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            # Filter for person class only (class 0) if enabled
            if self.config.filter_person_class and cls_id != 0:
                continue
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = box.astype(int)
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(x1 + 1, min(x2, frame_width))
            y2 = max(y1 + 1, min(y2, frame_height))
            
            # Skip invalid detections
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Add detection
            detections['boxes'].append([x1, y1, x2, y2])
            detections['confidences'].append(float(conf))
            detections['class_ids'].append(int(cls_id))
            
            # Extract crop if requested
            if return_crops:
                crop = original_frame[y1:y2, x1:x2]
                detections['crops'].append(crop)
        
        return detections
    
    def batch_detect(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Detect persons in multiple frames efficiently using batch processing
        
        Args:
            frames: List of input frames in BGR format
            
        Returns:
            List of detection dictionaries for each frame
        """
        if not frames:
            return []
        
        start_time = time.time()
        
        # Run batch inference
        results = self.model(
            frames,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            verbose=False
        )
        
        # Process results for each frame
        batch_detections = []
        for i, (result, frame) in enumerate(zip(results, frames)):
            detections = self._process_results(result, frame, False)
            detections['detection_time'] = 0  # Will be updated below
            detections['frame_id'] = self.frame_count + i + 1
            batch_detections.append(detections)
        
        # Update performance tracking
        total_time = time.time() - start_time
        avg_time_per_frame = total_time / len(frames)
        
        for detection in batch_detections:
            detection['detection_time'] = avg_time_per_frame
            self.detection_times.append(avg_time_per_frame)
        
        self.frame_count += len(frames)
        
        print(f"📦 Batch processed {len(frames)} frames in {total_time:.3f}s "
              f"({len(frames)/total_time:.1f} FPS)")
        
        return batch_detections
    
    def visualize_detections(self, 
                           frame: np.ndarray, 
                           detections: Dict,
                           show_confidence: bool = True,
                           show_count: bool = True,
                           thickness: int = 2) -> np.ndarray:
        """
        Draw detection results on frame
        
        Args:
            frame: Input frame
            detections: Detection results from detect_persons()
            show_confidence: Whether to show confidence scores
            show_count: Whether to show person count
            thickness: Line thickness for bounding boxes
            
        Returns:
            Annotated frame with detection visualizations
        """
        annotated = frame.copy()
        
        # Colors for different confidence levels
        colors = {
            'high': (0, 255, 0),    # Green for high confidence (>0.8)
            'medium': (0, 255, 255), # Yellow for medium confidence (0.6-0.8)
            'low': (0, 165, 255)     # Orange for low confidence (<0.6)
        }
        
        # Draw each detection
        for i, (box, conf) in enumerate(zip(detections['boxes'], detections['confidences'])):
            x1, y1, x2, y2 = box
            
            # Choose color based on confidence
            if conf > 0.8:
                color = colors['high']
            elif conf > 0.6:
                color = colors['medium']
            else:
                color = colors['low']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            if show_confidence:
                label = f"Person {conf:.2f}"
            else:
                label = "Person"
            
            # Label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 5, y1), 
                         color, -1)
            
            # Label text
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show person count
        if show_count:
            count_text = f"Persons: {len(detections['boxes'])}"
            cv2.putText(annotated, count_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            cv2.putText(annotated, count_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return annotated
    
    def get_performance_stats(self) -> Dict:
        """Get detection performance statistics"""
        if not self.detection_times:
            return {
                'avg_fps': 0,
                'avg_detection_time': 0,
                'total_frames': 0,
                'min_detection_time': 0,
                'max_detection_time': 0
            }
        
        avg_detection_time = np.mean(self.detection_times)
        
        return {
            'avg_fps': 1.0 / avg_detection_time if avg_detection_time > 0 else 0,
            'avg_detection_time': avg_detection_time,
            'total_frames': self.frame_count,
            'min_detection_time': np.min(self.detection_times),
            'max_detection_time': np.max(self.detection_times),
            'std_detection_time': np.std(self.detection_times)
        }
    
    def benchmark(self, num_frames: int = 100, frame_size: Tuple[int, int] = (640, 640)) -> Dict:
        """
        Benchmark detector performance
        
        Args:
            num_frames: Number of test frames to process
            frame_size: Size of test frames (width, height)
            
        Returns:
            Benchmark results
        """
        print(f"🏃‍♂️ Benchmarking detector with {num_frames} frames of size {frame_size}")
        
        # Generate random test frames
        test_frames = []
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (*frame_size[::-1], 3), dtype=np.uint8)
            test_frames.append(frame)
        
        # Warm up (run a few frames to initialize GPU memory)
        print("🔥 Warming up...")
        for _ in range(5):
            self.detect_persons(test_frames[0])
        
        # Reset stats
        self.detection_times = []
        self.frame_count = 0
        
        # Run benchmark
        print("⏱️ Running benchmark...")
        start_time = time.time()
        
        for frame in test_frames:
            self.detect_persons(frame)
        
        total_time = time.time() - start_time
        
        # Calculate results
        stats = self.get_performance_stats()
        stats.update({
            'benchmark_total_time': total_time,
            'benchmark_fps': num_frames / total_time,
            'test_frames': num_frames,
            'frame_size': frame_size
        })
        
        print(f"📊 Benchmark Results:")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Average FPS: {stats['avg_fps']:.2f}")
        print(f"   Average Detection Time: {stats['avg_detection_time']*1000:.1f}ms")
        
        return stats
    
    def save_config(self, path: str):
        """Save current configuration to file"""
        config_dict = {
            'model_path': self.config.model_path,
            'confidence_threshold': self.config.confidence_threshold,
            'iou_threshold': self.config.iou_threshold,
            'max_detections': self.config.max_detections,
            'device': self.config.device,
            'half_precision': self.config.half_precision,
            'filter_person_class': self.config.filter_person_class,
            'model_info': self.model_info
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"💾 Configuration saved to: {path}")
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'PersonDetector':
        """Load detector from configuration file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Remove non-config fields
        config_dict.pop('model_info', None)
        
        config = DetectionConfig(**config_dict)
        return cls(config)


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Person Detection Module")
    print("=" * 50)
    
    # Test 1: Basic initialization
    print("\n1️⃣ Testing basic initialization...")
    try:
        detector = PersonDetector()
        print("✅ Basic initialization successful")
    except Exception as e:
        print(f"❌ Basic initialization failed: {e}")
        exit(1)
    
    # Test 2: Custom configuration
    print("\n2️⃣ Testing custom configuration...")
    try:
        custom_config = DetectionConfig(
            model_path="yolov8s.pt",
            confidence_threshold=0.7,
            iou_threshold=0.5
        )
        custom_detector = PersonDetector(custom_config)
        print("✅ Custom configuration successful")
    except Exception as e:
        print(f"❌ Custom configuration failed: {e}")
    
    # Test 3: Detection on sample image
    print("\n3️⃣ Testing detection on sample image...")
    try:
        # Create a test image with some random content
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some person-like shapes for visual testing
        cv2.rectangle(test_image, (100, 100), (200, 400), (120, 80, 60), -1)  # Person-like rectangle
        cv2.rectangle(test_image, (300, 150), (400, 350), (60, 120, 80), -1)  # Another person-like shape
        
        # Run detection
        detections = detector.detect_persons(test_image, return_crops=True)
        
        print(f"✅ Detection successful:")
        print(f"   Detected {len(detections['boxes'])} objects")
        print(f"   Confidences: {[f'{c:.2f}' for c in detections['confidences']]}")
        
        # Visualize results
        annotated = detector.visualize_detections(test_image, detections)
        
        # Save test result
        cv2.imwrite('test_detection_result.jpg', annotated)
        print("   💾 Test result saved as 'test_detection_result.jpg'")
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
    
    # Test 4: Performance benchmark
    print("\n4️⃣ Running performance benchmark...")
    try:
        benchmark_results = detector.benchmark(num_frames=50, frame_size=(640, 640))
        print("✅ Benchmark completed successfully")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
    
    # Test 5: Save/load configuration
    print("\n5️⃣ Testing configuration save/load...")
    try:
        detector.save_config('test_detector_config.json')
        loaded_detector = PersonDetector.from_config_file('test_detector_config.json')
        print("✅ Configuration save/load successful")
    except Exception as e:
        print(f"❌ Configuration save/load failed: {e}")
    
    print("\n🎉 All tests completed!")
    print("\nNext steps:")
    print("1. Run 'python person_detector.py' to test the module")
    print("2. Check the generated files: test_detection_result.jpg, test_detector_config.json")
    print("3. Proceed to the next component: Person Tracker")
