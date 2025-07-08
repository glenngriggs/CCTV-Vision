"""
Complete Usage Examples and Deployment Guide
Comprehensive examples for using the CCTV pipeline in various scenarios
"""

import cv2
import numpy as np
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import threading
import queue
import logging
from datetime import datetime, timedelta

# Import our modules
from person_detector import PersonDetector, DetectionConfig
from person_tracker import PersonTracker, TrackingConfig
from attribute_classifier import AttributeClassifier
from enhanced_pipeline import EnhancedCVPipeline, EnhancedPipelineConfig


class DeploymentManager:
    """Manages different deployment scenarios and configurations"""
    
    @staticmethod
    def get_real_time_config() -> EnhancedPipelineConfig:
        """Configuration optimized for real-time CCTV monitoring"""
        return EnhancedPipelineConfig(
            # Fast detection model
            detector_model="yolov8n.pt",  # Use your trained model: "data/models/person_detector_yolov8n_best.pt"
            detection_confidence=0.6,
            
            # Tracking optimized for real-time
            tracking_iou_threshold=0.3,
            max_disappeared=20,  # Shorter for real-time
            min_hits=2,         # Faster confirmation
            
            # Performance settings
            batch_processing=True,
            max_tracks_to_classify=8,
            target_fps=30.0,
            
            # Visualization
            show_trails=False,  # Disable trails for cleaner display
            show_confidence_scores=False,
            
            # Output
            save_output=False,  # Disable for real-time
            save_annotations=False
        )
    
    @staticmethod
    def get_high_accuracy_config() -> EnhancedPipelineConfig:
        """Configuration optimized for high accuracy analysis"""
        return EnhancedPipelineConfig(
            # Accurate detection model
            detector_model="yolov8m.pt",  # Use your trained model: "data/models/person_detector_yolov8m_best.pt"
            detection_confidence=0.4,  # Lower threshold for higher recall
            
            # Tracking optimized for accuracy
            tracking_iou_threshold=0.2,
            max_disappeared=50,  # Longer tracking
            min_hits=3,         # More confirmation
            use_kalman=True,
            
            # Performance settings
            batch_processing=True,
            max_tracks_to_classify=15,
            target_fps=15.0,   # Lower FPS for accuracy
            
            # Detailed visualization
            show_trails=True,
            show_confidence_scores=True,
            
            # Full logging
            save_output=True,
            save_annotations=True,
            save_statistics=True
        )
    
    @staticmethod
    def get_edge_device_config() -> EnhancedPipelineConfig:
        """Configuration optimized for edge devices (limited resources)"""
        return EnhancedPipelineConfig(
            # Lightweight model
            detector_model="yolov8n.pt",  # Use quantized model if available
            detection_confidence=0.7,  # Higher threshold to reduce processing
            
            # Minimal tracking
            tracking_iou_threshold=0.4,
            max_disappeared=15,
            min_hits=2,
            use_kalman=False,  # Disable Kalman for speed
            
            # Reduced processing
            batch_processing=False,  # Individual processing
            max_tracks_to_classify=3,
            target_fps=15.0,
            
            # Minimal visualization
            show_detection_boxes=True,
            show_track_ids=True,
            show_attributes=False,  # Disable attributes for speed
            show_trails=False,
            
            # No output saving
            save_output=False,
            save_annotations=False,
            device="cpu"  # Force CPU for edge devices
        )


class ScenarioExamples:
    """Examples for different use case scenarios"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def example_1_basic_detection(self):
        """Example 1: Basic person detection on a video file"""
        print("🎯 Example 1: Basic Person Detection")
        print("-" * 40)
        
        # Initialize detector
        config = DetectionConfig(
            model_path="yolov8n.pt",  # Replace with your trained model
            confidence_threshold=0.6
        )
        detector = PersonDetector(config)
        
        # Process video file
        video_path = "sample_video.mp4"  # Replace with your video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"⚠️ Could not open video: {video_path}")
            print("   Creating synthetic video for demonstration...")
            
            # Create synthetic video for demo
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Add person-like shapes
            cv2.rectangle(frame, (100, 100), (200, 300), (120, 80, 60), -1)
            cv2.rectangle(frame, (300, 150), (400, 350), (60, 120, 80), -1)
            
            detections = detector.detect_persons(frame, return_crops=True)
            annotated = detector.visualize_detections(frame, detections)
            
            print(f"✅ Detected {len(detections['boxes'])} people in synthetic frame")
            cv2.imwrite('example_1_result.jpg', annotated)
            print("💾 Result saved as 'example_1_result.jpg'")
            return
        
        frame_count = 0
        total_detections = 0
        
        while cap.isOpened() and frame_count < 100:  # Process first 100 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect people
            detections = detector.detect_persons(frame)
            total_detections += len(detections['boxes'])
            
            # Visualize every 10th frame
            if frame_count % 10 == 0:
                annotated = detector.visualize_detections(frame, detections)
                cv2.imwrite(f'example_1_frame_{frame_count}.jpg', annotated)
                print(f"Frame {frame_count}: {len(detections['boxes'])} people detected")
            
            frame_count += 1
        
        cap.release()
        
        # Statistics
        stats = detector.get_performance_stats()
        print(f"\n📊 Detection Summary:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Total detections: {total_detections}")
        print(f"   Average detections per frame: {total_detections/frame_count:.1f}")
        print(f"   Average FPS: {stats['avg_fps']:.2f}")
    
    def example_2_tracking_with_attributes(self):
        """Example 2: Person tracking with attribute classification"""
        print("\n🎯 Example 2: Tracking with Attributes")
        print("-" * 40)
        
        # Configure pipeline for tracking and attributes
        config = EnhancedPipelineConfig(
            detector_model="yolov8n.pt",
            detection_confidence=0.6,
            attribute_model=None,  # Will use demo classifier
            enable_interactive_query=False,
            save_output=False
        )
        
        pipeline = EnhancedCVPipeline(config)
        
        # Simulate video frames with moving people
        frames = []
        for i in range(30):
            frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            
            # Add moving people
            offset_x = i * 5
            offset_y = i * 2
            
            # Person 1 (moving right)
            cv2.rectangle(frame, (100 + offset_x, 100), (180 + offset_x, 300), (120, 80, 60), -1)
            cv2.rectangle(frame, (110 + offset_x, 80), (170 + offset_x, 120), (255, 255, 0), -1)  # Yellow hair
            
            # Person 2 (moving down-right)
            cv2.rectangle(frame, (300 + offset_x//2, 150 + offset_y), (380 + offset_x//2, 350 + offset_y), (0, 0, 255), -1)  # Red shirt
            cv2.rectangle(frame, (310 + offset_x//2, 130 + offset_y), (370 + offset_x//2, 170 + offset_y), (139, 69, 19), -1)  # Brown hair
            
            frames.append(frame)
        
        # Process frames
        all_results = []
        for i, frame in enumerate(frames):
            results = pipeline.process_frame(frame)
            all_results.append(results)
            
            # Save annotated frames periodically
            if i % 10 == 0:
                annotated = pipeline.visualize_results(frame, results)
                cv2.imwrite(f'example_2_frame_{i}.jpg', annotated)
                
                print(f"Frame {i}:")
                print(f"  Active tracks: {len(results['tracked_persons'])}")
                for track_id, tp in results['tracked_persons'].items():
                    attrs = tp.get_display_attributes()
                    if attrs:
                        print(f"    Track {track_id}: {attrs.gender.value}, {attrs.shirt_color.value} shirt, {attrs.hair_color.value} hair")
        
        # Generate summary
        summary = pipeline._generate_summary()
        print(f"\n📊 Tracking Summary:")
        print(f"   Total tracks created: {summary['component_stats']['tracker']['total_tracks_created']}")
        print(f"   Tracks with attributes: {summary['tracking_stats']['tracks_with_attributes']}")
        print(f"   Attribute coverage: {summary['tracking_stats']['attribute_coverage']:.1%}")
    
    def example_3_interactive_search(self):
        """Example 3: Interactive person search"""
        print("\n🎯 Example 3: Interactive Person Search")
        print("-" * 40)
        
        # Configure pipeline for interactive search
        config = EnhancedPipelineConfig(
            detector_model="yolov8n.pt",
            enable_interactive_query=False,  # We'll set query programmatically
            highlight_matches=True,
            save_output=False
        )
        
        pipeline = EnhancedCVPipeline(config)
        
        # Set search criteria programmatically
        search_scenarios = [
            {"gender": "female", "shirt_color": "red", "hair_color": None},
            {"gender": "male", "shirt_color": "blue", "hair_color": "brown"},
            {"gender": None, "shirt_color": "green", "hair_color": None}
        ]
        
        for scenario_idx, search_criteria in enumerate(search_scenarios):
            print(f"\n🔍 Search Scenario {scenario_idx + 1}: {search_criteria}")
            pipeline.current_query = search_criteria
            
            # Create test frame with diverse people
            frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            
            # Add people with different attributes
            people = [
                {"pos": (50, 100), "size": (80, 200), "shirt": (0, 0, 255), "hair": (255, 255, 0)},    # Red shirt, blonde hair
                {"pos": (200, 120), "size": (70, 180), "shirt": (255, 0, 0), "hair": (139, 69, 19)},   # Blue shirt, brown hair
                {"pos": (350, 80), "size": (90, 220), "shirt": (0, 255, 0), "hair": (0, 0, 0)},        # Green shirt, black hair
                {"pos": (500, 150), "size": (75, 190), "shirt": (128, 128, 128), "hair": (255, 255, 255)} # Gray shirt, white hair
            ]
            
            for person in people:
                x, y = person["pos"]
                w, h = person["size"]
                
                # Draw person body (shirt color)
                cv2.rectangle(frame, (x, y + 40), (x + w, y + h), person["shirt"], -1)
                # Draw person head (hair color) 
                cv2.rectangle(frame, (x + 10, y), (x + w - 10, y + 40), person["hair"], -1)
            
            # Process frame
            results = pipeline.process_frame(frame)
            
            # Visualize with search highlighting
            annotated = pipeline.visualize_results(frame, results)
            cv2.imwrite(f'example_3_search_{scenario_idx}.jpg', annotated)
            
            # Report matches
            matches = results['query_matches']
            print(f"   Found {len(matches)} matches")
            for i, tp in enumerate(matches):
                attrs = tp.get_display_attributes()
                if attrs:
                    print(f"     Match {i+1}: Track {tp.track.track_id} - {attrs.gender.value}, {attrs.shirt_color.value}, {attrs.hair_color.value}")
    
    def example_4_multi_camera_setup(self):
        """Example 4: Multi-camera surveillance setup"""
        print("\n🎯 Example 4: Multi-Camera Setup")
        print("-" * 40)
        
        class MultiCameraManager:
            def __init__(self, camera_configs):
                self.cameras = {}
                self.results_queue = queue.Queue()
                
                for camera_id, config in camera_configs.items():
                    pipeline = EnhancedCVPipeline(config)
                    self.cameras[camera_id] = {
                        'pipeline': pipeline,
                        'active': False,
                        'thread': None
                    }
            
            def process_camera(self, camera_id, video_source):
                """Process single camera in separate thread"""
                pipeline = self.cameras[camera_id]['pipeline']
                
                # Simulate camera feed
                for frame_idx in range(20):  # Process 20 frames per camera
                    # Create synthetic frame for each camera
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    
                    # Add camera-specific content
                    cv2.putText(frame, f"Camera {camera_id}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Add random people
                    num_people = np.random.randint(0, 4)
                    for _ in range(num_people):
                        x = np.random.randint(50, 500)
                        y = np.random.randint(50, 300)
                        cv2.rectangle(frame, (x, y), (x+60, y+150), 
                                    (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), -1)
                    
                    # Process frame
                    results = pipeline.process_frame(frame)
                    
                    # Add camera ID to results
                    results['camera_id'] = camera_id
                    results['timestamp'] = time.time()
                    
                    # Put results in queue
                    self.results_queue.put(results)
                    
                    time.sleep(0.1)  # Simulate frame rate
            
            def start_all_cameras(self):
                """Start processing all cameras"""
                for camera_id in self.cameras:
                    video_source = f"camera_{camera_id}.mp4"  # Placeholder
                    
                    thread = threading.Thread(
                        target=self.process_camera,
                        args=(camera_id, video_source)
                    )
                    thread.start()
                    
                    self.cameras[camera_id]['thread'] = thread
                    self.cameras[camera_id]['active'] = True
                
                print(f"🎥 Started {len(self.cameras)} cameras")
            
            def collect_results(self, duration=5):
                """Collect results from all cameras"""
                start_time = time.time()
                all_results = []
                
                while time.time() - start_time < duration:
                    try:
                        result = self.results_queue.get(timeout=1)
                        all_results.append(result)
                        
                        # Print summary
                        camera_id = result['camera_id']
                        num_people = len(result['tracked_persons'])
                        print(f"Camera {camera_id}: Frame {result['frame_id']}, {num_people} people")
                        
                    except queue.Empty:
                        continue
                
                return all_results
            
            def stop_all_cameras(self):
                """Stop all camera processing"""
                for camera_id, camera_info in self.cameras.items():
                    if camera_info['thread'] and camera_info['thread'].is_alive():
                        camera_info['thread'].join(timeout=1)
                    camera_info['active'] = False
                
                print("🛑 All cameras stopped")
        
        # Configure different cameras
        camera_configs = {
            "entrance": EnhancedPipelineConfig(
                detector_model="yolov8s.pt",  # Higher accuracy for entrance
                detection_confidence=0.5,
                enable_interactive_query=False,
                save_output=False
            ),
            "hallway": EnhancedPipelineConfig(
                detector_model="yolov8n.pt",  # Faster for hallway monitoring
                detection_confidence=0.6,
                enable_interactive_query=False,
                save_output=False
            ),
            "exit": EnhancedPipelineConfig(
                detector_model="yolov8s.pt",  # Higher accuracy for exit
                detection_confidence=0.5,
                enable_interactive_query=False,
                save_output=False
            )
        }
        
        # Initialize multi-camera manager
        manager = MultiCameraManager(camera_configs)
        
        # Start processing
        manager.start_all_cameras()
        
        # Collect results for 5 seconds
        print("📊 Collecting results...")
        results = manager.collect_results(duration=3)
        
        # Stop cameras
        manager.stop_all_cameras()
        
        # Analyze results
        camera_stats = {}
        for result in results:
            camera_id = result['camera_id']
            if camera_id not in camera_stats:
                camera_stats[camera_id] = {'frames': 0, 'total_people': 0}
            
            camera_stats[camera_id]['frames'] += 1
            camera_stats[camera_id]['total_people'] += len(result['tracked_persons'])
        
        print(f"\n📊 Multi-Camera Summary:")
        for camera_id, stats in camera_stats.items():
            avg_people = stats['total_people'] / stats['frames'] if stats['frames'] > 0 else 0
            print(f"   Camera {camera_id}: {stats['frames']} frames, avg {avg_people:.1f} people/frame")
    
    def example_5_performance_optimization(self):
        """Example 5: Performance optimization techniques"""
        print("\n🎯 Example 5: Performance Optimization")
        print("-" * 40)
        
        # Test different configurations
        test_configs = [
            ("Fast Config", DeploymentManager.get_real_time_config()),
            ("Accurate Config", DeploymentManager.get_high_accuracy_config()),
            ("Edge Config", DeploymentManager.get_edge_device_config())
        ]
        
        results = {}
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some people to the test frame
        for i in range(3):
            x = 50 + i * 200
            y = 100 + i * 30
            cv2.rectangle(test_frame, (x, y), (x+80, y+200), (120, 80+i*40, 60+i*30), -1)
        
        for config_name, config in test_configs:
            print(f"\n🔧 Testing {config_name}...")
            
            pipeline = EnhancedCVPipeline(config)
            
            # Benchmark processing time
            num_frames = 10
            start_time = time.time()
            
            for _ in range(num_frames):
                result = pipeline.process_frame(test_frame.copy())
            
            total_time = time.time() - start_time
            avg_fps = num_frames / total_time
            
            # Get component stats
            stats = pipeline._generate_summary()
            
            results[config_name] = {
                'avg_fps': avg_fps,
                'total_time': total_time,
                'detection_time': np.mean(pipeline.performance_stats['detection_time']),
                'tracking_time': np.mean(pipeline.performance_stats['tracking_time']),
                'attribute_time': np.mean(pipeline.performance_stats['attribute_time'])
            }
            
            print(f"   Average FPS: {avg_fps:.2f}")
            print(f"   Detection time: {results[config_name]['detection_time']*1000:.1f}ms")
            print(f"   Tracking time: {results[config_name]['tracking_time']*1000:.1f}ms")
            print(f"   Attribute time: {results[config_name]['attribute_time']*1000:.1f}ms")
        
        # Performance comparison
        print(f"\n📈 Performance Comparison:")
        print(f"{'Config':<15} {'FPS':<8} {'Detection':<12} {'Tracking':<10} {'Attributes':<12}")
        print("-" * 60)
        
        for config_name, metrics in results.items():
            print(f"{config_name:<15} {metrics['avg_fps']:<8.1f} "
                  f"{metrics['detection_time']*1000:<12.1f} "
                  f"{metrics['tracking_time']*1000:<10.1f} "
                  f"{metrics['attribute_time']*1000:<12.1f}")


class ProductionDeployment:
    """Production deployment utilities and examples"""
    
    @staticmethod
    def create_production_config(scenario="general"):
        """Create production-ready configurations"""
        
        if scenario == "real_time_monitoring":
            return EnhancedPipelineConfig(
                # Use your trained models
                detector_model="data/models/person_detector_yolov8n_best.pt",
                detection_confidence=0.6,
                
                # Optimized for real-time
                batch_processing=True,
                max_tracks_to_classify=10,
                target_fps=25.0,
                
                # Logging and monitoring
                save_output=False,
                save_annotations=True,
                save_statistics=True,
                log_level="INFO",
                
                # Output directory with timestamp
                output_directory=f"production_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        elif scenario == "forensic_analysis":
            return EnhancedPipelineConfig(
                # High accuracy model
                detector_model="data/models/person_detector_yolov8m_best.pt",
                detection_confidence=0.3,  # Lower threshold for forensics
                
                # Detailed tracking
                tracking_iou_threshold=0.2,
                max_disappeared=100,
                min_hits=3,
                use_kalman=True,
                
                # Full processing
                batch_processing=True,
                max_tracks_to_classify=20,
                target_fps=10.0,  # Lower FPS for accuracy
                
                # Complete logging
                save_output=True,
                save_annotations=True,
                save_statistics=True,
                show_confidence_scores=True,
                show_trails=True,
                
                output_directory=f"forensic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        elif scenario == "edge_deployment":
            return EnhancedPipelineConfig(
                # Lightweight model
                detector_model="data/models/person_detector_yolov8n_best.pt",
                detection_confidence=0.7,
                
                # Minimal processing
                tracking_iou_threshold=0.4,
                max_disappeared=20,
                min_hits=2,
                use_kalman=False,
                
                # Reduced load
                batch_processing=False,
                max_tracks_to_classify=5,
                target_fps=15.0,
                
                # Minimal output
                save_output=False,
                save_annotations=False,
                show_attributes=False,
                
                device="cpu"  # For edge devices
            )
        
        else:  # general
            return EnhancedPipelineConfig(
                detector_model="data/models/person_detector_yolov8s_best.pt",
                detection_confidence=0.6,
                attribute_model="data/models/attribute_classifier.pth",  # If available
                enable_interactive_query=True,
                save_output=True,
                save_annotations=True
            )
    
    @staticmethod
    def setup_monitoring(pipeline, alert_thresholds=None):
        """Setup production monitoring and alerting"""
        
        if alert_thresholds is None:
            alert_thresholds = {
                'min_fps': 10.0,
                'max_people_per_frame': 20,
                'max_processing_time': 0.5  # seconds
            }
        
        class PipelineMonitor:
            def __init__(self, thresholds):
                self.thresholds = thresholds
                self.alerts = []
                
            def check_performance(self, results):
                """Check performance metrics and generate alerts"""
                current_time = datetime.now()
                
                # Check FPS
                fps = results['performance']['current_fps']
                if fps < self.thresholds['min_fps']:
                    self.alerts.append({
                        'timestamp': current_time,
                        'type': 'LOW_FPS',
                        'value': fps,
                        'threshold': self.thresholds['min_fps']
                    })
                
                # Check processing time
                proc_time = results['processing_times']['total']
                if proc_time > self.thresholds['max_processing_time']:
                    self.alerts.append({
                        'timestamp': current_time,
                        'type': 'HIGH_PROCESSING_TIME',
                        'value': proc_time,
                        'threshold': self.thresholds['max_processing_time']
                    })
                
                # Check people count
                people_count = len(results['tracked_persons'])
                if people_count > self.thresholds['max_people_per_frame']:
                    self.alerts.append({
                        'timestamp': current_time,
                        'type': 'HIGH_PEOPLE_COUNT',
                        'value': people_count,
                        'threshold': self.thresholds['max_people_per_frame']
                    })
            
            def get_recent_alerts(self, minutes=5):
                """Get alerts from last N minutes"""
                cutoff_time = datetime.now() - timedelta(minutes=minutes)
                return [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
        
        return PipelineMonitor(alert_thresholds)
    
    @staticmethod
    def create_deployment_script():
        """Create a production deployment script"""
        
        deployment_script = '''#!/usr/bin/env python3
"""
Production CCTV Pipeline Deployment Script
"""

import argparse
import signal
import sys
from pathlib import Path

# Import your modules
from enhanced_pipeline import EnhancedCVPipeline
from usage_examples import ProductionDeployment

def signal_handler(sig, frame):
    print('\\n🛑 Stopping pipeline...')
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Production CCTV Pipeline')
    parser.add_argument('--video', required=True, help='Video source')
    parser.add_argument('--scenario', default='general', 
                       choices=['real_time_monitoring', 'forensic_analysis', 'edge_deployment', 'general'])
    parser.add_argument('--output-dir', default='production_output')
    parser.add_argument('--max-duration', type=int, help='Max duration in seconds')
    
    args = parser.parse_args()
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create configuration
    config = ProductionDeployment.create_production_config(args.scenario)
    config.output_directory = args.output_dir
    
    # Initialize pipeline
    print(f"🚀 Starting {args.scenario} pipeline...")
    pipeline = EnhancedCVPipeline(config)
    
    # Setup monitoring
    monitor = ProductionDeployment.setup_monitoring(pipeline)
    
    # Run pipeline
    try:
        summary = pipeline.run_video(
            video_source=args.video,
            max_frames=args.max_duration * 30 if args.max_duration else None
        )
        
        print("✅ Pipeline completed successfully")
        pipeline.print_summary(summary)
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
        
        with open('deploy_pipeline.py', 'w') as f:
            f.write(deployment_script)
        
        print("📄 Deployment script created: deploy_pipeline.py")
        print("Usage: python deploy_pipeline.py --video camera_feed.mp4 --scenario real_time_monitoring")


def main():
    """Main function to run all examples"""
    print("🎯 CCTV Pipeline Usage Examples")
    print("=" * 50)
    
    examples = ScenarioExamples()
    
    # Run all examples
    try:
        examples.example_1_basic_detection()
        examples.example_2_tracking_with_attributes()
        examples.example_3_interactive_search()
        examples.example_4_multi_camera_setup()
        examples.example_5_performance_optimization()
        
        print("\n🏭 Production Deployment")
        print("-" * 40)
        
        # Create production configurations
        prod_configs = {
            "Real-time Monitoring": ProductionDeployment.create_production_config("real_time_monitoring"),
            "Forensic Analysis": ProductionDeployment.create_production_config("forensic_analysis"),
            "Edge Deployment": ProductionDeployment.create_production_config("edge_deployment")
        }
        
        print("📋 Available production configurations:")
        for name, config in prod_configs.items():
            print(f"   {name}: FPS target {config.target_fps}, confidence {config.detection_confidence}")
        
        # Create deployment script
        ProductionDeployment.create_deployment_script()
        
        print("\n🎉 All examples completed successfully!")
        print("\n📚 Next Steps:")
        print("1. Replace 'yolov8n.pt' with your trained models")
        print("2. Test with your actual CCTV footage")
        print("3. Adjust confidence thresholds based on your environment")
        print("4. Set up production monitoring for your use case")
        print("5. Consider multi-camera deployment for complete coverage")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
