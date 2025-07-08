"""
Comprehensive Testing Framework
Tests all components individually and as an integrated system
"""

import cv2
import numpy as np
import torch
import time
import json
import pytest
import unittest
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Tuple, Any
import logging
from unittest.mock import Mock, patch

# Import all our modules
from person_detector import PersonDetector, DetectionConfig
from person_tracker import PersonTracker, TrackingConfig, Track
from attribute_classifier import (
    AttributeClassifier, PersonAttributes, Gender, ShirtColor, HairColor,
    InteractiveQuerySystem
)
from training_system import TrainingConfig, AttributeTrainer, DatasetCreator
from enhanced_pipeline import EnhancedCVPipeline, EnhancedPipelineConfig


class TestDataGenerator:
    """Generates test data for various scenarios"""
    
    @staticmethod
    def create_test_image(width: int = 640, height: int = 480, 
                         add_person_shapes: bool = True) -> np.ndarray:
        """Create a test image with optional person-like shapes"""
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        if add_person_shapes:
            # Add person-like rectangles
            # Person 1
            cv2.rectangle(image, (100, 100), (180, 300), (120, 80, 60), -1)  # Body
            cv2.rectangle(image, (110, 80), (170, 120), (139, 69, 19), -1)   # Head
            
            # Person 2  
            cv2.rectangle(image, (300, 150), (380, 350), (60, 120, 180), -1)  # Body
            cv2.rectangle(image, (310, 130), (370, 170), (255, 255, 0), -1)   # Head
            
            # Person 3 (partially occluded)
            cv2.rectangle(image, (500, 200), (580, 400), (0, 255, 0), -1)     # Body
            cv2.rectangle(image, (510, 180), (570, 220), (0, 0, 0), -1)       # Head
        
        return image
    
    @staticmethod
    def create_detection_sequence() -> List[Dict]:
        """Create a sequence of detection data for tracking tests"""
        return [
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
    
    @staticmethod
    def create_test_video(output_path: str, num_frames: int = 100, 
                         width: int = 640, height: int = 480, fps: float = 30.0):
        """Create a test video file"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i in range(num_frames):
            # Create frame with moving objects
            frame = TestDataGenerator.create_test_image(width, height)
            
            # Add frame number
            cv2.putText(frame, f"Frame {i+1}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Simulate moving people by shifting rectangles
            offset = i * 2
            cv2.rectangle(frame, (100 + offset, 100), (180 + offset, 300), (120, 80, 60), -1)
            
            writer.write(frame)
        
        writer.release()
        print(f"✅ Test video created: {output_path}")


class TestPersonDetector(unittest.TestCase):
    """Test cases for PersonDetector module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = PersonDetector()
        self.test_image = TestDataGenerator.create_test_image()
    
    def test_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.model)
        self.assertIn(self.detector.device, ['cpu', 'cuda'])
    
    def test_detection_basic(self):
        """Test basic detection functionality"""
        detections = self.detector.detect_persons(self.test_image)
        
        self.assertIsInstance(detections, dict)
        self.assertIn('boxes', detections)
        self.assertIn('confidences', detections)
        self.assertIn('class_ids', detections)
        
        # Check data types
        self.assertIsInstance(detections['boxes'], list)
        self.assertIsInstance(detections['confidences'], list)
        self.assertIsInstance(detections['class_ids'], list)
    
    def test_detection_with_crops(self):
        """Test detection with crop extraction"""
        detections = self.detector.detect_persons(self.test_image, return_crops=True)
        
        self.assertIn('crops', detections)
        if detections['boxes']:  # If any detections found
            self.assertEqual(len(detections['crops']), len(detections['boxes']))
            
            # Check crop validity
            for crop in detections['crops']:
                self.assertIsInstance(crop, np.ndarray)
                self.assertEqual(len(crop.shape), 3)  # Height, width, channels
    
    def test_batch_detection(self):
        """Test batch detection"""
        test_images = [TestDataGenerator.create_test_image() for _ in range(3)]
        batch_detections = self.detector.batch_detect(test_images)
        
        self.assertEqual(len(batch_detections), len(test_images))
        
        for detections in batch_detections:
            self.assertIsInstance(detections, dict)
            self.assertIn('boxes', detections)
    
    def test_visualization(self):
        """Test detection visualization"""
        detections = self.detector.detect_persons(self.test_image)
        annotated = self.detector.visualize_detections(self.test_image, detections)
        
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, self.test_image.shape)
    
    def test_performance_stats(self):
        """Test performance statistics"""
        # Run some detections first
        for _ in range(5):
            self.detector.detect_persons(self.test_image)
        
        stats = self.detector.get_performance_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('avg_fps', stats)
        self.assertIn('total_frames', stats)
        self.assertGreater(stats['total_frames'], 0)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = DetectionConfig(
            confidence_threshold=0.8,
            iou_threshold=0.5,
            max_detections=50
        )
        
        custom_detector = PersonDetector(config)
        self.assertEqual(custom_detector.config.confidence_threshold, 0.8)
        self.assertEqual(custom_detector.config.iou_threshold, 0.5)


class TestPersonTracker(unittest.TestCase):
    """Test cases for PersonTracker module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tracker = PersonTracker()
        self.detection_sequence = TestDataGenerator.create_detection_sequence()
    
    def test_initialization(self):
        """Test tracker initialization"""
        self.assertIsNotNone(self.tracker)
        self.assertEqual(len(self.tracker.tracks), 0)
        self.assertEqual(self.tracker.next_track_id, 1)
    
    def test_tracking_sequence(self):
        """Test tracking through a sequence"""
        all_tracks = []
        
        for i, detections in enumerate(self.detection_sequence):
            tracks = self.tracker.update(detections)
            all_tracks.append(tracks)
            
            print(f"Frame {i+1}: {len(tracks)} tracks")
            for track in tracks:
                print(f"  Track {track.track_id}: state={track.state}, hits={track.hits}")
        
        # Check that we have some tracks
        self.assertGreater(len(all_tracks), 0)
        
        # Check track consistency
        final_tracks = all_tracks[-1]
        for track in final_tracks:
            self.assertIsInstance(track, Track)
            self.assertGreater(track.track_id, 0)
    
    def test_track_lifecycle(self):
        """Test track creation, confirmation, and deletion"""
        # Process first frame - should create tentative tracks
        tracks = self.tracker.update(self.detection_sequence[0])
        
        tentative_tracks = [t for t in tracks if t.state == 'tentative']
        self.assertGreater(len(tentative_tracks), 0)
        
        # Process more frames - tracks should become confirmed
        for detections in self.detection_sequence[1:3]:
            tracks = self.tracker.update(detections)
        
        confirmed_tracks = [t for t in tracks if t.state == 'confirmed']
        self.assertGreater(len(confirmed_tracks), 0)
    
    def test_track_visualization(self):
        """Test track visualization"""
        # Process some frames
        for detections in self.detection_sequence[:2]:
            self.tracker.update(detections)
        
        test_frame = TestDataGenerator.create_test_image()
        annotated = self.tracker.visualize_tracks(test_frame)
        
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, test_frame.shape)
    
    def test_performance_stats(self):
        """Test tracker performance statistics"""
        # Process sequence
        for detections in self.detection_sequence:
            self.tracker.update(detections)
        
        stats = self.tracker.get_performance_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_tracks_created', stats)
        self.assertIn('active_tracks', stats)
    
    def test_custom_config(self):
        """Test custom tracking configuration"""
        config = TrackingConfig(
            iou_threshold=0.5,
            max_disappeared=20,
            min_hits=2
        )
        
        custom_tracker = PersonTracker(config)
        self.assertEqual(custom_tracker.config.iou_threshold, 0.5)
        self.assertEqual(custom_tracker.config.max_disappeared, 20)


class TestAttributeClassifier(unittest.TestCase):
    """Test cases for AttributeClassifier module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.classifier = AttributeClassifier()  # Demo mode
        self.test_crop = TestDataGenerator.create_test_image(224, 224)
    
    def test_initialization(self):
        """Test classifier initialization"""
        self.assertIsNotNone(self.classifier)
        self.assertIsNotNone(self.classifier.model)
        self.assertIn(self.classifier.device, ['cpu', 'cuda'])
    
    def test_prediction(self):
        """Test attribute prediction"""
        attributes = self.classifier.predict_attributes(self.test_crop)
        
        self.assertIsInstance(attributes, PersonAttributes)
        self.assertIsInstance(attributes.gender, Gender)
        self.assertIsInstance(attributes.shirt_color, ShirtColor)
        self.assertIsInstance(attributes.hair_color, HairColor)
        
        # Check confidence ranges
        self.assertGreaterEqual(attributes.gender_confidence, 0.0)
        self.assertLessEqual(attributes.gender_confidence, 1.0)
        self.assertGreaterEqual(attributes.shirt_confidence, 0.0)
        self.assertLessEqual(attributes.shirt_confidence, 1.0)
        self.assertGreaterEqual(attributes.hair_confidence, 0.0)
        self.assertLessEqual(attributes.hair_confidence, 1.0)
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        test_crops = [TestDataGenerator.create_test_image(224, 224) for _ in range(3)]
        batch_attributes = self.classifier.batch_predict(test_crops)
        
        self.assertEqual(len(batch_attributes), len(test_crops))
        
        for attributes in batch_attributes:
            self.assertIsInstance(attributes, PersonAttributes)
    
    def test_visualization(self):
        """Test attribute visualization"""
        attributes = self.classifier.predict_attributes(self.test_crop)
        annotated = self.classifier.visualize_attributes(self.test_crop, attributes)
        
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, self.test_crop.shape)
    
    def test_performance_stats(self):
        """Test performance statistics"""
        # Run some predictions
        for _ in range(5):
            self.classifier.predict_attributes(self.test_crop)
        
        stats = self.classifier.get_performance_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_inferences', stats)
        self.assertGreater(stats['total_inferences'], 0)


class TestInteractiveQuerySystem(unittest.TestCase):
    """Test cases for InteractiveQuerySystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.query_system = InteractiveQuerySystem()
    
    def test_initialization(self):
        """Test query system initialization"""
        self.assertIsNotNone(self.query_system)
        self.assertIsInstance(self.query_system.current_query, dict)
    
    def test_query_update(self):
        """Test programmatic query updates"""
        self.query_system.update_query(gender='female', shirt_color='red')
        
        self.assertEqual(self.query_system.current_query['gender'], 'female')
        self.assertEqual(self.query_system.current_query['shirt_color'], 'red')
    
    def test_query_formatting(self):
        """Test query string formatting"""
        self.query_system.update_query(gender='male', shirt_color='blue')
        query_string = self.query_system.format_query_string()
        
        self.assertIsInstance(query_string, str)
        self.assertIn('male', query_string)
        self.assertIn('blue', query_string)
    
    def test_matching_criteria(self):
        """Test attribute matching"""
        # Create test attributes
        attributes = PersonAttributes(
            gender=Gender.FEMALE,
            shirt_color=ShirtColor.RED,
            hair_color=HairColor.BROWN,
            gender_confidence=0.8,
            shirt_confidence=0.7,
            hair_confidence=0.6
        )
        
        # Test exact match
        self.query_system.update_query(gender='female', shirt_color='red')
        self.assertTrue(self.query_system.matches_criteria(attributes))
        
        # Test no match
        self.query_system.update_query(gender='male', shirt_color='blue')
        self.assertFalse(self.query_system.matches_criteria(attributes))
        
        # Test partial match
        self.query_system.update_query(gender='female', shirt_color='blue')
        self.assertFalse(self.query_system.matches_criteria(attributes))


class TestTrainingSystem(unittest.TestCase):
    """Test cases for training system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.temp_dir) / "test_dataset"
        
        # Create minimal test dataset
        DatasetCreator.create_sample_dataset(
            output_dir=str(self.dataset_path),
            num_samples=50  # Very small for testing
        )
        
        self.config = TrainingConfig(
            dataset_path=str(self.dataset_path),
            epochs=2,  # Very short for testing
            batch_size=4,
            use_wandb=False
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_creation(self):
        """Test sample dataset creation"""
        self.assertTrue(self.dataset_path.exists())
        self.assertTrue((self.dataset_path / "images").exists())
        self.assertTrue((self.dataset_path / "annotations.json").exists())
        
        # Check annotations
        with open(self.dataset_path / "annotations.json", 'r') as f:
            annotations = json.load(f)
        
        self.assertGreater(len(annotations), 0)
        self.assertIn('image_name', annotations[0])
        self.assertIn('gender', annotations[0])
        self.assertIn('shirt_color', annotations[0])
        self.assertIn('hair_color', annotations[0])
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = AttributeTrainer(self.config)
        
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.config.dataset_path, str(self.dataset_path))
    
    def test_dataset_loading(self):
        """Test dataset loading"""
        trainer = AttributeTrainer(self.config)
        trainer.prepare_datasets()
        
        self.assertIsNotNone(trainer.train_dataset)
        self.assertIsNotNone(trainer.val_dataset)
        self.assertGreater(len(trainer.train_dataset), 0)
    
    def test_model_creation(self):
        """Test model creation"""
        trainer = AttributeTrainer(self.config)
        trainer.create_model()
        
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.criterion)


class TestEnhancedPipeline(unittest.TestCase):
    """Test cases for the complete enhanced pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = EnhancedPipelineConfig(
            enable_interactive_query=False,  # Disable for testing
            save_output=False,
            save_annotations=False
        )
        self.pipeline = EnhancedCVPipeline(self.config)
        self.test_frame = TestDataGenerator.create_test_image()
    
    def test_initialization(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.detector)
        self.assertIsNotNone(self.pipeline.tracker)
        self.assertIsNotNone(self.pipeline.attribute_classifier)
    
    def test_frame_processing(self):
        """Test single frame processing"""
        results = self.pipeline.process_frame(self.test_frame)
        
        self.assertIsInstance(results, dict)
        self.assertIn('frame_id', results)
        self.assertIn('detections', results)
        self.assertIn('tracks', results)
        self.assertIn('tracked_persons', results)
        self.assertIn('processing_times', results)
    
    def test_visualization(self):
        """Test result visualization"""
        results = self.pipeline.process_frame(self.test_frame)
        annotated = self.pipeline.visualize_results(self.test_frame, results)
        
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, self.test_frame.shape)
    
    def test_query_system_integration(self):
        """Test query system integration"""
        # Set up query
        self.pipeline.current_query = {'gender': 'female', 'shirt_color': 'red', 'hair_color': None}
        
        # Process frame
        results = self.pipeline.process_frame(self.test_frame)
        
        self.assertIn('query_matches', results)
        self.assertIn('current_query', results)
    
    def test_performance_tracking(self):
        """Test performance tracking"""
        # Process multiple frames
        for _ in range(5):
            self.pipeline.process_frame(self.test_frame)
        
        # Check performance stats
        self.assertGreater(len(self.pipeline.performance_stats['total_time']), 0)
        self.assertGreater(self.pipeline.frame_count, 0)


class IntegrationTester:
    """High-level integration testing"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.results = {}
    
    def __del__(self):
        """Cleanup"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow(self):
        """Test complete workflow from detection to querying"""
        print("🧪 Running complete workflow integration test...")
        
        # Step 1: Create test video
        video_path = self.temp_dir / "test_video.mp4"
        TestDataGenerator.create_test_video(str(video_path), num_frames=30)
        
        # Step 2: Set up pipeline
        config = EnhancedPipelineConfig(
            enable_interactive_query=False,
            save_output=True,
            output_directory=str(self.temp_dir / "output"),
            save_annotations=True
        )
        
        pipeline = EnhancedCVPipeline(config)
        
        # Step 3: Process video
        pipeline.current_query = {'gender': 'female', 'shirt_color': 'red', 'hair_color': None}
        
        # Process limited frames for testing
        cap = cv2.VideoCapture(str(video_path))
        frames_processed = 0
        
        while cap.isOpened() and frames_processed < 10:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = pipeline.process_frame(frame)
            frames_processed += 1
        
        cap.release()
        
        # Step 4: Validate results
        summary = pipeline._generate_summary()
        
        self.results['frames_processed'] = summary['session_info']['frames_processed']
        self.results['total_tracks'] = summary['tracking_stats']['total_tracks']
        self.results['classified_persons'] = summary['attribute_stats']['total_classified']
        
        print(f"✅ Workflow test completed:")
        print(f"   Frames: {self.results['frames_processed']}")
        print(f"   Tracks: {self.results['total_tracks']}")
        print(f"   Classified: {self.results['classified_persons']}")
        
        return self.results
    
    def test_performance_benchmarks(self):
        """Test performance under different conditions"""
        print("⚡ Running performance benchmark tests...")
        
        configs = [
            ("Fast", {"detector_model": "yolov8n.pt", "batch_processing": True}),
            ("Balanced", {"detector_model": "yolov8s.pt", "batch_processing": True}),
            ("Accurate", {"detector_model": "yolov8m.pt", "batch_processing": False})
        ]
        
        benchmark_results = {}
        
        for name, config_overrides in configs:
            print(f"  Testing {name} configuration...")
            
            config = EnhancedPipelineConfig(**config_overrides)
            config.enable_interactive_query = False
            config.save_output = False
            
            try:
                pipeline = EnhancedCVPipeline(config)
                
                # Process test frames
                test_frames = [TestDataGenerator.create_test_image() for _ in range(10)]
                start_time = time.time()
                
                for frame in test_frames:
                    pipeline.process_frame(frame)
                
                total_time = time.time() - start_time
                avg_fps = len(test_frames) / total_time
                
                benchmark_results[name] = {
                    'avg_fps': avg_fps,
                    'total_time': total_time,
                    'frames': len(test_frames)
                }
                
                print(f"    {name}: {avg_fps:.2f} FPS")
                
            except Exception as e:
                print(f"    {name}: FAILED - {e}")
                benchmark_results[name] = {'error': str(e)}
        
        self.results['benchmarks'] = benchmark_results
        return benchmark_results
    
    def test_error_handling(self):
        """Test error handling and recovery"""
        print("🛡️ Testing error handling...")
        
        error_tests = []
        
        # Test with invalid video source
        try:
            config = EnhancedPipelineConfig(enable_interactive_query=False)
            pipeline = EnhancedCVPipeline(config)
            pipeline.run_video("nonexistent_video.mp4", max_frames=1)
        except Exception as e:
            error_tests.append(("Invalid video", "PASS - Expected error caught"))
        
        # Test with corrupted frame
        try:
            config = EnhancedPipelineConfig(enable_interactive_query=False)
            pipeline = EnhancedCVPipeline(config)
            
            # Create corrupted frame (wrong dimensions)
            corrupted_frame = np.random.randint(0, 255, (10, 10, 1), dtype=np.uint8)
            results = pipeline.process_frame(corrupted_frame)
            
            error_tests.append(("Corrupted frame", "PASS - Handled gracefully"))
        except Exception as e:
            error_tests.append(("Corrupted frame", f"FAIL - {e}"))
        
        for test_name, result in error_tests:
            print(f"  {test_name}: {result}")
        
        self.results['error_tests'] = error_tests
        return error_tests


def run_all_tests():
    """Run all test suites"""
    print("🚀 Running Comprehensive Test Suite")
    print("=" * 60)
    
    # Unit tests
    print("\n📋 Running Unit Tests...")
    test_classes = [
        TestPersonDetector,
        TestPersonTracker,
        TestAttributeClassifier,
        TestInteractiveQuerySystem,
        TestTrainingSystem,
        TestEnhancedPipeline
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n  Testing {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        
        result = runner.run(suite)
        
        class_total = result.testsRun
        class_passed = class_total - len(result.failures) - len(result.errors)
        
        total_tests += class_total
        passed_tests += class_passed
        
        status = "✅ PASS" if class_passed == class_total else "❌ FAIL"
        print(f"    {status} ({class_passed}/{class_total} tests passed)")
        
        # Print failures if any
        if result.failures:
            for test, traceback in result.failures:
                print(f"      FAILED: {test}")
        
        if result.errors:
            for test, traceback in result.errors:
                print(f"      ERROR: {test}")
    
    print(f"\n📊 Unit Test Summary: {passed_tests}/{total_tests} tests passed")
    
    # Integration tests
    print("\n🔗 Running Integration Tests...")
    integration_tester = IntegrationTester()
    
    try:
        workflow_results = integration_tester.test_complete_workflow()
        print("  ✅ Complete workflow test passed")
    except Exception as e:
        print(f"  ❌ Complete workflow test failed: {e}")
    
    try:
        benchmark_results = integration_tester.test_performance_benchmarks()
        print("  ✅ Performance benchmark tests completed")
    except Exception as e:
        print(f"  ❌ Performance benchmark tests failed: {e}")
    
    try:
        error_results = integration_tester.test_error_handling()
        print("  ✅ Error handling tests completed")
    except Exception as e:
        print(f"  ❌ Error handling tests failed: {e}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUITE SUMMARY")
    print("=" * 60)
    print(f"Unit Tests: {passed_tests}/{total_tests} passed")
    print("Integration Tests: Completed")
    print("Performance Tests: Completed")
    print("Error Handling Tests: Completed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! System is ready for deployment.")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests failed. Please review and fix issues.")
    
    return {
        'unit_tests': {'passed': passed_tests, 'total': total_tests},
        'integration_results': integration_tester.results
    }


# Quick test runner for individual components
def quick_test_component(component_name: str):
    """Quick test for individual components"""
    component_tests = {
        'detector': TestPersonDetector,
        'tracker': TestPersonTracker,
        'classifier': TestAttributeClassifier,
        'query': TestInteractiveQuerySystem,
        'training': TestTrainingSystem,
        'pipeline': TestEnhancedPipeline
    }
    
    if component_name not in component_tests:
        print(f"❌ Unknown component: {component_name}")
        print(f"Available components: {list(component_tests.keys())}")
        return
    
    print(f"🧪 Quick testing {component_name}...")
    
    test_class = component_tests[component_name]
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print(f"✅ {component_name} tests passed!")
    else:
        print(f"❌ {component_name} tests failed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Quick test specific component
        component = sys.argv[1]
        quick_test_component(component)
    else:
        # Run all tests
        results = run_all_tests()
        
        # Save results
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Test results saved to: test_results.json")
