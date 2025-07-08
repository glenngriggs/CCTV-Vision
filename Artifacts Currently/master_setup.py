#!/usr/bin/env python3
"""
Master Setup and Integration Script for Enhanced CCTV Pipeline
Handles complete setup, configuration, and provides unified interface
"""

import os
import sys
import subprocess
import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import platform
import urllib.request
import zipfile
from datetime import datetime

# Version and metadata
__version__ = "1.0.0"
__author__ = "Enhanced CCTV Pipeline Team"
__description__ = "Complete CCTV computer vision pipeline with person detection, tracking, and attribute classification"

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

class PipelineSetup:
    """Main setup and management class"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.setup_logging()
        
        # Default directories
        self.directories = {
            'data': self.project_root / 'data',
            'models': self.project_root / 'data' / 'models',
            'videos': self.project_root / 'data' / 'videos',
            'output': self.project_root / 'data' / 'output',
            'configs': self.project_root / 'configs',
            'logs': self.project_root / 'logs',
            'tests': self.project_root / 'tests'
        }
        
        # System requirements
        self.requirements = {
            'python_version': (3, 9),
            'disk_space_gb': 5,
            'ram_gb': 4
        }
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format=f'{Colors.CYAN}%(asctime)s{Colors.RESET} - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def print_banner(self):
        """Print welcome banner"""
        banner = f"""
{Colors.CYAN}{'='*80}{Colors.RESET}
{Colors.BOLD}{Colors.GREEN}    🎯 Enhanced CCTV Computer Vision Pipeline v{__version__}{Colors.RESET}
{Colors.CYAN}{'='*80}{Colors.RESET}

{Colors.BOLD}Features:{Colors.RESET}
  🎯 Person Detection with custom YOLOv8/YOLOv11 models
  🎬 Multi-object tracking with Kalman filtering  
  🏷️ Attribute classification (gender, shirt, hair color)
  🔍 Interactive search and querying
  ⚡ Real-time processing optimization
  🏭 Production-ready deployment

{Colors.BOLD}Author:{Colors.RESET} {__author__}
{Colors.BOLD}Description:{Colors.RESET} {__description__}
{Colors.CYAN}{'='*80}{Colors.RESET}
        """
        print(banner)
    
    def check_system_requirements(self) -> bool:
        """Check if system meets requirements"""
        self.logger.info("🔍 Checking system requirements...")
        
        issues = []
        
        # Check Python version
        py_version = sys.version_info[:2]
        if py_version < self.requirements['python_version']:
            issues.append(f"Python {self.requirements['python_version'][0]}.{self.requirements['python_version'][1]}+ required, got {py_version[0]}.{py_version[1]}")
        
        # Check disk space
        try:
            disk_usage = shutil.disk_usage(self.project_root)
            free_gb = disk_usage.free / (1024**3)
            if free_gb < self.requirements['disk_space_gb']:
                issues.append(f"Insufficient disk space: {free_gb:.1f}GB free, {self.requirements['disk_space_gb']}GB required")
        except:
            self.logger.warning("Could not check disk space")
        
        # Check available RAM
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
            if ram_gb < self.requirements['ram_gb']:
                issues.append(f"Insufficient RAM: {ram_gb:.1f}GB available, {self.requirements['ram_gb']}GB recommended")
        except ImportError:
            self.logger.warning("Could not check RAM (psutil not installed)")
        
        # Check GPU availability
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"✅ GPU detected: {gpu_name}")
            else:
                self.logger.warning("⚠️ No GPU detected - will use CPU (slower performance)")
        except ImportError:
            self.logger.warning("PyTorch not installed - cannot check GPU")
        
        if issues:
            self.logger.error("❌ System requirement issues:")
            for issue in issues:
                self.logger.error(f"   - {issue}")
            return False
        
        self.logger.info("✅ System requirements check passed")
        return True
    
    def create_project_structure(self):
        """Create project directory structure"""
        self.logger.info("📁 Creating project directory structure...")
        
        for name, path in self.directories.items():
            path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created: {path}")
        
        # Create subdirectories
        subdirs = [
            self.directories['data'] / 'attribute_dataset' / 'images',
            self.directories['data'] / 'attribute_dataset' / 'annotations',
            self.directories['models'] / 'detection',
            self.directories['models'] / 'attributes',
            self.directories['output'] / 'videos',
            self.directories['output'] / 'annotations',
            self.directories['output'] / 'statistics'
        ]
        
        for subdir in subdirs:
            subdir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ Project structure created")
    
    def check_dependencies(self) -> Tuple[List[str], List[str]]:
        """Check which dependencies are installed"""
        required_packages = [
            'torch', 'torchvision', 'ultralytics', 'opencv-python',
            'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn',
            'pandas', 'tqdm', 'albumentations', 'optuna', 'wandb'
        ]
        
        installed = []
        missing = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                installed.append(package)
            except ImportError:
                missing.append(package)
        
        return installed, missing
    
    def install_dependencies(self, missing_packages: List[str]) -> bool:
        """Install missing dependencies"""
        if not missing_packages:
            self.logger.info("✅ All dependencies already installed")
            return True
        
        self.logger.info(f"📦 Installing {len(missing_packages)} missing packages...")
        
        # Group packages for efficient installation
        torch_packages = [p for p in missing_packages if p.startswith(('torch', 'torchvision'))]
        other_packages = [p for p in missing_packages if not p.startswith(('torch', 'torchvision'))]
        
        try:
            # Install PyTorch with CUDA if available
            if torch_packages:
                self.logger.info("Installing PyTorch with CUDA support...")
                cmd = [
                    sys.executable, '-m', 'pip', 'install', 
                    'torch', 'torchvision', 'torchaudio',
                    '--index-url', 'https://download.pytorch.org/whl/cu118'
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            
            # Install other packages
            if other_packages:
                self.logger.info(f"Installing remaining packages: {', '.join(other_packages)}")
                cmd = [sys.executable, '-m', 'pip', 'install'] + other_packages
                subprocess.run(cmd, check=True, capture_output=True)
            
            self.logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def download_default_models(self):
        """Download default pre-trained models"""
        self.logger.info("📥 Downloading default models...")
        
        models_to_download = [
            {
                'name': 'yolov8n.pt',
                'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
                'description': 'YOLOv8 Nano - fastest detection model'
            },
            {
                'name': 'yolov8s.pt', 
                'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
                'description': 'YOLOv8 Small - balanced speed/accuracy'
            }
        ]
        
        for model_info in models_to_download:
            model_path = self.directories['models'] / 'detection' / model_info['name']
            
            if model_path.exists():
                self.logger.info(f"⏭️ {model_info['name']} already exists")
                continue
            
            try:
                self.logger.info(f"Downloading {model_info['name']}...")
                urllib.request.urlretrieve(model_info['url'], model_path)
                self.logger.info(f"✅ Downloaded {model_info['name']}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to download {model_info['name']}: {e}")
    
    def create_default_configs(self):
        """Create default configuration files"""
        self.logger.info("⚙️ Creating default configuration files...")
        
        # Detection config
        detection_config = {
            "model_path": "data/models/detection/yolov8s.pt",
            "confidence_threshold": 0.6,
            "iou_threshold": 0.7,
            "max_detections": 100,
            "device": "auto",
            "half_precision": False
        }
        
        # Tracking config
        tracking_config = {
            "iou_threshold": 0.3,
            "max_disappeared": 30,
            "min_hits": 3,
            "use_kalman": True,
            "confidence_threshold": 0.5
        }
        
        # Pipeline config
        pipeline_config = {
            "detection": detection_config,
            "tracking": tracking_config,
            "attributes": {
                "model_path": None,
                "confidence_threshold": 0.5,
                "batch_processing": True,
                "max_tracks_to_classify": 10
            },
            "performance": {
                "target_fps": 30.0,
                "batch_processing": True,
                "device": "auto"
            },
            "visualization": {
                "show_detection_boxes": True,
                "show_track_ids": True,
                "show_attributes": True,
                "show_confidence_scores": False,
                "highlight_matches": True,
                "show_trails": True
            },
            "output": {
                "save_output": False,
                "save_annotations": True,
                "save_statistics": True,
                "output_directory": "data/output"
            }
        }
        
        # Save configs
        configs = {
            'detection_config.json': detection_config,
            'tracking_config.json': tracking_config,
            'pipeline_config.json': pipeline_config
        }
        
        for filename, config in configs.items():
            config_path = self.directories['configs'] / filename
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            self.logger.debug(f"Created: {config_path}")
        
        self.logger.info("✅ Default configurations created")
    
    def run_tests(self) -> bool:
        """Run basic functionality tests"""
        self.logger.info("🧪 Running basic functionality tests...")
        
        try:
            # Test imports
            self.logger.info("Testing module imports...")
            
            test_imports = [
                'person_detector',
                'person_tracker', 
                'attribute_classifier',
                'enhanced_pipeline'
            ]
            
            sys.path.insert(0, str(self.project_root))
            
            for module in test_imports:
                try:
                    __import__(module)
                    self.logger.debug(f"✅ {module} imported successfully")
                except ImportError as e:
                    self.logger.error(f"❌ Failed to import {module}: {e}")
                    return False
            
            # Test basic functionality
            self.logger.info("Testing basic detection functionality...")
            
            from person_detector import PersonDetector
            import numpy as np
            
            # Create test detector
            detector = PersonDetector()
            
            # Test with synthetic image
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detections = detector.detect_persons(test_image)
            
            self.logger.info(f"✅ Detection test passed - found {len(detections['boxes'])} objects")
            
            self.logger.info("✅ All basic tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Tests failed: {e}")
            return False
    
    def setup_development_environment(self):
        """Setup development tools and pre-commit hooks"""
        self.logger.info("🛠️ Setting up development environment...")
        
        dev_packages = ['pytest', 'black', 'flake8', 'mypy', 'pre-commit']
        
        try:
            cmd = [sys.executable, '-m', 'pip', 'install'] + dev_packages
            subprocess.run(cmd, check=True, capture_output=True)
            self.logger.info("✅ Development packages installed")
        except subprocess.CalledProcessError:
            self.logger.warning("⚠️ Failed to install development packages")
    
    def create_sample_data(self):
        """Create sample data for testing"""
        self.logger.info("📊 Creating sample data...")
        
        try:
            from training_system import DatasetCreator
            
            # Create small sample dataset
            dataset_path = self.directories['data'] / 'sample_attribute_dataset'
            DatasetCreator.create_sample_dataset(
                output_dir=str(dataset_path),
                num_samples=100  # Small sample for testing
            )
            
            self.logger.info("✅ Sample dataset created")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create sample data: {e}")
    
    def print_next_steps(self):
        """Print next steps for the user"""
        next_steps = f"""
{Colors.GREEN}🎉 Setup Complete!{Colors.RESET}

{Colors.BOLD}Next Steps:{Colors.RESET}

{Colors.YELLOW}1. Quick Test:{Colors.RESET}
   python enhanced_pipeline.py --video 0

{Colors.YELLOW}2. Process a Video File:{Colors.RESET}
   python enhanced_pipeline.py --video your_video.mp4 --save-output

{Colors.YELLOW}3. Train Custom Models (Optional):{Colors.RESET}
   - Open Google Colab: notebooks/colab_training.ipynb
   - Or run locally: python training_system.py

{Colors.YELLOW}4. Configure for Your Use Case:{Colors.RESET}
   - Edit: configs/pipeline_config.json
   - Adjust detection confidence, tracking parameters, etc.

{Colors.YELLOW}5. Production Deployment:{Colors.RESET}
   python deploy_pipeline.py --scenario real_time_monitoring --video rtsp://camera_ip/stream

{Colors.BOLD}Documentation:{Colors.RESET}
   - README.md - Complete documentation
   - docs/ - Detailed guides
   - examples/ - Usage examples

{Colors.BOLD}Need Help?{Colors.RESET}
   - Run: python master_setup.py --help
   - Check: docs/TROUBLESHOOTING.md
   - Issues: https://github.com/your-repo/enhanced-cctv-pipeline/issues

{Colors.CYAN}{'='*60}{Colors.RESET}
        """
        print(next_steps)


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description="Enhanced CCTV Pipeline Setup and Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master_setup.py --setup-all          # Complete setup
  python master_setup.py --install-deps       # Install dependencies only  
  python master_setup.py --download-models    # Download models only
  python master_setup.py --test               # Run tests only
  python master_setup.py --dev                # Setup development environment
        """
    )
    
    # Setup options
    parser.add_argument('--setup-all', action='store_true',
                       help='Run complete setup process')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install dependencies only')
    parser.add_argument('--download-models', action='store_true',
                       help='Download default models only')
    parser.add_argument('--create-configs', action='store_true',
                       help='Create default configurations only')
    parser.add_argument('--test', action='store_true',
                       help='Run functionality tests only')
    parser.add_argument('--dev', action='store_true',
                       help='Setup development environment')
    parser.add_argument('--sample-data', action='store_true',
                       help='Create sample data for testing')
    
    # Configuration options
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip system requirement checks')
    parser.add_argument('--force-reinstall', action='store_true',
                       help='Force reinstall all dependencies')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    # Pipeline execution options
    parser.add_argument('--run-demo', action='store_true',
                       help='Run a quick demo after setup')
    parser.add_argument('--video', type=str,
                       help='Video source for demo (default: webcam)')
    
    # Version
    parser.add_argument('--version', action='version', version=f'Enhanced CCTV Pipeline v{__version__}')
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = PipelineSetup()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    setup.print_banner()
    
    # If no specific options, run complete setup
    if not any([args.install_deps, args.download_models, args.create_configs, 
                args.test, args.dev, args.sample_data]):
        args.setup_all = True
    
    success = True
    
    try:
        # System checks
        if not args.skip_checks and not setup.check_system_requirements():
            setup.logger.error("System requirements not met. Use --skip-checks to continue anyway.")
            return 1
        
        # Create project structure
        if args.setup_all:
            setup.create_project_structure()
        
        # Install dependencies
        if args.setup_all or args.install_deps:
            installed, missing = setup.check_dependencies()
            
            if args.force_reinstall:
                missing = installed + missing
                setup.logger.info("Force reinstalling all dependencies...")
            
            if missing and not setup.install_dependencies(missing):
                success = False
        
        # Download models
        if args.setup_all or args.download_models:
            setup.download_default_models()
        
        # Create configs
        if args.setup_all or args.create_configs:
            setup.create_default_configs()
        
        # Development environment
        if args.dev:
            setup.setup_development_environment()
        
        # Sample data
        if args.sample_data:
            setup.create_sample_data()
        
        # Run tests
        if args.setup_all or args.test:
            if not setup.run_tests():
                success = False
                setup.logger.warning("Some tests failed, but setup can continue")
        
        # Run demo
        if args.run_demo and success:
            setup.logger.info("🎬 Running demo...")
            video_source = args.video or "0"
            
            try:
                from enhanced_pipeline import EnhancedCVPipeline, EnhancedPipelineConfig
                
                config = EnhancedPipelineConfig(
                    enable_interactive_query=False,
                    save_output=False
                )
                
                pipeline = EnhancedCVPipeline(config)
                
                # Process just a few frames for demo
                import cv2
                cap = cv2.VideoCapture(int(video_source) if video_source.isdigit() else video_source)
                
                if cap.isOpened():
                    for i in range(10):  # Process 10 frames
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        results = pipeline.process_frame(frame)
                        setup.logger.info(f"Demo frame {i+1}: {len(results['tracked_persons'])} people detected")
                    
                    cap.release()
                    setup.logger.info("✅ Demo completed successfully")
                else:
                    setup.logger.warning(f"Could not open video source: {video_source}")
                
            except Exception as e:
                setup.logger.error(f"Demo failed: {e}")
        
        # Print results
        if success:
            setup.logger.info(f"{Colors.GREEN}🎉 Setup completed successfully!{Colors.RESET}")
            setup.print_next_steps()
        else:
            setup.logger.error(f"{Colors.RED}❌ Setup completed with errors{Colors.RESET}")
            return 1
    
    except KeyboardInterrupt:
        setup.logger.info(f"\n{Colors.YELLOW}⏹️ Setup interrupted by user{Colors.RESET}")
        return 1
    except Exception as e:
        setup.logger.error(f"{Colors.RED}❌ Setup failed: {e}{Colors.RESET}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
