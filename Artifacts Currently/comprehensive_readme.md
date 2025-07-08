# 🎯 Enhanced CCTV Computer Vision Pipeline

## Complete Person Detection, Tracking, and Attribute Classification System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive computer vision pipeline for CCTV surveillance that combines:
- 🎯 **Person Detection** using custom-trained YOLOv8/YOLOv11 models
- 🎬 **Multi-Object Tracking** with Kalman filtering
- 🏷️ **Attribute Classification** for gender, shirt color, and hair color
- 🔍 **Interactive Search** for finding specific people
- ⚡ **Real-time Processing** optimized for various hardware configurations

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🏗️ Project Structure](#️-project-structure)
- [🎓 Training Custom Models](#-training-custom-models)
- [🔧 Configuration](#-configuration)
- [💻 Usage Examples](#-usage-examples)
- [🏭 Production Deployment](#-production-deployment)
- [📊 Performance Optimization](#-performance-optimization)
- [🧪 Testing](#-testing)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- 4GB+ RAM
- OpenCV-compatible camera or video files

### 1-Minute Setup
```bash
# Clone the repository
git clone https://github.com/your-repo/enhanced-cctv-pipeline.git
cd enhanced-cctv-pipeline

# Run the setup script
bash setup_environment.sh

# Activate environment
conda activate enhanced_cctv

# Run basic demo
python enhanced_pipeline.py --video 0  # Use webcam
```

### Quick Demo with Sample Video
```bash
# Download sample video (optional)
wget https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4

# Run pipeline
python enhanced_pipeline.py --video SampleVideo_1280x720_1mb.mp4 --save-output
```

## 📦 Installation

### Option 1: Automated Setup (Recommended)
```bash
# Download and run the setup script
curl -fsSL https://raw.githubusercontent.com/your-repo/enhanced-cctv-pipeline/main/setup_environment.sh | bash
```

### Option 2: Manual Installation
```bash
# Create conda environment
conda create -n enhanced_cctv python=3.9 -y
conda activate enhanced_cctv

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_installation.py
```

### Option 3: Docker Installation
```bash
# Build Docker image
docker build -t enhanced-cctv-pipeline .

# Run container
docker run -it --gpus all -v $(pwd):/workspace enhanced-cctv-pipeline
```

## 🏗️ Project Structure

```
enhanced-cctv-pipeline/
│
├── 📁 src/                          # Source code
│   ├── person_detector.py           # YOLO-based person detection
│   ├── person_tracker.py            # Multi-object tracking with Kalman filtering
│   ├── attribute_classifier.py      # Gender, shirt color, hair color classification
│   ├── training_system.py           # Model training with Optuna optimization
│   ├── enhanced_pipeline.py         # Main integrated pipeline
│   └── usage_examples.py            # Usage examples and deployment configs
│
├── 📁 data/                         # Data directory
│   ├── models/                      # Trained model files
│   ├── videos/                      # Input video files
│   ├── attribute_dataset/           # Training data for attributes
│   └── output/                      # Processing results
│
├── 📁 configs/                      # Configuration files
│   ├── detection_config.yaml        # Detection parameters
│   ├── tracking_config.yaml         # Tracking parameters
│   └── pipeline_config.yaml         # Complete pipeline configuration
│
├── 📁 notebooks/                    # Jupyter notebooks
│   ├── training_demo.ipynb          # Model training demonstration
│   ├── evaluation_analysis.ipynb    # Performance evaluation
│   └── colab_training.ipynb         # Google Colab training notebook
│
├── 📁 tests/                        # Test suite
│   ├── test_detection.py            # Detection module tests
│   ├── test_tracking.py             # Tracking module tests
│   ├── test_attributes.py           # Attribute classification tests
│   └── test_integration.py          # End-to-end integration tests
│
├── 📁 scripts/                      # Utility scripts
│   ├── setup_environment.sh         # Environment setup
│   ├── download_models.py           # Pre-trained model downloader
│   ├── create_dataset.py            # Dataset creation utilities
│   └── deploy_pipeline.py           # Production deployment script
│
├── 📁 docs/                         # Documentation
│   ├── INSTALLATION.md              # Detailed installation guide
│   ├── API_REFERENCE.md             # API documentation
│   ├── TRAINING_GUIDE.md            # Model training guide
│   └── DEPLOYMENT_GUIDE.md          # Production deployment guide
│
├── requirements.txt                 # Python dependencies
├── setup.py                        # Package installation
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Multi-container setup
└── README.md                       # This file
```

## 🎓 Training Custom Models

### Option 1: Google Colab Training (Recommended for Beginners)

1. **Open the Colab Notebook**
   ```
   https://colab.research.google.com/github/your-repo/enhanced-cctv-pipeline/blob/main/notebooks/colab_training.ipynb
   ```

2. **Follow the Step-by-Step Guide**
   - Setup environment and dependencies
   - Prepare/upload your dataset
   - Configure training parameters
   - Train multiple model sizes (nano, small, medium)
   - Evaluate and compare models
   - Download trained models

3. **Integration**
   ```python
   # After downloading from Colab
   from person_detector import PersonDetector, DetectionConfig
   
   config = DetectionConfig(
       model_path="data/models/person_detector_yolov8s_best.pt"  # Your trained model
   )
   detector = PersonDetector(config)
   ```

### Option 2: Local Training

1. **Prepare Your Dataset**
   ```bash
   # Create dataset structure
   python scripts/create_dataset.py --output data/person_dataset --samples 5000
   
   # Or prepare your own dataset
   mkdir -p data/person_dataset/{images,labels}/{train,val,test}
   ```

2. **Configure Training**
   ```python
   from training_system import TrainingConfig, AttributeTrainer
   
   config = TrainingConfig(
       dataset_path="data/person_dataset",
       epochs=100,
       batch_size=32,
       use_wandb=True  # Enable experiment tracking
   )
   ```

3. **Start Training**
   ```bash
   python training_system.py --config configs/training_config.yaml
   ```

4. **Hyperparameter Optimization**
   ```bash
   python training_system.py --optimize --trials 50
   ```

### Dataset Requirements

For optimal results, your dataset should include:
- **Person Detection**: 2000+ annotated images with bounding boxes
- **Attribute Classification**: 1000+ person crops with labels for:
  - Gender: male, female
  - Shirt colors: red, blue, green, black, white, yellow, pink, purple, orange, brown, gray
  - Hair colors: black, brown, blonde, red, gray, white

## 🔧 Configuration

### Basic Configuration
```python
from enhanced_pipeline import EnhancedPipelineConfig

# Real-time monitoring
config = EnhancedPipelineConfig(
    detector_model="data/models/person_detector_yolov8n_best.pt",
    detection_confidence=0.6,
    tracking_iou_threshold=0.3,
    max_tracks_to_classify=10,
    target_fps=30.0
)
```

### Advanced Configuration
```python
# High-accuracy forensic analysis
config = EnhancedPipelineConfig(
    detector_model="data/models/person_detector_yolov8m_best.pt",
    detection_confidence=0.4,
    attribute_model="data/models/attribute_classifier.pth",
    tracking_iou_threshold=0.2,
    max_disappeared=100,
    use_kalman=True,
    batch_processing=True,
    save_output=True,
    save_annotations=True
)
```

### Configuration Files
```yaml
# configs/pipeline_config.yaml
detection:
  model_path: "data/models/person_detector_yolov8s_best.pt"
  confidence_threshold: 0.6
  iou_threshold: 0.7

tracking:
  iou_threshold: 0.3
  max_disappeared: 30
  min_hits: 3
  use_kalman: true

attributes:
  model_path: "data/models/attribute_classifier.pth"
  confidence_threshold: 0.5
  batch_processing: true

performance:
  target_fps: 30.0
  device: "auto"
  mixed_precision: true
```

## 💻 Usage Examples

### 1. Basic Person Detection
```python
from person_detector import PersonDetector
import cv2

# Initialize detector
detector = PersonDetector("yolov8n.pt")

# Process single image
image = cv2.imread("test_image.jpg")
detections = detector.detect_persons(image, return_crops=True)

print(f"Found {len(detections['boxes'])} people")
```

### 2. Complete Pipeline with Tracking
```python
from enhanced_pipeline import EnhancedCVPipeline, EnhancedPipelineConfig

# Configure pipeline
config = EnhancedPipelineConfig(
    detector_model="data/models/custom_person_detector.pt",
    attribute_model="data/models/attribute_classifier.pth",
    enable_interactive_query=True
)

# Initialize and run
pipeline = EnhancedCVPipeline(config)
summary = pipeline.run_video("security_footage.mp4", output_path="results.mp4")
```

### 3. Interactive Person Search
```python
# Set search criteria
pipeline.current_query = {
    'gender': 'female',
    'shirt_color': 'red',
    'hair_color': None  # Any hair color
}

# Process video with search highlighting
results = pipeline.process_frame(frame)
matches = results['query_matches']

print(f"Found {len(matches)} people matching criteria")
```

### 4. Multi-Camera Setup
```python
from usage_examples import MultiCameraManager

# Configure multiple cameras
camera_configs = {
    "entrance": EnhancedPipelineConfig(detector_model="yolov8s.pt"),
    "hallway": EnhancedPipelineConfig(detector_model="yolov8n.pt"),
    "exit": EnhancedPipelineConfig(detector_model="yolov8s.pt")
}

# Initialize and start
manager = MultiCameraManager(camera_configs)
manager.start_all_cameras()
```

### 5. Batch Processing
```python
# Process multiple videos
video_files = ["cam1.mp4", "cam2.mp4", "cam3.mp4"]

for video_file in video_files:
    print(f"Processing {video_file}...")
    summary = pipeline.run_video(video_file, output_path=f"results_{video_file}")
    print(f"Detected {summary['tracking_stats']['total_tracks']} people")
```

## 🏭 Production Deployment

### Real-time Monitoring Setup
```bash
# Production deployment script
python deploy_pipeline.py \
    --scenario real_time_monitoring \
    --video rtsp://camera_ip:554/stream \
    --output-dir /var/log/cctv \
    --alert-thresholds alerts_config.json
```

### Docker Production Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  cctv-pipeline:
    build: .
    volumes:
      - ./data:/app/data
      - ./output:/app/output
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: python deploy_pipeline.py --scenario real_time_monitoring
  
  monitoring:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-storage:/var/lib/grafana
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cctv-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cctv-pipeline
  template:
    metadata:
      labels:
        app: cctv-pipeline
    spec:
      containers:
      - name: pipeline
        image: enhanced-cctv-pipeline:latest
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "4Gi"
            cpu: "2"
```

### Performance Monitoring
```python
# Setup monitoring and alerting
from usage_examples import ProductionDeployment

# Create monitoring
monitor = ProductionDeployment.setup_monitoring(
    pipeline,
    alert_thresholds={
        'min_fps': 15.0,
        'max_people_per_frame': 25,
        'max_processing_time': 0.3
    }
)

# Check performance
alerts = monitor.get_recent_alerts(minutes=5)
for alert in alerts:
    print(f"Alert: {alert['type']} - {alert['value']}")
```

## 📊 Performance Optimization

### Hardware Recommendations

| Use Case | CPU | GPU | RAM | Storage |
|----------|-----|-----|-----|---------|
| **Development** | 4+ cores | GTX 1060+ | 8GB | 100GB SSD |
| **Single Camera** | 6+ cores | RTX 3060+ | 16GB | 500GB SSD |
| **Multi-Camera (4+)** | 8+ cores | RTX 3080+ | 32GB | 1TB SSD |
| **Edge Deployment** | ARM64 | None | 4GB | 64GB eMMC |

### Performance Tuning

#### For Real-time Applications
```python
config = EnhancedPipelineConfig(
    detector_model="yolov8n.pt",  # Fastest model
    detection_confidence=0.7,     # Higher threshold = faster
    batch_processing=True,
    max_tracks_to_classify=5,     # Limit attribute processing
    show_trails=False,            # Disable trails
    save_output=False             # Disable saving
)
```

#### For High Accuracy
```python
config = EnhancedPipelineConfig(
    detector_model="yolov8x.pt",  # Most accurate model
    detection_confidence=0.3,     # Lower threshold = more detections
    tracking_iou_threshold=0.2,   # More sensitive tracking
    max_disappeared=100,          # Longer tracking
    use_kalman=True,             # Motion prediction
    batch_processing=True
)
```

#### For Edge Devices
```python
config = EnhancedPipelineConfig(
    detector_model="yolov8n.pt",
    detection_confidence=0.8,     # Very conservative
    batch_processing=False,       # Individual processing
    max_tracks_to_classify=3,     # Minimal attribute processing
    show_attributes=False,        # Disable attributes
    device="cpu"                  # Force CPU
)
```

### Model Optimization
```bash
# Export optimized models
python scripts/optimize_models.py \
    --input data/models/person_detector.pt \
    --formats onnx tensorrt \
    --precision fp16
```

### Profiling and Benchmarking
```bash
# Run performance benchmark
python scripts/benchmark.py \
    --config configs/production_config.yaml \
    --duration 60 \
    --output benchmark_results.json

# Profile memory usage
python -m memory_profiler enhanced_pipeline.py --video test.mp4
```

## 🧪 Testing

### Run All Tests
```bash
# Complete test suite
python -m pytest tests/ -v

# Run specific test categories
python testing_framework.py detector
python testing_framework.py tracker
python testing_framework.py classifier
python testing_framework.py pipeline
```

### Integration Testing
```bash
# End-to-end pipeline test
python tests/test_integration.py

# Performance regression tests
python tests/test_performance.py --baseline baseline_metrics.json
```

### Custom Testing
```python
from testing_framework import TestDataGenerator, IntegrationTester

# Create test data
generator = TestDataGenerator()
test_video = generator.create_test_video("test_output.mp4", num_frames=100)

# Run integration tests
tester = IntegrationTester()
results = tester.test_complete_workflow()
```

## 📈 Monitoring and Analytics

### Built-in Metrics
- **Performance**: FPS, processing times, memory usage
- **Detection**: Person count, confidence distributions
- **Tracking**: Track lifecycle, ID consistency
- **Attributes**: Classification accuracy, confidence scores

### Integration with Monitoring Systems
```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

detection_counter = Counter('detections_total', 'Total detections')
processing_time = Histogram('processing_seconds', 'Processing time')
active_tracks = Gauge('active_tracks', 'Number of active tracks')

# Log to monitoring system
def log_metrics(results):
    detection_counter.inc(len(results['detections']['boxes']))
    processing_time.observe(results['processing_times']['total'])
    active_tracks.set(len(results['tracked_persons']))
```

### Dashboard Setup
```bash
# Start monitoring stack
docker-compose -f monitoring/docker-compose.yml up -d

# Access dashboards
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

## 🐛 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Low FPS Performance
```python
# Optimize configuration
config.detector_model = "yolov8n.pt"  # Use faster model
config.detection_confidence = 0.7     # Higher threshold
config.batch_processing = True        # Enable batching
config.max_tracks_to_classify = 5     # Limit processing
```

#### Memory Issues
```python
# Reduce memory usage
config.batch_size = 8                 # Smaller batches
config.max_detections = 50            # Limit detections
config.save_output = False            # Disable saving
```

#### Model Loading Errors
```bash
# Verify model files
python scripts/verify_models.py

# Download missing models
python scripts/download_models.py --model yolov8n.pt
```

### Debug Mode
```bash
# Enable debug logging
export PIPELINE_LOG_LEVEL=DEBUG
python enhanced_pipeline.py --video test.mp4 --debug
```

### Performance Profiling
```bash
# Profile CPU usage
python -m cProfile -o profile.stats enhanced_pipeline.py --video test.mp4

# Analyze profile
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/enhanced-cctv-pipeline.git
cd enhanced-cctv-pipeline

# Create development environment
conda env create -f environment-dev.yml
conda activate enhanced-cctv-dev

# Install in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Code Standards
- **Python**: PEP 8, type hints, docstrings
- **Testing**: pytest, >90% coverage
- **Documentation**: Comprehensive docstrings and examples
- **Performance**: Benchmark critical paths

### Submitting Changes
1. Create feature branch: `git checkout -b feature/amazing-feature`
2. Make changes and add tests
3. Run test suite: `python -m pytest tests/`
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push branch: `git push origin feature/amazing-feature`
6. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for the excellent YOLOv8 implementation
- **OpenCV** community for computer vision tools
- **PyTorch** team for the deep learning framework
- **Contributors** who helped improve this project

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-repo/enhanced-cctv-pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/enhanced-cctv-pipeline/discussions)
- **Email**: support@your-domain.com

## 🗺️ Roadmap

### Version 2.0 (Planned)
- [ ] Real-time re-identification across cameras
- [ ] Advanced behavioral analysis
- [ ] Cloud deployment templates
- [ ] Mobile app for monitoring
- [ ] Advanced analytics dashboard

### Version 2.1 (Future)
- [ ] 3D pose estimation
- [ ] Emotion recognition
- [ ] Integration with access control systems
- [ ] Privacy-preserving features
- [ ] Automated report generation

---

## 🏁 Getting Started Checklist

- [ ] Install dependencies with `bash setup_environment.sh`
- [ ] Test installation with `python test_installation.py`
- [ ] Run basic demo with `python enhanced_pipeline.py --video 0`
- [ ] Train custom models (optional) using Google Colab notebook
- [ ] Configure for your use case in `configs/pipeline_config.yaml`
- [ ] Deploy to production with `python deploy_pipeline.py`
- [ ] Set up monitoring and alerts
- [ ] Scale to multiple cameras as needed

**Need help?** Check our [troubleshooting guide](docs/TROUBLESHOOTING.md) or [open an issue](https://github.com/your-repo/enhanced-cctv-pipeline/issues/new).

---

*Built with ❤️ for the computer vision community*