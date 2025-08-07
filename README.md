# 🎯 Optimal CCTV-Vision Models

## Performance Results
- **Detection Accuracy**: 89.25%
- **Attribute Accuracy**: 93.20%
- **Expected vs Actual**: Detection 90.3% vs 89.2%, Attributes 70.6% vs 93.2%

## Optimal Configuration
- **Model**: efficientnet_b4
- **Learning Rate**: 5.56e-04
- **Batch Size**: 32
- **Dropout**: 0.698
- **Epochs**: 48

## Package Contents
- `best_detection_model.pth` - Person detection model
- `best_attribute_model.pth` - Multi-task attribute model
- `config.json` - Complete configuration and results
- `optimal_training_results.png` - Training visualization
- `README.md` - This documentation

## Usage
```python
import torch
import timm

# Load detection model
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)
checkpoint = torch.load('best_detection_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```
