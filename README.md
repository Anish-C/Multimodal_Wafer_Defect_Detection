# Multimodal_Wafer_Defect_Detection
# Multi-Modal Deep Learning for Semiconductor Wafer Defect Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/your-notebook-id)

## 🎯 Project Overview

This project implements a **multi-modal deep learning framework** that integrates wafer map image data with high-dimensional process sensor measurements to improve semiconductor defect detection and classification. Our approach combines CNNs for image processing and MLPs for sensor data analysis, unified through attention-based fusion mechanisms.

### 🏆 Key Achievements
- **Multi-modal Architecture**: Novel fusion of image and sensor data modalities
- **Attention Mechanism**: Dynamic weighting of modality contributions
- **Physics-Informed Design**: Sensor grouping based on manufacturing domain knowledge
- **Class Imbalance Handling**: Balanced sampling and weighted loss functions

## 📊 Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **Overall Accuracy** | 34.01% | [31.71%, 36.32%] 95% CI |
| **Macro F1-Score** | 0.3184 | Balanced across all classes |
| **Weighted F1-Score** | 0.3191 | Accounts for class distribution |
| **ROC AUC (OvR)** | 0.8059 | Strong discriminative capability |
| **Matthews Correlation** | 0.2602 | Better than random classification |
| **Cohen's Kappa** | 0.2575 | Moderate agreement |

### 📈 Per-Class Performance

| Defect Class | F1-Score | Precision | Recall | Key Insights |
|--------------|----------|-----------|--------|--------------|
| **Near-full** | 0.618 | 0.464 | 0.923 | ✅ Best performing class |
| **none** | 0.457 | 0.500 | 0.421 | ✅ Improved from baseline |
| **Donut** | 0.358 | 0.407 | 0.319 | 🔄 Moderate performance |
| **Random** | 0.309 | 0.308 | 0.310 | 🔄 Consistent metrics |
| **Edge-Loc** | 0.291 | 0.273 | 0.313 | ⚠️ Confused with Edge-Ring |
| **Loc** | 0.263 | 0.284 | 0.244 | ⚠️ Needs improvement |
| **Center** | 0.241 | 0.234 | 0.249 | ⚠️ Low performance |
| **Edge-Ring** | 0.173 | 0.183 | 0.164 | ❌ Most challenging class |
| **Scratch** | 0.156 | 0.263 | 0.111 | ❌ Low recall |

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐
│   Wafer Maps    │    │  Sensor Data    │
│   (25×27 px)    │    │  (590 → 6 dims) │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│   CNN Branch    │    │   MLP Branch    │
│                 │    │                 │
│ Conv2D(32)→64→  │    │ Dense(512)→     │
│ 128→256         │    │ 256→128         │
│ + BatchNorm     │    │ + Dropout       │
│ + MaxPool       │    │                 │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     ▼
           ┌─────────────────┐
           │ Attention Fusion │
           │   α_img, α_sensor │
           └─────────┬───────┘
                     ▼
           ┌─────────────────┐
           │ Classification  │
           │  Head (9 classes) │
           └─────────────────┘
```

### 🔬 Technical Components

1. **CNN Branch**: Processes 25×27 wafer map images through 4 convolutional layers
2. **MLP Branch**: Handles physics-grouped sensor data (590 sensors → 6 groups)
3. **Attention Fusion**: Dynamic weighting of image vs. sensor contributions
4. **Classification Head**: Final layers with dropout regularization

## 📁 Repository Structure

```
wafer-defect-detection/
├── 📓 notebooks/
│   ├── main_analysis.ipynb          # Primary analysis notebook
│   ├── data_preprocessing.ipynb     # Data cleaning and preparation
│   └── model_evaluation.ipynb      # Detailed results analysis
├── 📄 docs/
│   ├── project_report.pdf          # Complete technical report
│   └── architecture_diagram.png    # Visual model architecture
├── 📊 results/
│   ├── confusion_matrix.png        # Confusion matrix visualization
│   ├── per_class_metrics.csv       # Detailed performance metrics
│   └── training_history.json       # Training progress logs
├── 🔧 src/
│   ├── models.py                   # Model architecture definitions
│   ├── data_utils.py               # Data loading and preprocessing
│   ├── training.py                 # Training loops and utilities
│   └── evaluation.py               # Evaluation metrics and plots
├── 📋 requirements.txt              # Python dependencies
├── 🏃‍♂️ quick_start.py               # Simple demo script
└── 📖 README.md                    # This file
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/your-notebook-id)

1. Click the "Open in Colab" button above
2. Upload the required datasets (instructions in notebook)
3. Run all cells to reproduce results
4. Explore the interactive visualizations

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/wafer-defect-detection.git
cd wafer-defect-detection

# Install dependencies
pip install -r requirements.txt

# Run quick demo
python quick_start.py
```

## 📊 Datasets

### WM-811K Wafer Map Dataset
- **Size**: 811,457 labeled wafer images
- **Classes**: 9 defect categories (Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none)
- **Format**: 25×27 pixel grayscale images
- **Source**: [Kaggle WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)

### SECOM Manufacturing Dataset
- **Size**: 1,567 samples with 590 sensor measurements
- **Features**: Process sensor readings during semiconductor fabrication
- **Labels**: Pass/Fail quality outcomes
- **Source**: [UCI SECOM Dataset](https://archive.ics.uci.edu/ml/datasets/secom)

## 🛠️ Technical Requirements

### Dependencies
```
torch>=1.9.0
torchvision>=0.10.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
Pillow>=8.3.0
tqdm>=4.62.0
```

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only training possible
- **Recommended**: 16GB RAM + GPU (CUDA support)
- **Optimal**: Google Colab Pro with high-RAM runtime

## 🧠 Model Architecture Details

### Physics-Based Sensor Grouping
```python
sensor_groups = {
    'pressure': sensors[0:98],      # Pressure measurements
    'temperature': sensors[98:196], # Temperature readings  
    'flow': sensors[196:294],       # Flow rate data
    'chemistry': sensors[294:392],  # Chemical concentrations
    'power': sensors[392:490],      # Power/voltage levels
    'timing': sensors[490:590]      # Timing/cycle parameters
}
```

### Attention Mechanism
The model uses learnable attention weights to dynamically balance image and sensor contributions:

```
α_img = W_att^T * tanh(W_img * f_img)
α_sensor = W_att^T * tanh(W_sensor * f_sensor)
[β_img, β_sensor] = softmax([α_img, α_sensor])
f_fused = β_img * f_img + β_sensor * f_sensor
```

## 📈 Training Strategy

- **Loss Function**: Weighted Cross-Entropy (addresses class imbalance)
- **Optimizer**: AdamW with weight decay (0.01)
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduling
- **Regularization**: Dropout (0.1-0.3), Batch Normalization
- **Early Stopping**: Patience of 10 epochs on validation accuracy
- **Data Augmentation**: Random rotation, flipping, translation for images

## 🔍 Key Findings

### ✅ Strengths
1. **Multi-modal Approach**: Successfully integrates heterogeneous data sources
2. **Attention Mechanism**: Provides interpretable modality weighting
3. **Physics-Informed Design**: Leverages domain knowledge in sensor grouping
4. **Class Balance**: Reduces "none" class dominance from 97% to 11%

### ⚠️ Areas for Improvement
1. **Overall Accuracy**: 34% accuracy needs significant improvement
2. **Class Confusion**: High misclassification between similar defect types
3. **Edge Defects**: Poor performance on Edge-Ring and Edge-Loc classes
4. **Decision Boundaries**: Gap between AUC (0.81) and accuracy suggests suboptimal thresholds

### 🎯 Future Work
- **Advanced Architectures**: Vision Transformers, ResNet backbones
- **Data Augmentation**: More sophisticated augmentation strategies
- **Ensemble Methods**: Combining multiple model predictions
- **Uncertainty Quantification**: Bayesian neural networks for confidence estimation

## 👥 Team Contributors

| Name | Role | Contributions |
|------|------|---------------|
| **Anish Chhabra** | Project Lead | CNN architecture, methodology, literature review |
| **Parth Aditya** | Data Engineer | Dataset curation, preprocessing, evaluation framework |
| **Joey Madigan** | ML Architect | Multi-modal fusion, sensor processing, uncertainty quantification |

## 🤖 AI Assistance Disclosure

**This project utilized AI assistance for code development and documentation:**

- **Google Colab with Gemini**: Used for generating code comments, landmarks for output visualization, and debugging assistance
- **Code Enhancement**: AI helped create clear section markers, detailed comments, and improved code readability
- **Documentation**: AI assisted in creating comprehensive docstrings and README structure
- **Debugging**: AI tools helped identify and resolve tensor shape mismatches and data loading issues

**Human Contributions**: All core algorithms, model architecture design, experimental methodology, and scientific insights were developed by the human team members. AI was used as a coding assistant and documentation tool.

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{chhabra2025multimodal,
  title={Multi-Modal Deep Learning for Semiconductor Wafer Defect Detection Using Image and Sensor Fusion},
  author={Chhabra, Anish and Aditya, Parth and Madigan, Joey},
  journal={CS230 Deep Learning Project},
  year={2025},
  institution={University of Wisconsin-Madison}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Project Report**: [PDF](docs/project_report.pdf)
- **Google Colab Notebook**: [Interactive Demo](https://colab.research.google.com/drive/your-notebook-id)
- **WM-811K Dataset**: [Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- **SECOM Dataset**: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/secom)

## 📞 Contact

For questions or collaboration opportunities:

- **Anish Chhabra**: chhabra8@wisc.edu
- **Parth Aditya**: paditya2@wisc.edu  
- **Joey Madigan**: jpmadigan@wisc.edu

---

⭐ **Star this repository if you found it helpful!** ⭐
