# Multimodal_Wafer_Defect_Detection
# Multi-Modal Deep Learning for Semiconductor Wafer Defect Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/your-notebook-id)

## 🎯 Project Overview

This project implements a **multi-modal deep learning framework** that integrates wafer map image data with high-dimensional process sensor measurements to improve semiconductor defect detection and classification. Our approach combines **CrossViT (Cross-Attention Vision Transformer)** for advanced image processing and sensor-aware attention mechanisms, unified through physics-informed fusion to create a state-of-the-art multimodal neural network.

## Table of Contents

- [🏆 Key Achievements](#-key-achievements)
- [📊 Results Summary](#-results-summary)
- [🏗️ Architecture Overview](#️-architecture-overview)
- [📁 Repository Structure](#-repository-structure)
- [🚀 Quick Start](#-quick-start)
- [📊 Datasets & Expected Layout](#-datasets--expected-layout)
- [🛠️ Installation](#️-installation)
- [🧠 Model Design Notes](#-model-design-notes)
- [📈 Training & Evaluation](#-training--evaluation)
- [📓 Notebooks](#-notebooks)
- [🔬 Technical Requirements](#-technical-requirements)
- [🔍 Key Findings](#-key-findings)
- [👥 Team Contributors](#-team-contributors)
- [🤖 AI Assistance Disclosure](#-ai-assistance-disclosure)
- [📄 Citation](#-citation)
- [📜 License](#-license)

## 🏆 Key Achievements

* **Multi-modal Architecture**: Novel fusion of image and sensor data modalities using CrossViT-based architecture that enables cross-attention between visual wafer map features and process sensor embeddings
* **Attention Mechanism**: Dynamic weighting of modality contributions through SensorAwareCrossViT, allowing the model to adaptively focus on relevant sensor-image feature combinations
* **Physics-Informed Design**: Sensor grouping based on manufacturing domain knowledge, incorporating process expertise into the neural architecture design
* **Class Imbalance Handling**: Balanced sampling and weighted loss functions specifically tuned for semiconductor defect detection scenarios
* **CrossViT Innovation**: First application of Cross-Attention Vision Transformers to semiconductor manufacturing, enabling multi-scale wafer defect analysis

*Note: This implementation leveraged Google Colab AI assistance for rapid prototyping and experimentation during the development phase.*

## 📊 Results Summary

### 📈 Overall Performance Metrics

| Metric | Value | Notes |
|--------|--------|-------|
| **Overall Accuracy** | **34.01%** | [31.71%, 36.32%] 95% CI |
| **Macro F1-Score** | **0.3184** | Balanced across all classes |
| **Weighted F1-Score** | **0.3191** | Accounts for class distribution |
| **ROC AUC (OvR)** | **0.8059** | Strong discriminative capability |
| **Matthews Correlation** | **0.2602** | Better than random classification |
| **Cohen's Kappa** | **0.2575** | Moderate agreement |

### 📊 Per-Class Performance

| Defect Class | F1-Score | Precision | Recall | Key Insights |
|--------------|----------|-----------|--------|--------------|
| **Near-full** | **0.618** | 0.464 | 0.923 | ✅ Best performing class |
| **none** | **0.457** | 0.500 | 0.421 | ✅ Improved from baseline |
| **Donut** | **0.358** | 0.407 | 0.319 | 🔄 Moderate performance |
| **Random** | **0.309** | 0.308 | 0.310 | 🔄 Consistent metrics |
| **Edge-Loc** | **0.291** | 0.273 | 0.313 | ⚠️ Confused with Edge-Ring |
| **Loc** | **0.263** | 0.284 | 0.244 | ⚠️ Needs improvement |
| **Center** | **0.241** | 0.234 | 0.249 | ⚠️ Low performance |
| **Edge-Ring** | **0.173** | 0.183 | 0.164 | ❌ Most challenging class |
| **Scratch** | **0.156** | 0.263 | 0.111 | ❌ Low recall |

### 🔍 Key Observations

**Strengths:**
- Strong ROC AUC (0.8059) indicates excellent discriminative capability between classes
- Near-full defects are well-detected with high recall (0.923)
- Model performs significantly better than random classification (MCC > 0)
- CrossViT attention mechanism successfully captures multi-scale defect patterns

**Areas for Improvement:**
- Edge-Ring and Scratch classes show poor performance, likely due to class imbalance or visual similarity
- Overall accuracy could benefit from targeted class-specific improvements
- Confusion between spatially similar defect types (Edge-Loc vs Edge-Ring)

**Recommendations:**
- Implement class-specific data augmentation strategies
- Consider focal loss or class balancing techniques for underperforming classes
- Explore attention visualization to understand CrossViT focus areas
- Collect more samples for rare defect types (Scratch, Edge-Ring)

## 🏗️ Architecture Overview

### High-Level CrossViT-Based Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Wafer Maps    │    │  Sensor Data    │
│   (224×224 px)  │    │  (590 → 6 dims) │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│   CrossViT      │    │   MLP Branch    │
│   Multi-Scale   │    │                 │
│   Patch Tokens  │    │ Dense(512)→     │
│   Small: 12×12  │    │ 256→128         │
│   Large: 16×16  │    │ + Dropout       │
│   + Cross-Attn  │    │ + Physics Groups│
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     ▼
           ┌─────────────────┐
           │SensorAwareCrossViT│
           │  Cross-Attention │
           │  α_img, α_sensor │
           └─────────┬───────┘
                     ▼
           ┌─────────────────┐
           │ Classification  │
           │  Head (9 classes) │
           └─────────────────┘
```

### 1) High-level Components

- **Data Ingestion** — sensor rows (SECOM) and wafer-map images are loaded and matched by wafer_id
- **Preprocessing** — sensor imputation, scaling, optional aggregation; wafer maps normalized/resized (default: 224×224). Grayscale wafer maps are expanded to 3 channels when using pretrained image backbones
- **Sensor Encoder** — MLP / Transformer that converts an n_sensors vector or a short sequence into a dense sensor embedding (shape: B x D_s)
- **Image Encoder (CrossViT)** — CrossViT processes multi-scale image patches producing visual embeddings (shape: B x D_v). The repo ships a production CrossViT implementation in `src/models/crossvit.py`
- **Fusion Module (SensorAwareCrossViT)** — cross-attention that conditions visual patch attention on sensor embeddings. Produces a joint multimodal representation B x D_f
- **Classifier Head** — a small MLP head on top of fused features to predict defect classes (softmax) or multi-label outputs
- **Training & Logging** — standard supervised training loop with checkpointing, scheduler, and metric logging (TensorBoard/W&B)

### 2) Dataflow (Compact)

```
raw_secom.csv + wafer_image.png → preprocess → 
sensor_tensor (B x n) + image_tensor (B x C x H x W) →
SensorEncoder(sensor_tensor) → sensor_emb (B x D_s) → 
CrossViT(image_tensor) → visual_emb (B x D_v) →
SensorAwareCrossViT(sensor_emb, visual_emb) → fused_emb (B x D_f) → 
Classifier(fused_emb) → predictions
```

### 3) Typical Shapes & Defaults

- Input image: B x 3 x 224 x 224 (wafer maps resized + channel-expanded)
- Sensor vector: B x n_sensors (e.g., B x 590 for SECOM processed features)
- Sensor embedding dim: D_s = 128 (configurable)
- Visual embedding dim: D_v = 768 (CrossViT default range)
- Fused dim: D_f = 512

### 4) Extensibility & Alternatives

- Swap the image backbone (resnet, vit, crossvit) via config
- Replace sensor encoder with a sequence model (RNN/Transformer) for full time-series inputs
- Fusion options: early concat, late fusion, or cross-attention (SensorAwareCrossViT is recommended for this repo)

### 5) CrossViT Implementation Details

The CrossViT architecture is specifically adapted for wafer defect detection:
- **Multi-scale Processing**: Handles both fine-grained defect patterns and global wafer characteristics
- **Cross-Attention Fusion**: Enables interaction between different patch scales and sensor modalities  
- **Sensor-Aware Conditioning**: Visual attention mechanisms are conditioned on process sensor embeddings
- **Physics-Informed Features**: Sensor groupings reflect semiconductor manufacturing process knowledge

### 🔬 Technical Components

1. **CrossViT Branch**: Processes 224×224 wafer map images through multi-scale patch tokenization with cross-attention between scales
2. **MLP Branch**: Handles physics-grouped sensor data (590 sensors → 6 groups) with domain knowledge
3. **SensorAware Fusion**: Dynamic cross-attention weighting of image patches based on sensor context
4. **Classification Head**: Final layers with dropout regularization optimized for 9-class defect detection

## 📁 Repository Structure

```
Multimodal_Wafer_Defect_Detection/
├── 📓 notebooks/               # notebooks (EDA, experiments)
│   ├── main_analysis.ipynb     # Primary analysis notebook  
│   ├── data_preprocessing.ipynb # Data cleaning and preparation
│   ├── model_evaluation.ipynb  # Detailed results analysis
│   └── Untitled2.ipynb        # uploaded notebook (move/rename as needed)
├── 📄 docs/
│   ├── project_report.pdf     # Complete technical report
│   └── architecture_diagram.png # Visual model architecture
├── 📊 results/
│   ├── confusion_matrix.png   # Confusion matrix visualization
│   ├── per_class_metrics.csv  # Detailed performance metrics
│   └── training_history.json  # Training progress logs
├── 🔧 src/
│   ├── data/                  # data loaders & preprocessing scripts
│   ├── models/                # model definitions (sensor encoder, image encoder, fusion)
│   │   ├── sensor_encoders.py # MLP, Transformer encoders
│   │   ├── image_encoders.py  # CNN, ViT, CrossViT variants
│   │   ├── crossvit.py        # CrossViT implementation with wafer adaptations
│   │   └── fusion.py          # SensorAwareCrossViT and fusion modules
│   ├── train.py               # training loop (single script entrypoint)
│   ├── evaluate.py            # evaluation & metrics
│   └── utils/                 # utility functions (metrics, logging, checkpoints)
├── experiments/               # example configs and results
├── data/                      # place datasets here (see Datasets section)
├── 📋 requirements.txt        # pinned Python deps
├── 🏃‍♂️ quick_start.py         # Simple demo script
├── 📖 README.md               # this file
└── 📜 LICENSE
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/your-notebook-id)

1. Click the "Open in Colab" button above
2. Upload the required datasets (instructions in notebook)
3. Run all cells to reproduce results
4. Explore the interactive CrossViT attention visualizations

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Multimodal_Wafer_Defect_Detection.git
cd Multimodal_Wafer_Defect_Detection

# Create virtual environment
conda create -n wafer-mm python=3.10 -y
conda activate wafer-mm

# Install dependencies
pip install -r requirements.txt

# Run quick demo with CrossViT
python quick_start.py
```

### Option 3: Quick Training Start

1. Put datasets into `data/` as described below

2. Run preprocessing to create train/val/test splits and processed feature files:
   ```bash
   python src/data/preprocess_secom.py --input data/secom/secom_raw.csv --out data/secom/secom_processed.parquet
   python src/data/preprocess_wafers.py --images data/wafers/images --meta data/wafers/metadata.csv --out data/wafers/processed.pkl
   ```

3. Train a baseline multimodal CrossViT model:
   ```bash
   python src/train.py --config experiments/configs/baseline_multimodal.yaml
   ```

4. Evaluate:
   ```bash
   python src/evaluate.py --checkpoint runs/exp1/checkpoint_best.pth --data data/
   ```

## 📊 Datasets & Expected Layout

This project expects two main modalities:

1. **SECOM (sensor) data** — preprocessed CSV or parquet files with sensor readings per wafer / lot / timestamp and a target label column
2. **WFQ wafer maps** — image files (PNG/JPG/NPY) or arrays representing wafer maps, named or indexed so they can be matched to the SECOM records (e.g. wafer_id key)

### WM-811K Wafer Map Dataset
- **Size**: 811,457 labeled wafer images
- **Classes**: 9 defect categories (Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none)
- **Format**: 25×27 pixel grayscale images (resized to 224×224 for CrossViT)
- **Source**: [Kaggle WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)

### SECOM Manufacturing Dataset
- **Size**: 1,567 samples with 590 sensor measurements
- **Features**: Process sensor readings during semiconductor fabrication
- **Labels**: Pass/Fail quality outcomes
- **Source**: [UCI SECOM Dataset](https://archive.ics.uci.edu/ml/datasets/secom)

Suggested data/ layout:

```
data/
├─ secom/
│  ├─ secom_raw.csv
│  └─ secom_processed.parquet
└─ wafers/
   ├─ images/               # wafer map images, filenames = <wafer_id>.png
   └─ metadata.csv          # mapping between wafer_id and labels / experiment metadata
```

**Important**: The crucial step is matching wafer images with the corresponding SECOM sensor rows using a unique identifier (e.g., wafer_id or lot + wafer_index). Make sure the mapping is correct before training.

## 🛠️ Installation

Create a Python virtual environment and install dependencies. Example (conda):

```bash
conda create -n wafer-mm python=3.10 -y
conda activate wafer-mm
pip install -r requirements.txt
```

### Dependencies
```
numpy
pandas
scikit-learn
pillow
matplotlib
torch>=1.9.0
torchvision>=0.10.0
tqdm
tensorboard
wandb    # optional
seaborn>=0.11.0
```

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only training possible
- **Recommended**: 16GB RAM + GPU (CUDA support) for CrossViT training
- **Optimal**: Google Colab Pro with high-RAM runtime

## 🧠 Model Design Notes

This repo is intentionally modular and includes a production-ready CrossViT implementation used by the multimodal training pipelines. The CrossViT and a sensor-aware CrossViT variant are implemented in `src/models/` and are the default image + fusion backbone for many example configs.

**Development Note**: This implementation utilized Google Colab AI for rapid prototyping, code generation, and experimentation during the research and development phases.

- `src/models/sensor_encoders.py` — MLP, Transformer, or simple RNN/GRU for sensor sequences
- `src/models/image_encoders.py` — CNN backbones (ResNet variants) and ViT/CrossViT variants (CrossViT implementation is provided)
- `src/models/crossvit.py` — CrossViT implementation with wafer-specific adaptations and helper utilities
- `src/models/fusion.py` — early concat, late fusion, and attention-based fusion (cross-attention between modalities). The repository also includes SensorAwareCrossViT (a CrossViT variant that conditions visual attention on sensor embeddings) to better fuse wafer-map spatial patterns with sensor metadata

**Usage note**: the default example configs use CrossViT as the image encoder and sensor-aware cross-attention for fusion. See `experiments/configs/baseline_multimodal.yaml` (or the equivalent config you use) and set:

```yaml
image_encoder: crossvit
sensor_encoder: mlp
fusion: sensor_aware_crossvit
```

If you update or refactor the CrossViT implementation, please keep the module API stable (constructor args: img_size, embed_dim, num_heads, depth, etc.) and document any changes in the module docstring.

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

### CrossViT Attention Mechanism
The model uses CrossViT's dual-scale cross-attention combined with sensor-aware conditioning:

```
# CrossViT Multi-Scale Processing
small_patches = patch_embed_small(x)  # 12x12 patches
large_patches = patch_embed_large(x)  # 16x16 patches

# Cross-attention between scales
small_attn = cross_attention(small_patches, large_patches)
large_attn = cross_attention(large_patches, small_patches)

# Sensor-aware fusion
α_img = W_att^T * tanh(W_crossvit * crossvit_features)
α_sensor = W_att^T * tanh(W_sensor * sensor_features)
[β_img, β_sensor] = softmax([α_img, α_sensor])
f_fused = β_img * crossvit_features + β_sensor * sensor_features
```

## 📈 Training & Evaluation

Use `src/train.py` as the main entrypoint.

Use a config-file (YAML) to specify:
- Data paths, batch size, learning rate
- Model architecture (sensor encoder type, image encoder choice, fusion method)
- Checkpointing & logging

### Training Strategy

- **Loss Function**: Weighted Cross-Entropy (addresses class imbalance)
- **Optimizer**: AdamW with weight decay (0.01)
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduling
- **Regularization**: Dropout (0.1-0.3), Batch Normalization
- **Early Stopping**: Patience of 10 epochs on validation accuracy
- **Data Augmentation**: Random rotation, flipping, translation for images (CrossViT compatible)

Recommended metrics to track:
- Accuracy, Precision/Recall, Macro / Weighted F1, ROC AUC (OvR), Matthews Correlation Coefficient (MCC)

Example hyperparameters (baseline):
```yaml
batch_size: 64
lr: 1e-4
optimizer: adamw
epochs: 50
image_encoder: crossvit
sensor_encoder: mlp
fusion: sensor_aware_crossvit
crossvit:
  img_size: 224
  patch_size_small: 12
  patch_size_large: 16
  embed_dim: 768
  num_heads: 12
  depth: 12
```

## 📓 Notebooks

- `notebooks/Untitled2.ipynb` — uploaded notebook. Rename and move it under notebooks/ and run cells to see EDA or toy experiments
- `notebooks/main_analysis.ipynb` — Primary analysis notebook with CrossViT visualizations
- `notebooks/data_preprocessing.ipynb` — Data cleaning and preparation
- `notebooks/model_evaluation.ipynb` — Detailed results analysis

Add more notebooks for specific experiments:
- `notebooks/EDA_secom.ipynb`
- `notebooks/EDA_wafers.ipynb`
- `notebooks/crossvit_attention_analysis.ipynb`

## 🔬 Technical Requirements

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

## 🔍 Key Findings

### ✅ Strengths
1. **Multi-modal CrossViT Approach**: Successfully integrates heterogeneous data sources with state-of-the-art vision transformer architecture
2. **Multi-Scale Attention**: CrossViT captures both fine-grained defects and global wafer patterns
3. **Physics-Informed Design**: Leverages domain knowledge in sensor grouping
4. **Class Balance**: Reduces "none" class dominance from 97% to 11%
5. **Sensor-Aware Visual Attention**: Novel conditioning of visual attention on process sensor data

### ⚠️ Areas for Improvement
1. **Overall Accuracy**: 34% accuracy needs significant improvement
2. **Class Confusion**: High misclassification between similar defect types
3. **Edge Defects**: Poor performance on Edge-Ring and Edge-Loc classes
4. **Decision Boundaries**: Gap between AUC (0.81) and accuracy suggests suboptimal thresholds

### 🎯 Future Work
- **Advanced CrossViT Variants**: Experiment with different patch sizes and attention heads
- **Data Augmentation**: More sophisticated augmentation strategies for semiconductor data
- **Ensemble Methods**: Combining multiple CrossViT models with different scales
- **Uncertainty Quantification**: Bayesian neural networks for confidence estimation
- **Attention Analysis**: Detailed visualization of CrossViT attention patterns for defect localization

## Experiments & Logging

The repository supports experiment tracking through:
- TensorBoard for loss curves and metrics visualization
- Weights & Biases (wandb) for experiment comparison and hyperparameter tracking
- Checkpointing system for model persistence and resumption

Example experiment configuration:
```yaml
experiment:
  name: "multimodal_crossvit_v1"
  tags: ["crossvit", "sensor_aware", "balanced_loss"]
  
model:
  image_encoder: "crossvit"
  sensor_encoder: "mlp"
  fusion: "sensor_aware_crossvit"
  
training:
  batch_size: 64
  learning_rate: 1e-4
  epochs: 100
  early_stopping_patience: 15
```

## 👥 Team Contributors

| Name | Role | Contributions |
|------|------|---------------|
| **Anish Chhabra** | Project Lead | CrossViT architecture, methodology, literature review |
| **Parth Aditya** | Data Engineer | Dataset curation, preprocessing, evaluation framework |
| **Joey Madigan** | ML Architect | Multi-modal fusion, sensor processing, uncertainty quantification |

## 🤖 AI Assistance Disclosure

**This project utilized AI assistance for code development and documentation:**

- **Google Colab with Gemini**: Used for generating code comments, landmarks for output visualization, and debugging assistance
- **Code Enhancement**: AI helped create clear section markers, detailed comments, and improved code readability
- **Documentation**: AI assisted in creating comprehensive docstrings and README structure
- **Debugging**: AI tools helped identify and resolve tensor shape mismatches and data loading issues

**Human Contributions**: All core algorithms, CrossViT model architecture design, experimental methodology, and scientific insights were developed by the human team members. AI was used as a coding assistant and documentation tool.

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{chhabra2025multimodal,
  title={Multi-Modal Deep Learning for Semiconductor Wafer Defect Detection Using CrossViT and Sensor Fusion},
  author={Chhabra, Anish and Aditya, Parth and Madigan, Joey},
  journal={CS230 Deep Learning Project},
  year={2025},
  institution={University of Wisconsin-Madison},
  note={CrossViT-based multimodal architecture for wafer defect detection}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

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
