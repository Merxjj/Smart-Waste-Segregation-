# Smart Waste Segregation Assistant

An AI-based waste classification system developed as a capstone project to automatically identify different categories of waste from images and support efficient recycling and waste segregation.

---

## 📌 Overview

Improper waste segregation is a major obstacle to efficient recycling and municipal waste management. This project leverages **Computer Vision and Deep Learning (Convolutional Neural Networks)** to automatically classify waste items into designated categories.

The application allows users to either upload an image from their device or enter an image web URL, passes the image through our trained CNN model, and outputs the predicted waste category alongside an interactive **Class Confidence Breakdown** chart.

---

## 🎯 Project Objectives

- **Automated Waste Identification**: Accurately classify waste into predefined categories using deep learning.
- **Support Recycling**: Reduce contamination in recyclable batches through consistent classification.
- **Hardware & CPU Optimized**: Leverage Intel oneDNN optimizations for fast, energy-efficient CPU inference.
- **Interactive Web Interface**: Provide a real-time, user-friendly Streamlit web app for practical testing and demonstration.

---

## 🏗️ System Workflow

```text
┌─────────────────┐
│   Waste Image   │ (Uploaded via Device or URL)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │ (RGB conversion, 300x300 Resizing, Pixel Normalization /255)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    CNN Model    │ (3 Conv Blocks with ReLU & MaxPooling)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Dense Classifier│ (Dense 64 -> Dropout -> Dense 32 -> Softmax)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Classification  │ (Predicted Category & Confidence Breakdown)
└─────────────────┘
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python 3.11** | Core development environment |
| **TensorFlow / Keras** | CNN model architecture, training, and weight loading |
| **Pillow (PIL)** | Image decoding and standardizing inputs to 3-channel RGB |
| **NumPy** | Array transformations and tensor operations |
| **Streamlit** | Web application dashboard and interactive deployment |
| **oneDNN** | CPU-level deep learning optimizations via Intel oneAPI |

---

## 🧠 Neural Network Architecture

The classification model is built using a Convolutional Neural Network (CNN) tailored for image feature extraction:

1. **Convolutional Block 1**: 32 filters (3x3 kernel, ReLU activation, same padding) + MaxPooling2D (2x2)
2. **Convolutional Block 2**: 64 filters (3x3 kernel, ReLU activation, same padding) + MaxPooling2D (2x2)
3. **Convolutional Block 3**: 32 filters (3x3 kernel, ReLU activation, same padding) + MaxPooling2D (2x2)
4. **Flatten Layer**: Transforms 2D feature maps to a 1D feature vector
5. **Fully Connected Layers**:
   - Dense(64, ReLU) + Dropout(0.2)
   - Dense(32, ReLU) + Dropout(0.2)
   - Dense(Output Layer, Softmax activation)

### Training Parameters:
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Cross-Entropy
- **Metric**: Accuracy
- **Batch Size**: 32
- **Epochs**: 10

---

## ⚙️ Intel oneDNN Acceleration

To optimize CPU inference without requiring an expensive dedicated GPU, the project enables Intel's Deep Neural Network Library:

```python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
```

This accelerates convolutional and dense matrix operations directly on compatible CPU architectures, ensuring low inference latency.

---

## 📊 Target Waste Categories

The system classifies waste items into the following categories:
- 🍾 **Glass**
- 🥫 **Metal**
- 📄 **Paper**
- 🥤 **Plastic**
- 🗑️ **Trash** (General non-recyclable residual waste)

---

## 🚀 Running the Project Locally

### 1. Clone the repository
```bash
git clone https://github.com/Merxjj/Smart-Waste-Segregation-.git
cd Smart-Waste-Segregation-
```

### 2. Set up virtual environment and install dependencies
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r Deployment/requirements.txt
```

### 3. Launch the Streamlit application
```bash
streamlit run Deployment/app.py
```
The application will open in your web browser at `http://localhost:8501`.

---

## 👥 Project Team

**Developed as a B.Tech Capstone Project by:**
- Meraj Alam
- Ali Ghaznfar
- Veer Singhi
- Garvit Jain

---

## 📜 Academic Capstone Project
*Submitted in partial fulfillment of the requirements for the degree of Bachelor of Technology.*
