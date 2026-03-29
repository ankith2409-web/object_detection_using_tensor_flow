# 🔍 Object Detection using TensorFlow

A deep learning project that performs **object detection** on handwritten digits — simultaneously classifying what the digit is and localizing where it appears in an image using bounding box regression.

Built with TensorFlow and Keras as a custom multi-output CNN from scratch.

---

## 📌 What It Does

Given an image with a digit placed at a random location, the model:
- **Classifies** the digit (0–9)
- **Localizes** it by predicting a bounding box `[xmin, ymin, xmax, ymax]`

This is the core idea behind object detection — not just *what* is in the image, but *where*.

---

## 🧠 Model Architecture

```
Input (75x75x1)
     │
     ▼
Conv2D(16) → AveragePooling
Conv2D(32) → AveragePooling
Conv2D(64) → AveragePooling
     │
     ▼
Flatten → Dense(128)
     │
     ├──► Dense(10, softmax)   → Classification Output (digit 0–9)
     └──► Dense(4)             → Bounding Box Output [xmin, ymin, xmax, ymax]
```

The model uses **two output heads** trained simultaneously:
| Head | Loss Function | Metric |
|------|--------------|--------|
| Classification | Categorical Crossentropy | Accuracy |
| Bounding Box | Mean Squared Error (MSE) | MSE |

---

## 📂 Dataset

- **Source:** [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) via `tensorflow_datasets`
- Each 28×28 digit is randomly placed inside a **75×75 canvas**
- Bounding box coordinates are normalized between 0 and 1
- Train set: 60,000 images | Validation set: 10,000 images

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | Model building & training |
| TensorFlow Datasets | MNIST data loading |
| NumPy | Array operations |
| Matplotlib | Visualization |
| Pillow (PIL) | Bounding box drawing |

---

## 📊 Evaluation

Model is evaluated using:
- **Classification Accuracy** — how often the digit label is correct
- **Bounding Box MSE** — how close predicted box coordinates are to ground truth
- **IoU (Intersection over Union)** — overlap between predicted and true bounding boxes (threshold: 0.6)

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/object-detection-tensorflow.git
cd object-detection-tensorflow
```

### 2. Install dependencies
```bash
pip install tensorflow tensorflow-datasets numpy matplotlib pillow
```

### 3. Run the notebook
Open `objectdetection.ipynb` in Jupyter or Google Colab and run all cells.

---

## 📈 Training

- **Optimizer:** Adam
- **Epochs:** 20
- **Batch Size:** 64
- **Strategy:** `tf.distribute` (supports multi-GPU)

---

## 📁 Project Structure

```
object-detection-tensorflow/
│
├── objectdetection.ipynb   # Main notebook
├── my_mnist_model.keras    # Saved trained model (after training)
└── README.md
```

---

## 💡 Key Concepts Demonstrated

- Multi-output neural networks
- Bounding box regression
- IoU-based evaluation
- Custom data preprocessing pipelines with `tf.data`
- Transfer of object detection concepts to digit localization

---

## 👤 Author

**HB Mrudhal Ankith**  
B.Tech AI & ML | Amity University, Bangalore  
[GitHub](https://github.com/Ankith1324) • [LinkedIn](https://linkedin.com/in/your-profile)
