# ♻️ E-Waste Image Classification using EfficientNetV2B3 and MobileNetV3Large

This project performs **image classification of e-waste types** using two state-of-the-art convolutional neural networks: **EfficientNetV2B3** and **MobileNetV3Large**. Both models are pretrained on ImageNet and fine-tuned on a custom dataset of 10 e-waste categories.

---

## 📁 Dataset Structure

The dataset is structured as:
/modified-dataset
/train
/Battery
/Keyboard
...
/val
/test


- **Classes (10 total)**: Battery, Keyboard, Microwave, Mobile, Mouse, PCB, Player, Printer, Television, Washing Machine
- **Input Formats**: JPEG, PNG
- **Splits**: `train`, `val`, `test`

---

## 🔧 Base Model Comparison

| Feature | MobileNetV3Large | EfficientNetV2B3 |
|--------|------------------|------------------|
| Input shape | `(128, 128, 3)` | `(300, 300, 3)` |
| Pretrained Weights | ImageNet | ImageNet |
| Total Layers | 186 | 409 |
| Trainable Layers | Last 30 (~24 trainable) | Last 25% (~83 trainable) |
| Total Params | 3.1M | 24.5M |
| Model Size | ~11.9MB | ~93.5MB |
| Output Feature Shape | `(None, 4, 4, 960)` | `(None, 10, 10, 1536)` |
| Mixed Precision | ✅ Yes | ❌ No |

---

## 🧪 Data & Augmentation Differences

| Feature | MobileNetV3Large | EfficientNetV2B3 |
|--------|------------------|------------------|
| Image Size | `128x128` | `300x300` |
| Batch Size | `32` | `24 (train), 16 (val/test)` |
| Augmentation | Flip, Rotate, Zoom, Brightness, Contrast | Flip, Rotate, Zoom |
| Preprocessing | Manual Rescaling | Assumed `preprocess_input` (EfficientNet) |

---

## 🧠 Model Architecture

| Feature | MobileNetV3Large | EfficientNetV2B3 |
|--------|------------------|------------------|
| After Base Model | GAP → Dropout(0.3) | GAP → Dropout(0.3) |
| Intermediate Dense | ✅ Dense(128, ReLU) + Dropout(0.2) | ❌ None |
| Output Layer | Dense(10, Softmax) | Dense(10, Softmax) |

---

## 🚀 Training Configurations

| Feature | MobileNetV3Large | EfficientNetV2B3 |
|--------|------------------|------------------|
| Epochs | 20 | 20 (early stopped at 10) |
| Learning Rate | `0.001` | `0.00005` |
| Optimizer | Adam | Adam |
| Mixed Precision | ✅ Enabled | ❌ Not Used |
| Callbacks | EarlyStopping, ReduceLROnPlateau, Checkpoint | Same |
| File Format | `.keras` | `.h5` + `.keras` |
| Final Training Accuracy | ~0.96 | ~0.98 |
| Final Validation Accuracy | ~0.96 | ~0.9867 |
| Epoch Time | ~8–10s | ~12–13 min |

---

## 🧩 Deployment Considerations

| Feature | MobileNetV3Large | EfficientNetV2B3 |
|--------|------------------|------------------|
| Model Size | ✅ ~12MB | ❌ ~93MB |
| Suitable for TFLite / TF.js | ✅ Yes | ❌ Needs optimization |
| Inference Speed (CPU) | ✅ Fast | ❌ Slower |
| GPU Optimization | ✅ Decent | ✅ Excellent |
| Training Cost (Colab) | ✅ Low | ❌ High |

---

## ✅ Summary: Which One to Use?

| Goal | Best Model |
|------|------------|
| Max Accuracy (Server-side) | **EfficientNetV2B3** |
| Web or Mobile Deployment | **MobileNetV3Large** |
| Low-cost Training | **MobileNetV3Large** |
| Feature Learning Capacity | **EfficientNetV2B3** (with added dense layer recommended) |

---

## 📂 Saved Models

- `mobilenet_model.keras` - Web-optimized, lightweight, fast inference.
- `efficientnet_model.keras` - Heavyweight, accurate, best for GPU inference.
- Both can be loaded directly for inference without retraining.

---

## Final Evaluation Results
📊 MobileNetV3Large Evaluation:

10/10 ━━━━━━━━━━━━━━━━━━━━ 51s 4s/step - accuracy: 0.9143 - loss: 0.2739
✅ Accuracy: 0.9167

📊 EfficientNetV2B3 Evaluation:

10/10 ━━━━━━━━━━━━━━━━━━━━ 74s 6s/step - accuracy: 0.9768 - loss: 0.1207
✅ Accuracy: 0.9800

---

## Contribution & License

[Pranav.TJ](https://github.com/PranavTJ-05)

---
