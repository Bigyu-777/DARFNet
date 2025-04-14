# DARFNet: Dynamic Attention Residual Fusion Network

**DARFNet** is a lightweight and efficient multispectral object detection network built on the YOLOv5 framework. It is specifically designed for detecting small objects in high-resolution **remote sensing imagery**, where traditional methods often struggle with low object resolution, complex backgrounds, and modality gaps between RGB and infrared images.

## 🔍 Highlights

- 🔸 **DARF Module**: A novel **Dynamic Attention Residual Fusion** module that adaptively learns spatial offsets and fuses RGB-IR features with high precision.
- 🔸 **Lightweight & Fast**: Only **14.3M parameters** and **36.3 GFLOPs**—ideal for edge computing and real-time deployment.
- 🔸 **Superior Performance**: Achieves state-of-the-art results on multiple remote sensing benchmarks, especially for small object detection.

---

## 📦 Model Overview

DARFNet integrates the **DARF module** into the YOLOv5 backbone to improve multispectral feature interaction. The module consists of:

- Dynamic offset convolution for spatially adaptive feature sampling.
- Channel interaction with residual connections.
- Efficient fusion strategy to maintain real-time performance.

---

## 📊 Benchmark Results

### 🛰️ VEDAI Dataset

| Method         | mAP@50 (%) | mAP (%) | Params (M) | GFLOPs |
|----------------|------------|---------|------------|--------|
| DACFusion      | 87.20      | 57.60   | -          | -      |
| ICAFusion      | 85.70      | 38.50   | 120.2      | 180.0  |
| C²Former       | 87.70      | 46.50   | 101.0      | 258.3  |
| **DARFNet (Ours)** | **99.80** | **71.40** | **14.3**     | **36.3**  |

### 🛰️ DroneVehicle Dataset

| Method          | AP (%) | AP50 (%) | Params (M) | GFLOPs |
|-----------------|--------|----------|------------|--------|
| CFT             | 50.1   | 70.1     | 206.0      | 403.9  |
| ViT-B+RVSA      | 42.6   | 65.1     | 60.5       | 134.7  |
| **DARFNet (Ours)** | **49.0** | **76.1**     | **14.3**     | **36.3**  |

### 🛰️ NWPU Dataset

| Method         | mAP@50 (%) | Params (M) | GFLOPs |
|----------------|------------|------------|--------|
| Faster R-CNN   | 77.80      | 41.17      | 127.70 |
| RetainNet      | 89.40      | 36.29      | 123.27 |
| YOLOv3         | 88.30      | 61.57      | 121.27 |
| GFL            | 88.80      | 19.13      | 91.73  |
| FCOS           | 89.65      | 31.86      | 116.63 |
| ATSS           | 90.50      | 18.96      | 89.90  |
| MobileNetV2    | 76.90      | 10.29      | 71.49  |
| ShuffleNet     | 83.00      | 12.10      | 82.17  |
| **DARFNet (Ours)** | **90.90** | **14.3**     | **36.3**  |

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/DARFNet.git
cd DARFNet
pip install -r requirements.txt
```

## 🚀 Usage
🔧 Training
Prepare your RGB+IR dataset in YOLO format. Update the .yaml file under data/ to include your dataset paths.
```bash
python train.py \
  --img 1024 \
  --batch 8 \
  --epochs 100 \
  --data data/vedai.yaml \
  --cfg models/darfnet.yaml \
```
