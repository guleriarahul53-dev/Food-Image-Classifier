# 🍔 Food Image Classification App

A Deep Learning-based Food Image Classifier built using **TensorFlow, MobileNetV2, and Streamlit**.

Upload a food image and let AI predict its category along with confidence score.

---

## 📌 Project Overview

This project consists of two main parts:

1️⃣ Model Training Script (Transfer Learning using MobileNetV2)  
2️⃣ Streamlit Web Application (Frontend for predictions)

The model classifies food images into the following categories:

- Donuts  
- French Fries  
- Hamburger  
- Hot Dog  
- Pizza  
- Samosa  
- Sushi  
- Waffles  

---

# 🧠 Model Training

The model uses **MobileNetV2 (ImageNet pretrained weights)** with transfer learning.

## 🔹 Features

- Image size: `160x160`
- Data Augmentation:
  - Random Flip
  - Random Rotation
  - Random Zoom
- MobileNetV2 as base model (frozen)
- Global Average Pooling
- Dense Layer (256 units, ReLU)
- Dropout (0.5)
- Softmax Output Layer
- Optimizer: Adam (1e-4)
- Loss: Sparse Categorical Crossentropy

---


