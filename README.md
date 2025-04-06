# 🌿 Plant Disease Detection using Deep Learning

A smart plant disease detection web application built with **Streamlit** and **TensorFlow**, designed to help farmers, researchers, and plant enthusiasts diagnose diseases from leaf images with high accuracy.


## 🚀 Features
- 🔍 Detects **38 plant diseases** using a CNN trained on **87,000+ leaf images**
- 🧠 Built with **TensorFlow** deep learning models
- 📊 Interactive and user-friendly UI with **Streamlit**
- 🌱 Pages: **Home**, **About**, and **Disease Recognition**
- 📷 Upload your own leaf image to get instant disease prediction
- 💡 Insightful results with predicted disease name and confidence score

## 🧰 Tech Stack
- **Frontend/UI:** Streamlit
- **Deep Learning:** TensorFlow, Keras
- **Data:** PlantVillage dataset (with custom preprocessing)
- **Language:** Python 3.9+

## 🧪 Dataset
- Sourced from the popular **Kaggle**
- Includes 87,000+ labeled images across **38 distinct classes**

## 🧠 Model Architecture
- Convolutional Neural Network (CNN)
- Layers: Conv2D → MaxPool → Dropout → Flatten → Dense
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Accuracy: 97%
