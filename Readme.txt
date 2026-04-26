# 🌾 AgriScan - Rice Leaf Disease Detection System

## 📌 Overview

AgriScan is an AI-powered web application designed to detect rice leaf diseases using deep learning. The system allows users to upload an image of a rice leaf and receive predictions along with confidence levels, causes, and treatment suggestions.

---

## 🚀 Features

* Upload rice leaf images via a simple UI
* Detect multiple disease types
* Confidence score for predictions
* Displays cause and treatment suggestions
* Handles unknown or low-confidence cases
* Real-time inference using a trained CNN model

---

## 🧠 Technologies Used

* **TensorFlow / Keras** – Deep learning model training and inference
* **MobileNetV2** – Pretrained CNN for feature extraction
* **Flask** – Backend API for prediction
* **Streamlit** – Frontend user interface
* **NumPy & Pillow** – Image preprocessing

---

## 📂 Project Structure

```
project/
│
├── app.py                 # Flask backend
├── streamlit_app.py       # Streamlit frontend
├── agriscan_model.h5      # Trained model
├── requirements.txt       # Dependencies
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/agriscan-project.git
cd agriscan-project
```

### 2. Create virtual environment (recommended)

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1: Run Backend (Flask)

```
python app.py
```

### Step 2: Run Frontend (Streamlit)

```
streamlit run streamlit_app.py
```

---

## 🌐 API Endpoint

* **POST /api/predict**

  * Input: Image file
  * Output: JSON containing:

    * Prediction
    * Confidence
    * Cause
    * Treatment

---

## 📊 Model Details

* Architecture: MobileNetV2 (Transfer Learning)
* Input Size: 224 × 224 × 3
* Output: Multi-class classification (rice diseases)
* Optimization: Adam optimizer
* Loss Function: Categorical Crossentropy

---

## ⚠️ Limitations

* Model accuracy depends on image quality
* May return "Unknown" for unseen conditions
* Requires internet connection when deployed

---

## 🔮 Future Improvements

* Mobile app integration
* Offline model deployment
* More disease classes
* Multilingual support
* Real-time camera input

---

## 👩‍💻 Authors

* Project developed as part of academic coursework

---

## 📄 License

This project is for educational purposes only.
