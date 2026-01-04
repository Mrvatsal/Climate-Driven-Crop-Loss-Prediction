# 🌾 Climate-Driven Crop Loss Prediction

This project predicts **crop loss risk (Low / Medium / High)** using **climate and agricultural data** such as rainfall, temperature, humidity, soil moisture, drought index, and crop yield.

The aim is to help farmers and planners **identify climate-related agricultural risks early** using machine learning.

---

## 📌 Project Overview
Climate change has a major impact on agriculture. Unpredictable rainfall, rising temperatures, and drought conditions often lead to crop loss.

This project uses a **machine learning classification model** to analyze climate factors and predict the **risk level of crop loss**, enabling better planning and preventive action.

---

## 🛠 Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  

---

## 📊 Dataset Description
The dataset contains the following columns:

| Column Name | Description |
|------------|------------|
| rainfall | Annual rainfall (mm) |
| temperature | Average temperature (°C) |
| humidity | Average humidity (%) |
| soil_moisture | Soil moisture index |
| drought_index | Drought severity (0–1) |
| crop_yield | Yield (tons/hectare) |
| loss_risk | Crop loss risk (Low / Medium / High) |

> The dataset is synthetically created but follows realistic climate patterns.

---

## 🤖 Machine Learning Model
- Algorithm: **Random Forest Classifier**
- Problem Type: **Multi-class Classification**
- Output: **Low / Medium / High crop loss risk**

---

## ▶️ How to Run the Project

### 1️⃣ Install required libraries
```bash
pip install -r requirements.txt

2️⃣ Run the machine learning model

python crop_loss_prediction.py

3️⃣ Run the Streamlit web app

streamlit run app.py

