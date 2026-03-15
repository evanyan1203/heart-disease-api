# Heart Disease Prediction API

A machine learning API that predicts the risk of heart disease using clinical health data.

The model is trained using Scikit-learn and deployed as a REST API using FastAPI.

## 🚀 Live Demo

API Base URL:

https://heart-disease-api-n225.onrender.com

API Docs:

https://heart-disease-api-n225.onrender.com/docs

## 📊 Machine Learning Model

Model: Logistic Regression  
Dataset: Heart Disease Dataset  
Features:

- age
- sex
- chest pain type (cp)
- resting blood pressure (trestbps)
- cholesterol (chol)
- fasting blood sugar (fbs)
- resting ECG (restecg)
- maximum heart rate (thalach)
- exercise induced angina (exang)
- oldpeak
- slope
- ca
- thal

## 🧠 Model Performance

Accuracy: **0.81**

ROC-AUC: **0.93**

Confusion Matrix:

[[70 30]
[ 9 96]]


## 🛠 Tech Stack

Python  
FastAPI  
Scikit-learn  
Pandas  
NumPy  
Uvicorn  
Render (Cloud Deployment)

## 📡 API Usage

### Predict Endpoint

POST /predict

Predicts the probability of heart disease based on clinical features.

##Example Request
```
{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 2,
  "thal": 3
}
```
Example Response
```
{
  "prediction": 0,
  "probability": 0.0499,
  "message": "Low risk"
}

```

Where:

prediction = 1 → High risk of heart disease

prediction = 0 → Low risk of heart disease

probability = model confidence score

## Project Structure

```
heart-disease-api
│
├── app.py            # FastAPI application (prediction API)
├── train_model.py    # machine learning training script
├── heart_model.pkl   # trained ML model
├── heart.csv         # dataset
└── requirements.txt  # Python dependencies
```

Run Locally

1. Install dependencies
pip install -r requirements.txt

2. Start the API server
uvicorn app:app --reload

3. Open API documentation
http://127.0.0.1:8000/docs

Swagger UI allows you to test the prediction endpoint directly.

Deployment

The API is deployed on Render.

Live API documentation:

https://heart-disease-api-n225.onrender.com/docs

