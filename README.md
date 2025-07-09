# Smart Health Care Tracker – Backend

This is the backend for the **Smart Health Care Tracker** project. It uses Flask and XGBoost models to predict the likelihood of different diseases and offers personalized health suggestions.

## 🚀 Features

- Disease predictions for:
  - Cardiovascular Disease
  - Diabetes
  - COPD
  - Depression
  - Kidney Disease
- Tailored health tips based on predictions
- RESTful API endpoints
- CORS enabled for frontend integration

## 🛠️ Tech Stack

- Python
- Flask
- XGBoost
- joblib
- Flask-CORS

## 🧪 API Endpoints

| Endpoint              | Method | Description                    |
|-----------------------|--------|--------------------------------|
| `/predict/cardiovascular` | POST   | Predict cardiovascular disease |
| `/predict/diabetes`       | POST   | Predict diabetes               |
| `/predict/copd`           | POST   | Predict COPD                   |
| `/predict/depression`     | POST   | Predict depression             |
| `/predict/kidney`         | POST   | Predict kidney disease         |
| `/health-suggestions`     | GET    | Get health tips (optional)     |

## 📦 Installation

```bash
git clone https://github.com/arinVashisth/smart-health-care-tracker-backend.git
cd smart-health-care-tracker-backend
pip install -r requirements.txt
