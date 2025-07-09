from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import xgboost as xgb
cardio_model = xgb.XGBClassifier()
diabetes_model = xgb.XGBClassifier()
copd_model = xgb.XGBClassifier()
depression_model = xgb.XGBClassifier()
kidney_model = xgb.XGBClassifier()

app = Flask(__name__)
CORS(app)

cardio_model.load_model('models/cardio.xgb')
diabetes_model.load_model('models/diabetes.xgb')
copd_model.load_model('models/copd.xgb')
depression_model.load_model('models/depression.xgb')
kidney_model.load_model('models/kidney.xgb')

booster = kidney_model.get_booster()

raw_feature_names = booster.feature_names

HEALTH_TIPS = {
    "diabetes": {
        1: [
            "Consult your doctor for a personalized plan.",
            "Monitor your blood sugar regularly.",
            "Adopt a balanced, low-sugar diet.",
            "Increase physical activity."
        ],
        0: [
            "Maintain your healthy lifestyle.",
            "Continue regular checkups.",
            "Stay active and eat a balanced diet."
        ]
    },
    "cardiovascular": {
        1: [
            "Consult your cardiologist for tailored advice.",
            "Monitor your blood pressure regularly.",
            "Reduce salt and saturated fat intake.",
            "Exercise regularly, as advised by your doctor.",
            "Quit smoking and limit alcohol consumption."
        ],
        0: [
            "Maintain a heart-healthy diet.",
            "Continue regular physical activity.",
            "Monitor your blood pressure and cholesterol."
        ]
    },
    "copd": {
        1: [
            "Consult your pulmonologist for a treatment plan.",
            "Avoid exposure to smoke and air pollutants.",
            "Follow prescribed inhaler or medication routines.",
            "Practice breathing exercises.",
            "Get vaccinated against flu and pneumonia."
        ],
        0: [
            "Continue avoiding tobacco smoke.",
            "Stay active with regular, gentle exercise.",
            "Monitor your respiratory health."
        ]
    },
    "depression": {
        1: [
            "Reach out to a mental health professional.",
            "Talk to trusted friends or family.",
            "Maintain a daily routine and self-care.",
            "Engage in regular physical activity.",
            "Avoid isolation; seek support groups."
        ],
        0: [
            "Continue healthy coping strategies.",
            "Maintain supportive social connections.",
            "Monitor your mood and well-being."
        ]
    },
    "kidney": {
        1: [
            "Consult your nephrologist for a treatment plan.",
            "Monitor your blood pressure and blood sugar.",
            "Limit salt and protein intake as advised.",
            "Stay hydrated (but follow your doctor's advice).",
            "Avoid over-the-counter painkillers unless prescribed."
        ],
        0: [
            "Maintain a kidney-friendly diet.",
            "Monitor your blood pressure.",
            "Stay hydrated and active."
        ]
    }
}

def get_tips(disease, risk):
    tips_dict = HEALTH_TIPS.get(disease, {})
    return tips_dict.get(risk, [])

@app.route('/predict/cardiovascular', methods=['POST'])
def predict_cardiovascular():
    data = request.json
    def to_float(val, default=0.0):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def to_int(val, default=0):
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    features = [
        to_float(data.get('age_years')),
        to_float(data.get('ap_hi')),
        to_float(data.get('ap_lo')),
        to_int(data.get('cholesterol')),
        to_int(data.get('gluc')),
        to_int(data.get('active')),
        to_int(data.get('smoke')),
        to_int(data.get('alco')),
    ]

    prediction = int(cardio_model.predict([features])[0])

    return jsonify({
        "prediction": prediction,
        "tips": get_tips("cardiovascular",prediction),
        "disease": "Cardiovascular Disease"
    })

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    data = request.json
    height_cm = float(data.get('height', 0))
    weight_kg = float(data.get('weight', 0))
    height_m = height_cm / 100 if height_cm > 10 else height_cm

    bmi = weight_kg / (height_m ** 2) if height_m > 0 else 0

    features = [
        float(data.get('age', 0)),
        1 if data.get('sex', '').lower() == 'male' else 0,
        bmi,
        1 if data.get('highBP', '').lower() == 'yes' else 0,
        1 if data.get('highChol', '').lower() == 'yes' else 0,
        1 if data.get('smoker', '').lower() == 'yes' else 0,
        1 if data.get('exercise', '').lower() == 'yes' else 0,
        1 if data.get('fruits', '').lower() == 'yes' else 0,
        1 if data.get('veggies', '').lower() == 'yes' else 0,
        {'excellent': 5, 'very good': 4, 'good': 3, 'fair': 2, 'poor': 1}.get(data.get('genHealth', '').lower(), 3),
        {
            'less than high school': 1,
            'high school': 2,
            'some college': 3,
            'college grad': 4,
            'post-grad': 5
        }.get(data.get('education', '').lower(), 3),
        {
            'low': 1,
            'medium': 2,
            'high': 3
        }.get(data.get('income', '').lower(), 2),
    ]
    prediction = int(diabetes_model.predict([features])[0])
    
    return jsonify({
        "prediction": prediction,
        "tips": get_tips("diabetes", prediction),
        "disease": "Diabetes"
    })

@app.route('/predict/copd', methods=['POST'])
def predict_copd():
    data = request.json

    try:
        weight = float(data.get('weight', 0))
        height = float(data.get('height', 0))
        bmi = weight / (height ** 2) if height > 0 else 0
    except Exception:
        bmi = 0

    features = [
        float(data.get('age', 0)),
        int(data.get('gender', 0)),
        bmi,
        float(data.get('height', 0)),
        int(data.get('heartFailure', False)),
        int(data.get('workingPlace', 0)),
        int(data.get('mMRC', 0)),
        int(data.get('smokingStatus', 0)),
        float(data.get('packHistory', 0)),
        int(data.get('vaccination', False)),
        int(data.get('depression', False)),
        int(data.get('dependent', False)),
        int(data.get('temperature', 0)),
        float(data.get('respiratoryRate', 0)),
        int(data.get('heartRate', 0)),
        int(data.get('bloodPressure', 0)),
        float(data.get('oxygenSaturation', 0)),
        int(data.get('sputum', 0)),
        int(data.get('fev1', 0)),
    ]
    prediction = int(copd_model.predict([features])[0])
    return jsonify({
        "prediction": prediction,
        "tips": get_tips("copd",prediction),
        "disease": "COPD"
    })

@app.route('/predict/depression', methods=['POST'])
def predict_depression():
    data = request.json
    features = [
        float(data.get('age', 0)),
        1 if data.get('gender', '').lower() == 'male' else 0,

        1 if data.get('self_employed', '').lower() == 'yes' else 0,

        1 if data.get('family_history', '').lower() == 'yes' else 0,

        {'never': 0, 'rarely': 1, 'sometimes': 2, 'often': 3}.get(data.get('work_interfere', '').lower(), 0),

        1 if data.get('remote_work', '').lower() == 'yes' else 0,

        {'yes': 1, 'no': 0, "don't know": 2}.get(data.get('benefits', '').lower(), 2),

        {'yes': 1, 'no': 0, 'not sure': 2}.get(data.get('care_options', '').lower(), 2),

        {'yes': 1, 'no': 0, "don't know": 2}.get(data.get('wellness_program', '').lower(), 2),

        {'yes': 1, 'no': 0, "don't know": 2}.get(data.get('seek_help', '').lower(), 2),

        {'yes': 1, 'no': 0, "don't know": 2}.get(data.get('anonymity', '').lower(), 2),

        {
            'very easy': 4,
            'somewhat easy': 3,
            "don't know": 2,
            'somewhat difficult': 1,
            'very difficult': 0
        }.get(data.get('leave', '').lower(), 2),

        {'yes': 1, 'no': 0, 'maybe': 2}.get(data.get('mental_health_consequence', '').lower(), 2),

        {'yes': 1, 'no': 0, 'some of them': 2}.get(data.get('supervisor', '').lower(), 2),

        {'yes': 1, 'no': 0, "don't know": 2}.get(data.get('mental_vs_physical', '').lower(), 2),
    ]
    prediction = int(depression_model.predict([features])[0])
    return jsonify({
        "prediction": prediction,
        "tips": get_tips("depression",prediction),
        "disease": "Depression"
    })

@app.route('/predict/kidney', methods=['POST'])
def predict_kidney():
    data = request.json
    features = [
        float(data.get('hemo', 0)),
        float(data.get('sg', 0)),
        float(data.get('al', 0)),
        1 if data.get('htn', '').lower() == 'yes' else 0,
        float(data.get('sc', 0)),
        float(data.get('sod', 0)),
        float(data.get('bgr', 0)),
        float(data.get('pcv', 0)),
        float(data.get('bu', 0)),
        float(data.get('age', 0))
    ]
    prediction = int(kidney_model.predict([features])[0])
    return jsonify({
        "prediction": prediction,
        "tips": get_tips("kidney",prediction),
        "disease": "Kidney Disease"
    })


@app.route('/health-suggestions', methods=['GET'])
def health_suggestions():
    disease = request.args.get('disease', 'general')
    return jsonify({"tips": get_tips(disease)})



if __name__ == '__main__':
    app.run(port=5000,debug=True)

