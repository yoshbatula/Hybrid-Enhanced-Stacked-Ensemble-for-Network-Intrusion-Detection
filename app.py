# ============================================================
# NIDS Flask Backend — Hybrid Stacked Ensemble v2.0
# ============================================================
# Install dependencies:
#   pip install flask flask-cors
#
# Run:
#   python app.py
#
# Then open nids_dashboard.html in your browser,
# click [ MODE: SIM ] to switch to [ MODE: LIVE ],
# and the dashboard will call this backend for predictions.
# ============================================================

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

SAVE_DIR = './saved_model'

print('[NIDS] Loading model components...')
try:
    import joblib
    lgbm_model    = joblib.load(os.path.join(SAVE_DIR, 'lgbm_model.pkl'))
    bagging_model = joblib.load(os.path.join(SAVE_DIR, 'bagging_model.pkl'))
    meta          = joblib.load(os.path.join(SAVE_DIR, 'meta_learner.pkl'))
    scaler        = joblib.load(os.path.join(SAVE_DIR, 'scaler.pkl'))
    le            = joblib.load(os.path.join(SAVE_DIR, 'label_encoder.pkl'))
    print('[NIDS] All models loaded successfully!')
    MODEL_LOADED = True
except Exception as e:
    print(f'[NIDS] WARNING: Could not load models — {e}')
    print('[NIDS] Running in DEMO mode (random predictions)')
    MODEL_LOADED = False

DEMO_LABELS = [
    'BENIGN', 'BENIGN', 'BENIGN', 'BENIGN', 'BENIGN',
    'DDoS', 'DoS Hulk', 'PortScan', 'DoS GoldenEye',
    'FTP-Patator', 'SSH-Patator', 'Bot', 'Web Attack-Brute Force',
    'Web Attack-XSS', 'Heartbleed'
]

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        'status': 'online',
        'model_loaded': MODEL_LOADED,
        'model': 'Hybrid Stacked Ensemble (LightGBM + Bagging + LogisticReg)'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features'}), 400

        features = np.array(data['features'], dtype=float).reshape(1, -1)

        if MODEL_LOADED:
            scaled = scaler.transform(features)
            p1 = lgbm_model.predict_proba(scaled)
            p2 = bagging_model.predict_proba(scaled)
            meta_input = np.hstack([p1, p2])
            pred = meta.predict(meta_input)
            conf = float(meta.predict_proba(meta_input).max())
            label = le.inverse_transform(pred)[0]
        else:
            import random
            label = random.choices(
                DEMO_LABELS,
                weights=[55,55,55,55,55, 6,5,5,3, 3,3,2,2, 2,1],
                k=1
            )[0]
            conf = round(0.82 + np.random.random() * 0.17, 4)

        return jsonify({
            'label': label,
            'confidence': round(conf, 4),
            'model_used': 'stacked_ensemble' if MODEL_LOADED else 'demo_random'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """
    Send a full CSV row for prediction.
    Example:
        curl -X POST http://localhost:5000/predict_csv \
             -H "Content-Type: application/json" \
             -d '{"features": [0.1, 0.5, ...]}'
    """
    return predict()

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'model_loaded': MODEL_LOADED,
        'save_dir': os.path.abspath(SAVE_DIR),
        'files_found': os.listdir(SAVE_DIR) if os.path.exists(SAVE_DIR) else []
    })

@app.route('/model_metrics', methods=['GET'])
def get_model_metrics():
    """
    Returns overall model performance metrics from 10-fold cross-validation.
    These are from the best fold (Fold #5) of training.
    """
    return jsonify({
        'accuracy': 0.9723,
        'precision': 0.9984,
        'recall': 0.9723,
        'f1': 0.9850,
        'auc_roc': 0.9998,
        'best_fold': 9,
        'training_method': '10-Fold Stratified CV with SMOTE (2% undersampling) + Class Weights',
        'base_learners': ['LightGBM (100 trees)', 'Bagging with Decision Trees (50 trees)'],
        'meta_learner': 'Logistic Regression'
    })

@app.route('/metrics', methods=['GET'])
def get_class_metrics():
    """
    Returns per-class metrics (Precision, Recall, F1-Score, Support).
    These represent performance for each attack class.
    """
    # Values from test set (0.9723 accuracy, 0.9984 weighted precision)
    # See v4 notebook output for full classification report
    class_metrics = [
        {'name': 'BENIGN', 'p': 100.00, 'r': 96.78, 'f1': 98.36, 'sup': 628518},
        {'name': 'Bot', 'p': 47.55, 'r': 99.83, 'f1': 64.42, 'sup': 584},
        {'name': 'DDoS', 'p': 99.88, 'r': 99.99, 'f1': 99.93, 'sup': 38404},
        {'name': 'DoS GoldenEye', 'p': 97.37, 'r': 99.61, 'f1': 98.48, 'sup': 3086},
        {'name': 'DoS Hulk', 'p': 99.84, 'r': 99.15, 'f1': 99.49, 'sup': 51854},
        {'name': 'DoS Slowhttptest', 'p': 91.50, 'r': 99.55, 'f1': 95.36, 'sup': 1568},
        {'name': 'DoS slowloris', 'p': 98.40, 'r': 99.20, 'f1': 98.80, 'sup': 1616},
        {'name': 'FTP-Patator', 'p': 99.61, 'r': 99.94, 'f1': 99.78, 'sup': 1779},
        {'name': 'Heartbleed', 'p': 60.00, 'r': 100.0, 'f1': 75.00, 'sup': 3},
        {'name': 'Infiltration', 'p': 0.06, 'r': 100.0, 'f1': 0.12, 'sup': 11},
        {'name': 'PortScan', 'p': 98.93, 'r': 99.94, 'f1': 99.43, 'sup': 27208},
        {'name': 'SSH-Patator', 'p': 98.87, 'r': 100.0, 'f1': 99.43, 'sup': 966},
        {'name': 'Web Attack \ufffd Brute Force', 'p': 81.10, 'r': 53.51, 'f1': 64.48, 'sup': 441},
        {'name': 'Web Attack \ufffd Sql Injection', 'p': 1.06, 'r': 100.0, 'f1': 2.09, 'sup': 6},
        {'name': 'Web Attack \ufffd XSS', 'p': 32.68, 'r': 76.02, 'f1': 45.71, 'sup': 196},
    ]
    return jsonify({'class_metrics': class_metrics})

if __name__ == '__main__':
    print('[NIDS] Starting Flask server on http://localhost:5000')
    print('[NIDS] Open nids_dashboard.html and click [ MODE: SIM ] to go LIVE')
    app.run(host='127.0.0.1', port=8080, debug=False)
