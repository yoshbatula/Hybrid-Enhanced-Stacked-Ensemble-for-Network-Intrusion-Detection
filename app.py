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

if __name__ == '__main__':
    print('[NIDS] Starting Flask server on http://localhost:5000')
    print('[NIDS] Open nids_dashboard.html and click [ MODE: SIM ] to go LIVE')
    app.run(host='127.0.0.1', port=8080, debug=False)
