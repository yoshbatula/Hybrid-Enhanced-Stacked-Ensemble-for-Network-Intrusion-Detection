import os, sys, warnings, gc
import numpy as np
import pandas as pd
import joblib

SAVE_DIR = './saved_model'
PARQUET  = 'NF-BoT-IoT-V2.parquet'
SAMPLE   = 10000  # number of rows to test

# Load saved model
print('Loading saved model...')
lgbm    = joblib.load(os.path.join(SAVE_DIR, 'lgbm_model.pkl'))
bagging = joblib.load(os.path.join(SAVE_DIR, 'bagging_model.pkl'))
meta    = joblib.load(os.path.join(SAVE_DIR, 'meta_learner.pkl'))
scaler  = joblib.load(os.path.join(SAVE_DIR, 'scaler.pkl'))
le      = joblib.load(os.path.join(SAVE_DIR, 'label_encoder.pkl'))

# Load NF-BoT-IoT-V2
print(f'Loading {PARQUET}...')
df = pd.read_parquet(PARQUET)
df = df.head(SAMPLE).copy()

# Separate features and labels
y_true_raw = df['Attack'].values  # 'DoS','DDoS','Reconnaissance','Benign','Theft'
features_41 = df.drop(columns=['Label', 'Attack']).values.astype(float)

print(f'BoT-IoT samples: {len(df)}')
print(f'Attack distribution: {pd.Series(y_true_raw).value_counts().to_dict()}')

# Map 41 BoT-IoT features -> 78 CICIDS features
# Index mapping: CICIDS[i] = BoT-IoT[j] or computed
n = len(df)
X_78 = np.zeros((n, 78))

# Feature mapping (best-effort)
# 0: Destination Port -> L4_DST_PORT [1]
X_78[:, 0]  = features_41[:, 1]
# 1: Flow Duration -> FLOW_DURATION_MILLISECONDS [11]
X_78[:, 1]  = features_41[:, 11]
# 2: Total Fwd Packets -> OUT_PKTS [7]
X_78[:, 2]  = features_41[:, 7]
# 3: Total Backward Packets -> IN_PKTS [5]
X_78[:, 3]  = features_41[:, 5]
# 4: Total Length of Fwd Packets -> OUT_BYTES [6]
X_78[:, 4]  = features_41[:, 6]
# 5: Total Length of Bwd Packets -> IN_BYTES [4]
X_78[:, 5]  = features_41[:, 4]
# 6: Fwd Packet Length Max -> MAX_IP_PKT_LEN [19]
X_78[:, 6]  = features_41[:, 19]
# 7: Fwd Packet Length Min -> MIN_IP_PKT_LEN [18]
X_78[:, 7]  = features_41[:, 18]
# 8: Fwd Packet Length Mean
fwd_mask = features_41[:, 7] > 0
X_78[fwd_mask, 8] = features_41[fwd_mask, 6] / features_41[fwd_mask, 7]
# 9: Fwd Packet Length Std -> 0
# 10: Bwd Packet Length Max -> MAX_IP_PKT_LEN [19]
X_78[:, 10] = features_41[:, 19]
# 11: Bwd Packet Length Min -> MIN_IP_PKT_LEN [18]
X_78[:, 11] = features_41[:, 18]
# 12: Bwd Packet Length Mean
bwd_mask = features_41[:, 5] > 0
X_78[bwd_mask, 12] = features_41[bwd_mask, 4] / features_41[bwd_mask, 5]
# 13: Bwd Packet Length Std -> 0
# 14: Flow Bytes/s -> SRC_TO_DST_SECOND_BYTES [20]
X_78[:, 14] = features_41[:, 20]
# 15: Flow Packets/s -> compute
dur = features_41[:, 11] / 1000.0  # milliseconds to seconds
dur_mask = dur > 0
total_pkts = features_41[:, 5] + features_41[:, 7]
X_78[dur_mask, 15] = total_pkts[dur_mask] / dur[dur_mask]
# 16-19: Flow IAT -> DURATION_IN [12] / OUT [13] as proxy
X_78[:, 16] = features_41[:, 12]  # Flow IAT Mean
X_78[:, 17] = features_41[:, 13]  # Flow IAT Std
X_78[:, 18] = np.maximum(features_41[:, 12], features_41[:, 13])  # Flow IAT Max
X_78[:, 19] = np.minimum(features_41[:, 12], features_41[:, 13])  # Flow IAT Min
# 20-25: Fwd/Bwd IAT -> use DURATION_IN/OUT
X_78[:, 20] = features_41[:, 12]  # Fwd IAT Total
X_78[:, 21] = features_41[:, 12]  # Fwd IAT Mean
X_78[:, 22] = features_41[:, 13]  # Fwd IAT Std
X_78[:, 23] = features_41[:, 12]  # Fwd IAT Max
X_78[:, 24] = features_41[:, 13]  # Fwd IAT Min
X_78[:, 25] = features_41[:, 13]  # Bwd IAT Total
X_78[:, 26] = features_41[:, 13]  # Bwd IAT Mean
X_78[:, 27] = features_41[:, 12]  # Bwd IAT Std
X_78[:, 28] = features_41[:, 12]  # Bwd IAT Max
X_78[:, 29] = features_41[:, 13]  # Bwd IAT Min
# 30-33: PSH/URG flags -> from TCP_FLAGS [8]
tcp = features_41[:, 8].astype(int)
X_78[:, 30] = (tcp & 8) >> 3   # Fwd PSH
X_78[:, 31] = (tcp & 8) >> 3   # Bwd PSH
X_78[:, 32] = (tcp & 32) >> 5  # Fwd URG
X_78[:, 33] = (tcp & 32) >> 5  # Bwd URG
# 34: Fwd Header Length -> 40 (default TCP)
X_78[:, 34] = 40
# 35: Bwd Header Length -> 40
X_78[:, 35] = 40
# 36: Fwd Packets/s
X_78[dur_mask, 36] = features_41[dur_mask, 7] / dur[dur_mask]
# 37: Bwd Packets/s
X_78[dur_mask, 37] = features_41[dur_mask, 5] / dur[dur_mask]
# 38: Min Packet Length
X_78[:, 38] = features_41[:, 18]
# 39: Max Packet Length
X_78[:, 39] = features_41[:, 19]
# 40: Packet Length Mean
pk_mask = total_pkts > 0
total_bytes = features_41[:, 4] + features_41[:, 6]
X_78[pk_mask, 40] = total_bytes[pk_mask] / total_pkts[pk_mask]
# 41: Packet Length Std -> 0
# 42: Packet Length Variance -> 0
# 43-50: Flag counts from TCP_FLAGS [8]
X_78[:, 43] = (tcp & 1) >> 0    # FIN
X_78[:, 44] = (tcp & 2) >> 1    # SYN
X_78[:, 45] = (tcp & 4) >> 2    # RST
X_78[:, 46] = (tcp & 8) >> 3    # PSH
X_78[:, 47] = (tcp & 16) >> 4   # ACK
X_78[:, 48] = (tcp & 32) >> 5   # URG
# 49: CWE -> 0
# 50: ECE -> 0
# 51: Down/Up Ratio
up_mask = features_41[:, 6] > 0
X_78[up_mask, 51] = features_41[up_mask, 4] / features_41[up_mask, 6]
# 52: Average Packet Size
X_78[pk_mask, 52] = total_bytes[pk_mask] / total_pkts[pk_mask]
# 53: Avg Fwd Segment Size
X_78[fwd_mask, 53] = features_41[fwd_mask, 6] / features_41[fwd_mask, 7]
# 54: Avg Bwd Segment Size
X_78[bwd_mask, 54] = features_41[bwd_mask, 4] / features_41[bwd_mask, 5]
# 55: Fwd Header Length -> 40
X_78[:, 55] = 40
# 56-61: Bulk -> 0
# 62: Subflow Fwd Packets -> OUT_PKTS [7]
X_78[:, 62] = features_41[:, 7]
# 63: Subflow Fwd Bytes -> OUT_BYTES [6]
X_78[:, 63] = features_41[:, 6]
# 64: Subflow Bwd Packets -> IN_PKTS [5]
X_78[:, 64] = features_41[:, 5]
# 65: Subflow Bwd Bytes -> IN_BYTES [4]
X_78[:, 65] = features_41[:, 4]
# 66: Init_Win_bytes_forward -> TCP_WIN_MAX_OUT [34]
X_78[:, 66] = features_41[:, 34]
# 67: Init_Win_bytes_backward -> TCP_WIN_MAX_IN [33]
X_78[:, 67] = features_41[:, 33]
# 68: act_data_pkt_fwd -> 0
# 69: min_seg_size_forward -> MIN_IP_PKT_LEN [18]
X_78[:, 69] = features_41[:, 18]
# 70-77: Active/Idle -> 0

# Replace inf/nan
X_78 = np.nan_to_num(X_78, nan=0.0, posinf=0.0, neginf=0.0)

# Scale
print('Scaling and predicting...')
X_scaled = scaler.transform(X_78)

# Base predictions
p1 = lgbm.predict_proba(X_scaled)
p2 = bagging.predict_proba(X_scaled)
M  = np.hstack([p1, p2])

# Meta predictions
preds = meta.predict(M)
confs = meta.predict_proba(M).max(axis=1)
labels_pred = le.inverse_transform(preds)

# Results
print('\n===== RESULTS: Saved Model on NF-BoT-IoT =====\n')

results = pd.DataFrame({
    'true_attack': y_true_raw,
    'predicted': labels_pred,
    'confidence': confs
})

# Map BoT-IoT attack types to 2-class (0=Benign, 1=Attack)
y_binary = np.where(y_true_raw == 'Benign', 0, 1)
pred_binary = np.where(labels_pred == 'BENIGN', 0, 1)

tp = np.sum((pred_binary == 1) & (y_binary == 1))
tn = np.sum((pred_binary == 0) & (y_binary == 0))
fp = np.sum((pred_binary == 1) & (y_binary == 0))
fn = np.sum((pred_binary == 0) & (y_binary == 1))

print(f'Total samples: {len(results)}')
print(f'TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}')
prec = tp / (tp + fp) if (tp + fp) > 0 else 0
rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
acc  = (tp + tn) / len(results)
print(f'Accuracy:  {acc:.4f}')
print(f'Precision: {prec:.4f}')
print(f'Recall:    {rec:.4f}')
print(f'F1-Score:  {f1:.4f}')

print('\n--- Per true attack type ---')
for atk in ['Benign', 'DoS', 'DDoS', 'Reconnaissance', 'Theft']:
    mask = y_true_raw == atk
    if mask.sum() == 0:
        continue
    subset = results[mask]
    top = subset['predicted'].value_counts().head(3)
    avg_conf = subset['confidence'].mean()
    print(f'\n{atk} ({mask.sum()} samples):')
    for lbl, cnt in top.items():
        print(f'  predicted={lbl}: {cnt} ({cnt/mask.sum()*100:.1f}%)')
    print(f'  avg confidence: {avg_conf:.4f}')

# Save detailed results
results.to_csv('bot_iot_test_results.csv', index=False)
print('\nDetailed results saved to bot_iot_test_results.csv')
