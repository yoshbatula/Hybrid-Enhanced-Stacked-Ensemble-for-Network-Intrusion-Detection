"""
NIDS Predictor — Single-sample classification UI
Input numeric feature values → get instant attack classification
"""

import sys, os, glob, gc, re
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight

st.set_page_config(page_title="NIDS Predictor", page_icon="🛡️", layout="centered")

st.markdown("""
    <div style='text-align:center; padding:1rem'>
        <h1>🛡️ NIDS Predictor</h1>
        <p style='color:#888'>Hybrid Enhanced Stacked Ensemble — LightGBM + Bagging/DT</p>
    </div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────
# Train model (cached)
# ──────────────────────────────────────
@st.cache_resource
def train_model():
    with st.spinner("Loading CICIDS 2017 data..."):
        CSV_FOLDER = './MachineLearningCVE'
        csv_files = sorted(glob.glob(os.path.join(CSV_FOLDER, '*.csv')))
        dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        del dfs; gc.collect()

        df.columns = df.columns.str.strip()
        drop_cols = [c for c in ['Flow ID','Source IP','Destination IP','Timestamp',
                                  'Src IP','Dst IP','src_ip','dst_ip'] if c in df.columns]
        if drop_cols: df.drop(columns=drop_cols, inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(axis=1, thresh=len(df)*0.5, inplace=True)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols: df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df.drop_duplicates(inplace=True)

        le = LabelEncoder()
        df['label_enc'] = le.fit_transform(df['Label'].astype(str))
        class_names = le.classes_
        n_classes = len(class_names)
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c != 'label_enc']

        X = df[feature_cols]
        y = df['label_enc']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        _ = scaler.transform(X_test)
        del df; gc.collect()

    with st.spinner("Training ensemble (5-fold CV)..."):
        N_FOLDS = 5
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        lgbm = lgb.LGBMClassifier(
            objective='multiclass', metric='multi_logloss', num_class=n_classes,
            n_estimators=100, learning_rate=0.05, max_depth=8, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, class_weight='balanced',
            device='cpu', random_state=42, n_jobs=2, verbose=-1)
        bag = BaggingClassifier(
            estimator=DecisionTreeClassifier(criterion='gini', max_depth=10,
                        min_samples_split=20, min_samples_leaf=10, random_state=42),
            n_estimators=100, max_samples=0.8, max_features=0.5,
            bootstrap=True, n_jobs=2, random_state=42)

        oof_lgbm = np.zeros((len(X_train), n_classes))
        oof_bag = np.zeros((len(X_train), n_classes))
        ya = y_train.values
        for tr, val in skf.split(X_train, ya):
            lgbm.fit(X_train[tr], ya[tr], eval_set=[(X_train[tr], ya[tr]), (X_train[val], ya[val])],
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            oof_lgbm[val] = lgbm.predict_proba(X_train[val])
            sw = compute_sample_weight(class_weight='balanced', y=ya[tr])
            bag.fit(X_train[tr], ya[tr], sample_weight=sw)
            oof_bag[val] = bag.predict_proba(X_train[val])
            gc.collect()

        M_train = np.hstack([oof_lgbm, oof_bag])
        meta = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced',
                                   solver='lbfgs', n_jobs=-1)
        meta.fit(M_train, ya)

    return scaler, feature_cols, class_names, lgbm, bag, meta

scaler, feature_cols, class_names, lgbm_model, bag_model, meta_learner = train_model()
n_features = len(feature_cols)

# ──────────────────────────────────────
# UI: Input
# ──────────────────────────────────────
st.divider()
st.markdown("### Enter Network Flow Features")

input_method = st.radio("Input method:", ["Paste CSV line", "Manual entry"], horizontal=True)

values = None

if input_method == "Paste CSV line":
    csv_text = st.text_area(
        f"Paste {n_features} comma-separated numeric values:",
        height=100,
        placeholder="e.g. 0.0, 0.0, 0.0, 1.0, 0.0, ..."
    )
    if csv_text:
        try:
            nums = [float(x.strip()) for x in csv_text.replace('\n', ',').split(',') if x.strip()]
            if len(nums) == n_features:
                values = np.array(nums).reshape(1, -1)
            else:
                st.warning(f"Expected {n_features} values, got {len(nums)}")
        except:
            st.error("Invalid numbers — check format")

else:
    with st.expander(f"Enter {n_features} feature values", expanded=True):
        cols_per_row = 4
        rows = (n_features + cols_per_row - 1) // cols_per_row
        manual_values = []
        for r in range(rows):
            cols = st.columns(cols_per_row)
            for c in range(cols_per_row):
                idx = r * cols_per_row + c
                if idx < n_features:
                    short = feature_cols[idx][:20]
                    v = cols[c].number_input(short, value=0.0, format="%.6f", key=f"f{idx}")
                    manual_values.append(v)
        values = np.array(manual_values).reshape(1, -1)

# ──────────────────────────────────────
# Predict
# ──────────────────────────────────────
st.divider()
col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
with col_b2:
    predict_click = st.button("🔍 Predict", type="primary", use_container_width=True)

if predict_click and values is not None:
    # Scale
    X_scaled = scaler.transform(values)

    # Get meta-features
    p1 = lgbm_model.predict_proba(X_scaled)
    p2 = bag_model.predict_proba(X_scaled)
    M = np.hstack([p1, p2])

    pred = meta_learner.predict(M)[0]
    probs = meta_learner.predict_proba(M)[0]

    pred_name = class_names[pred]
    confidence = probs[pred]

    # Top-3
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(class_names[i], probs[i]) for i in top3_idx]

    # Result
    is_attack = pred != 0
    color = "#e74c3c" if is_attack else "#2ecc71"
    icon = "🚨" if is_attack else "✅"
    label = f"{icon} {pred_name}"
    sub = f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)"

    st.markdown(f"""
        <div style='text-align:center; padding:2rem; background:{color}22;
                    border-radius:1rem; border:2px solid {color}'>
            <h2 style='color:{color}'>{label}</h2>
            <p style='font-size:1.2rem'>{sub}</p>
        </div>
    """, unsafe_allow_html=True)

    # Confidence bar
    st.markdown("#### Confidence")
    st.progress(float(confidence))

    # Top-3
    st.markdown("#### Top-3 Predictions")
    for i, (name, prob) in enumerate(top3):
        c = "#e74c3c" if name != class_names[0] else "#2ecc71"
        st.markdown(f"""
            <div style='display:flex; justify-content:space-between; padding:0.3rem 0'>
                <span><b>{i+1}.</b> <span style='color:{c}'>{name}</span></span>
                <span>{prob:.4f} ({prob*100:.1f}%)</span>
            </div>
        """, unsafe_allow_html=True)
        st.progress(float(prob))

elif predict_click:
    st.warning("Please enter valid feature values first.")
