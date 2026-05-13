"""
Real-Time NIDS Dashboard — Streamlit UI
Interactive dashboard for the Hybrid Enhanced Stacked Ensemble model.
Simulates live network traffic with per-flow detection visualization.
"""

import sys, os, glob, gc, time, json
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import lightgbm as lgb
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight

st.set_page_config(
    page_title="NIDS Real-Time Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────────
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.flow_buffer = deque(maxlen=100)
    st.session_state.alert_buffer = deque(maxlen=50)
    st.session_state.latency_history = deque(maxlen=500)
    st.session_state.flow_count = 0
    st.session_state.alert_count = 0
    st.session_state.correct_alerts = 0
    st.session_state.false_positives = 0
    st.session_state.false_negatives = 0
    st.session_state.running = False
    st.session_state.model_ready = False
    st.session_state.simulation_complete = False

# ──────────────────────────────────────────────
# Title
# ──────────────────────────────────────────────
st.markdown("""
    <div style='text-align:center; padding:1rem'>
        <h1>🛡️ Hybrid Enhanced Stacked Ensemble — NIDS Simulator</h1>
        <p style='color:#888'>Real-time network intrusion detection with LightGBM + Bagging/DT stacking</p>
    </div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Phase 1: Train Model
# ──────────────────────────────────────────────
if not st.session_state.model_ready:
    st.info("### Phase 1: Training the Stacked Ensemble Model")
    st.markdown("This runs once. The model will be cached in session state.")

    with st.spinner("Loading and preprocessing CICIDS 2017 data..."):
        CSV_FOLDER = './MachineLearningCVE'
        csv_files = sorted(glob.glob(os.path.join(CSV_FOLDER, '*.csv')))
        dfs = []
        for f in csv_files:
            dfs.append(pd.read_csv(f, low_memory=False))
        df = pd.concat(dfs, ignore_index=True)
        del dfs; gc.collect()

        df.columns = df.columns.str.strip()
        drop_cols = [c for c in ['Flow ID','Source IP','Destination IP','Timestamp',
                                  'Src IP','Dst IP','src_ip','dst_ip'] if c in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(axis=1, thresh=len(df)*0.5, inplace=True)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df.drop_duplicates(inplace=True)

        le = LabelEncoder()
        df['label_enc'] = le.fit_transform(df['Label'].astype(str))
        class_names = le.classes_
        n_classes = len(class_names)

        X = df.drop(columns=['Label', 'label_enc']).select_dtypes(include=[np.number])
        y = df['label_enc']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        scaler = MinMaxScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test  = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
        del df; gc.collect()

        st.success(f"✅ Data loaded: {len(X_test):,} test samples, {n_classes} classes")

    # OOF predictions and meta-learner
    with st.spinner("Training base learners with 5-fold CV..."):
        N_FOLDS = 5
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

        lgbm_model = lgb.LGBMClassifier(
            objective='multiclass', metric='multi_logloss', num_class=n_classes,
            n_estimators=100, learning_rate=0.05, max_depth=8, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, class_weight='balanced',
            device='cpu', random_state=42, n_jobs=2, verbose=-1)

        bagging_model = BaggingClassifier(
            estimator=DecisionTreeClassifier(criterion='gini', max_depth=10,
                        min_samples_split=20, min_samples_leaf=10, random_state=42),
            n_estimators=100, max_samples=0.8, max_features=0.5,
            bootstrap=True, n_jobs=2, random_state=42)

        oof_lgbm = np.zeros((len(X_train), n_classes))
        oof_bagging = np.zeros((len(X_train), n_classes))
        test_lgbm_accum = np.zeros((len(X_test), n_classes))
        test_bagging_accum = np.zeros((len(X_test), n_classes))
        Xa, ya = X_train.values, y_train.values
        Xta = X_test.values

        for fold, (tr_idx, val_idx) in enumerate(skf.split(Xa, ya), 1):
            Xtr, Xval = Xa[tr_idx], Xa[val_idx]
            ytr, yval = ya[tr_idx], ya[val_idx]
            lgbm_model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xval, yval)],
                           eval_names=['train','eval'],
                           callbacks=[lgb.early_stopping(50, verbose=False),
                                      lgb.log_evaluation(-1)])
            oof_lgbm[val_idx] = lgbm_model.predict_proba(Xval)
            test_lgbm_accum += lgbm_model.predict_proba(Xta)
            sw = compute_sample_weight(class_weight='balanced', y=ytr)
            bagging_model.fit(Xtr, ytr, sample_weight=sw)
            oof_bagging[val_idx] = bagging_model.predict_proba(Xval)
            test_bagging_accum += bagging_model.predict_proba(Xta)
            gc.collect()

        test_lgbm_avg = test_lgbm_accum / N_FOLDS
        test_bagging_avg = test_bagging_accum / N_FOLDS
        M_train = np.hstack([oof_lgbm, oof_bagging])
        M_test  = np.hstack([test_lgbm_avg, test_bagging_avg])

        meta = LogisticRegression(max_iter=1000, random_state=42,
                                   class_weight='balanced', solver='lbfgs', n_jobs=-1)
        meta.fit(M_train, ya)

        st.success("✅ Model trained!")

    # Store in session state
    st.session_state.meta = meta
    st.session_state.M_test = M_test
    st.session_state.y_test = y_test.values
    st.session_state.class_names = class_names
    st.session_state.n_classes = n_classes
    st.session_state.X_test = X_test
    st.session_state.model_ready = True
    st.rerun()

# ──────────────────────────────────────────────
# Phase 2: Simulation Dashboard
# ──────────────────────────────────────────────
else:
    meta = st.session_state.meta
    M_test = st.session_state.M_test
    y_test_arr = st.session_state.y_test
    class_names = st.session_state.class_names
    n_classes = st.session_state.n_classes
    n_test = len(M_test)

    # ── Simulation config ──
    col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
    with col_cfg1:
        num_flows = st.number_input("Flows to simulate", min_value=10, max_value=min(2000, n_test),
                                     value=min(200, n_test), step=10, key='num_flows')
    with col_cfg2:
        batch_size = st.selectbox("Batch size", [1, 5, 10, 20, 50], index=0, key='batch_size')
    with col_cfg3:
        update_interval = st.slider("Update interval (s)", 0.01, 0.5, 0.05, key='interval')
    with col_cfg4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶️ Start Simulation", type="primary", use_container_width=True):
            st.session_state.running = True
            st.session_state.simulation_complete = False
            # Reset
            st.session_state.flow_buffer.clear()
            st.session_state.alert_buffer.clear()
            st.session_state.latency_history.clear()
            st.session_state.flow_count = 0
            st.session_state.alert_count = 0
            st.session_state.correct_alerts = 0
            st.session_state.false_positives = 0
            st.session_state.false_negatives = 0
            st.rerun()

    if st.session_state.running:
        st.divider()
        st.markdown("### 📡 Live Detection Feed")

        # ── Metrics row ──
        kpi_cols = st.columns(5)
        flow_placeholder = st.empty()

        # ── Simulation loop ──
        total = min(num_flows, n_test)
        start_global = time.time()
        alert_counts_per_class = {name: 0 for name in class_names}
        per_flow_latencies = []
        feed_placeholder = st.empty()
        alert_placeholder = st.empty()

        # Progress
        progress_bar = st.progress(0)

        for idx in range(0, total, batch_size):
            end_idx = min(idx + batch_size, total)
            batch = M_test[idx:end_idx]
            y_true_batch = y_test_arr[idx:end_idx]

            t0 = time.perf_counter()
            preds = meta.predict(batch)
            probs = meta.predict_proba(batch)
            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000
            per_flow_latencies.extend([latency_ms / len(batch)] * len(batch))

            for i in range(len(batch)):
                flow_num = idx + i + 1
                pred_name = class_names[preds[i]]
                true_name = class_names[y_true_batch[i]]
                confidence = probs[i].max()
                is_correct = preds[i] == y_true_batch[i]

                st.session_state.flow_count += 1
                per_flow_lat = (t1 - t0) * 1000 / len(batch)
                st.session_state.latency_history.append(per_flow_lat)

                flow_entry = {
                    'flow': flow_num, 'time': datetime.now().strftime('%H:%M:%S.%f')[:-3],
                    'predicted': pred_name, 'actual': true_name,
                    'confidence': f'{confidence:.3f}',
                    'result': '🟢' if pred_name == 'BENIGN' else ('🔴 CORRECT' if is_correct else '⚠️ FP'),
                    'latency': f'{per_flow_lat*1000:.0f}us'
                }
                st.session_state.flow_buffer.append(flow_entry)

                # Alert tracking
                if preds[i] != 0:
                    st.session_state.alert_count += 1
                    alert_counts_per_class[pred_name] += 1
                    st.session_state.alert_buffer.append(flow_entry)
                    if is_correct:
                        st.session_state.correct_alerts += 1
                    elif y_true_batch[i] == 0:
                        st.session_state.false_positives += 1
                elif y_true_batch[i] != 0:
                    st.session_state.false_negatives += 1

            # ── Update KPI metrics ──
            elapsed = time.time() - start_global
            throughput = st.session_state.flow_count / elapsed if elapsed > 0 else 0
            avg_lat = np.mean(list(st.session_state.latency_history)) * 1000 if st.session_state.latency_history else 0

            with kpi_cols[0]:
                st.metric("📊 Flows", st.session_state.flow_count, f"{throughput:.0f}/s")
            with kpi_cols[1]:
                st.metric("🚨 Alerts", st.session_state.alert_count)
            with kpi_cols[2]:
                st.metric("⏱️ Avg Latency", f"{avg_lat:.0f}μs")
            with kpi_cols[3]:
                alarm_acc = st.session_state.correct_alerts / max(st.session_state.alert_count, 1)
                st.metric("🎯 Alert Accuracy", f"{alarm_acc:.0%}")
            with kpi_cols[4]:
                st.metric("❌ False Pos", st.session_state.false_positives)

            # ── Live flow table ──
            with flow_placeholder.container():
                if st.session_state.flow_buffer:
                    df_show = pd.DataFrame(list(st.session_state.flow_buffer))[-20:]
                    st.dataframe(df_show, use_container_width=True, hide_index=True,
                                 column_config={
                                     'flow': 'Flow#', 'time': 'Time', 'predicted': 'Predicted',
                                     'actual': 'Actual', 'confidence': 'Conf',
                                     'result': st.column_config.SelectboxColumn('Result', width='small'),
                                     'latency': 'Latency'
                                 })

            # ── Alerts panel ──
            with alert_placeholder.container():
                st.markdown(f"**🚨 Recent Alerts ({len(st.session_state.alert_buffer)})**")
                if st.session_state.alert_buffer:
                    df_alerts = pd.DataFrame(list(st.session_state.alert_buffer))[-10:]
                    st.dataframe(df_alerts, use_container_width=True, hide_index=True)

            progress_bar.progress(min(idx / total, 1.0))
            time.sleep(update_interval)

        # Simulation complete
        progress_bar.progress(1.0)
        elapsed_total = time.time() - start_global
        avg_lat_all = np.mean(per_flow_latencies) * 1000 if per_flow_latencies else 0
        throughput_total = total / elapsed_total if elapsed_total > 0 else 0

        st.session_state.running = False
        st.session_state.simulation_complete = True
        st.success(f"✅ Simulation complete: {total} flows in {elapsed_total:.2f}s")
        st.rerun()

# ──────────────────────────────────────────────
# Phase 3: Results Summary (shown after simulation)
# ──────────────────────────────────────────────
if st.session_state.get('simulation_complete', False) and not st.session_state.get('running', True):
    st.divider()
    st.markdown("## 📊 Simulation Results")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        # Latency distribution
        lat_data = list(st.session_state.latency_history)
        if lat_data:
            lat_ms = [l * 1000 for l in lat_data]
            avg_lats = np.mean(lat_ms)
            p99_lats = np.percentile(lat_ms, 99)
            st.metric("Average Latency", f"{avg_lats:.0f} μs")
            st.metric("P99 Latency", f"{p99_lats:.0f} μs")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=lat_ms, nbinsx=30, marker_color='#3498db',
                                        name='Latency'))
            fig.add_vline(x=avg_lats, line_dash="dash", line_color="red",
                          annotation_text=f"Mean: {avg_lats:.0f}μs")
            fig.add_vline(x=p99_lats, line_dash="dash", line_color="orange",
                          annotation_text=f"P99: {p99_lats:.0f}μs")
            fig.update_layout(title="Per-Flow Latency Distribution",
                              xaxis_title="Latency (μs)", yaxis_title="Frequency",
                              height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        # Classification breakdown (alerts)
        alert_data = list(st.session_state.alert_buffer)
        if alert_data:
            df_al = pd.DataFrame(alert_data)
            class_counts = df_al['predicted'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=class_counts.index,
                                          values=class_counts.values,
                                          hole=0.4)])
            fig.update_layout(title="Detected Attack Distribution", height=350)
            st.plotly_chart(fig, use_container_width=True)

    # Confusion summary
    st.divider()
    st.markdown("### 📋 Detection Summary")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Total Flows", st.session_state.flow_count)
    with col_s2:
        st.metric("Total Alerts", st.session_state.alert_count)
    with col_s3:
        acc = st.session_state.correct_alerts / max(st.session_state.alert_count, 1)
        st.metric("Alert Precision", f"{acc:.1%}")
    with col_s4:
        fn = st.session_state.false_negatives
        total_attacks = st.session_state.alert_count + fn - st.session_state.false_positives
        tpr = (st.session_state.alert_count - st.session_state.false_positives) / max(total_attacks, 1)
        st.metric("Detection Rate (TPR)", f"{tpr:.1%}")

    if st.button("🔄 New Simulation", use_container_width=True):
        st.session_state.simulation_complete = False
        st.session_state.flow_buffer.clear()
        st.session_state.alert_buffer.clear()
        st.session_state.latency_history.clear()
        st.session_state.flow_count = 0
        st.session_state.alert_count = 0
        st.session_state.correct_alerts = 0
        st.session_state.false_positives = 0
        st.session_state.false_negatives = 0
        st.rerun()
