import glob, os, gc, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
import joblib

warnings.filterwarnings('ignore')

CSV_FOLDER = './MachineLearningCVE'
LABEL_COL = 'Label'
N_FOLDS = 10
TEST_SIZE = 0.30
RANDOM_STATE = 42
OUTPUT_DIR = './saved_model'

print("Loading data...")
csv_files = sorted(glob.glob(os.path.join(CSV_FOLDER, '*.csv')))
dfs = []
for f in csv_files:
    for chunk in pd.read_csv(f, chunksize=50000, low_memory=False, encoding='latin-1'):
        dfs.append(chunk)
    gc.collect()

df = pd.concat(dfs, ignore_index=True)
print(f"Loaded: {len(df)} rows")

df.columns = df.columns.str.strip()
drop_cols = [c for c in ['Flow ID','Source IP','Destination IP','Timestamp',
                          'Src IP','Dst IP','src_ip','dst_ip'] if c in df.columns]
if drop_cols:
    df.drop(columns=drop_cols, inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(axis=1, thresh=int(len(df) * 0.5), inplace=True)
df.dropna(axis=0, inplace=True)
df.drop_duplicates(inplace=True)

le = LabelEncoder()
df['label_enc'] = le.fit_transform(df[LABEL_COL].astype(str))
n_classes = len(le.classes_)
print(f"Classes: {n_classes}")

X = df.drop(columns=[LABEL_COL, 'label_enc']).select_dtypes(include=[np.number])
y = df['label_enc']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balanced training set (same as v4: 2% of largest class cap)
class_counts = pd.Series(y_train).value_counts().sort_values()
target_count = max(int(class_counts.max() * 0.02), 1000)
sampling_dict = {}
for cls in range(n_classes):
    if class_counts[cls] > target_count:
        sampling_dict[cls] = target_count
undersampler = RandomUnderSampler(sampling_strategy=sampling_dict, random_state=RANDOM_STATE)
X_bal, y_bal = undersampler.fit_resample(X_train_scaled, y_train)
print(f"Balanced CV data: {X_bal.shape}")

# 10-fold CV collecting OOF predictions
print(f"\nRunning {N_FOLDS}-fold CV...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
Xa = X_bal
ya = y_bal.values

oof_lgbm = np.zeros((len(Xa), n_classes))
oof_bagging = np.zeros((len(Xa), n_classes))
fold_scores = []
best_score = -1
best_lgbm = None
best_bagging = None

n_features = Xa.shape[1]

for fold, (tr_idx, val_idx) in enumerate(skf.split(Xa, ya), 1):
    Xtr, Xval = Xa[tr_idx], Xa[val_idx]
    ytr, yval = ya[tr_idx], ya[val_idx]

    lgbm_fold = lgb.LGBMClassifier(
        objective='multiclass', metric='multi_logloss', num_class=n_classes,
        n_estimators=100, learning_rate=0.05, max_depth=8, num_leaves=63,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, is_unbalance=False,
        class_weight='balanced', device='cpu', random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1
    )
    lgbm_fold.fit(Xtr, ytr)
    oof_lgbm[val_idx] = lgbm_fold.predict_proba(Xval)
    lgbm_acc = accuracy_score(yval, lgbm_fold.predict(Xval))

    bagging_fold = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            criterion='gini', max_depth=10,
            min_samples_split=20, min_samples_leaf=10,
            class_weight='balanced', random_state=RANDOM_STATE
        ),
        n_estimators=50, max_samples=0.8,
        max_features=max(1, int(np.sqrt(n_features))),
        bootstrap=True, n_jobs=1, random_state=RANDOM_STATE
    )
    bagging_fold.fit(Xtr, ytr)
    oof_bagging[val_idx] = bagging_fold.predict_proba(Xval)
    bagging_acc = accuracy_score(yval, bagging_fold.predict(Xval))

    score = (lgbm_acc + bagging_acc) / 2
    fold_scores.append({'fold': fold, 'lgbm_acc': lgbm_acc, 'bagging_acc': bagging_acc, 'avg': score})
    print(f"  Fold {fold}: LGBM={lgbm_acc:.4f} Bagging={bagging_acc:.4f} avg={score:.4f}")

    if score > best_score:
        best_score = score
        best_lgbm = lgbm_fold
        best_bagging = bagging_fold

print(f"\nBest fold: Fold {np.argmax([s['avg'] for s in fold_scores]) + 1} (avg={best_score:.4f})")

# Train meta-learner on ALL OOF predictions (same as v4)
print("\nTraining meta-learner on all OOF predictions...")
M_train = np.hstack([oof_lgbm, oof_bagging])
print(f"Meta training data: {M_train.shape}")

meta = LogisticRegression(
    max_iter=1000, random_state=RANDOM_STATE,
    class_weight='balanced', solver='lbfgs', n_jobs=-1
)
meta.fit(M_train, ya)
meta_train_acc = accuracy_score(ya, meta.predict(M_train))
print(f"Meta-learner OOF accuracy: {meta_train_acc:.4f}")

# Save all models
print("\nSaving models...")
# Use best-fold base models + meta-learner
joblib.dump(best_lgbm, f'{OUTPUT_DIR}/lgbm_model.pkl')
joblib.dump(best_bagging, f'{OUTPUT_DIR}/bagging_model.pkl')
joblib.dump(meta, f'{OUTPUT_DIR}/meta_learner.pkl')
joblib.dump(scaler, f'{OUTPUT_DIR}/scaler.pkl')
joblib.dump(le, f'{OUTPUT_DIR}/label_encoder.pkl')
joblib.dump(fold_scores, f'{OUTPUT_DIR}/fold_results.pkl')

# Quick test on test subset
print("Evaluating on test set...")
np.random.seed(RANDOM_STATE)
test_idx = np.random.choice(len(X_test_scaled), size=min(50000, len(X_test_scaled)), replace=False)
X_ts = X_test_scaled[test_idx]

p1 = best_lgbm.predict_proba(X_ts)
p2 = best_bagging.predict_proba(X_ts)
M_test = np.hstack([p1, p2])
y_pred = meta.predict(M_test)
y_true = y_test.iloc[test_idx].values if hasattr(y_test, 'iloc') else y_test[test_idx]

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
print(f"Stacked ensemble test accuracy: {acc:.4f}, F1: {f1:.4f}")

# Compare with LGBM alone
lgbm_acc_test = accuracy_score(y_true, best_lgbm.predict(X_ts))
print(f"LGBM alone test accuracy: {lgbm_acc_test:.4f}")

print("\nDone! All models saved to saved_model/")
