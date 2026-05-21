import glob, os, gc, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
import joblib

warnings.filterwarnings('ignore')

CSV_FOLDER = './MachineLearningCVE'
LABEL_COL = 'Label'
N_FOLDS = 5
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

# Use 5% of training data for CV (to manage memory)
class_counts = pd.Series(y_train).value_counts().sort_values()
target_count = max(int(class_counts.max() * 0.05), 2000)
sampling_dict = {}
for cls in range(n_classes):
    if class_counts[cls] > target_count:
        sampling_dict[cls] = target_count
undersampler = RandomUnderSampler(sampling_strategy=sampling_dict, random_state=RANDOM_STATE)
X_bal, y_bal = undersampler.fit_resample(X_train_scaled, y_train)
print(f"Balanced CV data: {X_bal.shape}")

# Subsample test set for CV prediction to manage memory
np.random.seed(RANDOM_STATE)
test_idx = np.random.choice(len(X_test_scaled), size=min(100000, len(X_test_scaled)), replace=False)
X_test_sub = X_test_scaled[test_idx]
y_test_sub = y_train.iloc[:0].copy()
print(f"Test subset: {X_test_sub.shape}")

print("\nRunning CV for meta-learner training...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
Xa = X_bal
ya = y_bal.values

oof_lgbm = np.zeros((len(Xa), n_classes))
oof_bagging = np.zeros((len(Xa), n_classes))
n_test = len(X_test_sub)
test_lgbm_accum = np.zeros((n_test, n_classes))
test_bagging_accum = np.zeros((n_test, n_classes))

for fold, (tr_idx, val_idx) in enumerate(skf.split(Xa, ya), 1):
    print(f"  Fold {fold}/{N_FOLDS}...", end="")
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
    test_lgbm_accum += lgbm_fold.predict_proba(X_test_sub)
    print(" LGBM", end="")

    bagging_fold = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            criterion='gini', max_depth=10,
            min_samples_split=20, min_samples_leaf=10,
            class_weight='balanced', random_state=RANDOM_STATE
        ),
        n_estimators=20, max_samples=0.8,
        max_features=max(1, int(np.sqrt(Xa.shape[1]))),
        bootstrap=True, n_jobs=1, random_state=RANDOM_STATE
    )
    bagging_fold.fit(Xtr, ytr)
    oof_bagging[val_idx] = bagging_fold.predict_proba(Xval)
    test_bagging_accum += bagging_fold.predict_proba(X_test_sub)
    print(" Bagging OK")

test_lgbm_avg = test_lgbm_accum / N_FOLDS
test_bagging_avg = test_bagging_accum / N_FOLDS

print(f"\nTraining meta-learner...")
M_train = np.hstack([oof_lgbm, oof_bagging])
M_test = np.hstack([test_lgbm_avg, test_bagging_avg])
print(f"Meta features: Train {M_train.shape}, Test {M_test.shape}")

meta = LogisticRegression(
    max_iter=1000, random_state=RANDOM_STATE,
    class_weight='balanced', solver='lbfgs', n_jobs=-1
)
meta.fit(M_train, ya)

# Evaluate
y_pred = meta.predict(M_test)
y_pred_full = meta.predict(M_train)
train_acc = accuracy_score(ya, y_pred_full)

# Get true labels for test subset
y_test_subset = y_train.iloc[test_idx].values if hasattr(y_train, 'iloc') else y_test[test_idx]
if len(y_test_subset) != n_test:
    from sklearn.model_selection import train_test_split
    _, _, yt_subset, _ = train_test_split(X_test_scaled, y_test, test_size=n_test/len(y_test), random_state=RANDOM_STATE, stratify=y_test)
    y_test_subset = yt_subset

acc = accuracy_score(y_test_subset, y_pred)
print(f"Train accuracy (OOF): {train_acc:.4f}")
print(f"Test accuracy: {acc:.4f}")

joblib.dump(meta, f'{OUTPUT_DIR}/meta_learner.pkl')
print(f"Meta-learner saved to {OUTPUT_DIR}/meta_learner.pkl")
