import glob, os, gc, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
import joblib

warnings.filterwarnings('ignore')

CSV_FOLDER = './MachineLearningCVE'
LABEL_COL = 'Label'
N_FOLDS = 10
TEST_SIZE = 0.30
RANDOM_STATE = 42
OUTPUT_DIR = './saved_model'
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(CSV_FOLDER, '*.csv')))
dfs = []
for f in csv_files:
    for chunk in pd.read_csv(f, chunksize=50000, low_memory=False, encoding='latin-1'):
        dfs.append(chunk)
    gc.collect()

df = pd.concat(dfs, ignore_index=True)
print(f'Loaded: {len(df)} rows')

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
print(f'Classes: {n_classes}')

X = df.drop(columns=[LABEL_COL, 'label_enc']).select_dtypes(include=[np.number])
y = df['label_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')

scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

class_counts = y_train.value_counts().sort_values()
target_count = max(int(class_counts.max() * 0.02), 1000)
sampling_dict = {}
for cls in range(n_classes):
    if class_counts[cls] > target_count:
        sampling_dict[cls] = target_count
undersampler = RandomUnderSampler(sampling_strategy=sampling_dict, random_state=RANDOM_STATE)
X_train_bal, y_train_bal = undersampler.fit_resample(X_train_scaled, y_train)
print(f'Balanced training: {X_train_bal.shape}')

# Save scaler and label encoder
joblib.dump(scaler, f'{OUTPUT_DIR}/scaler.pkl')
joblib.dump(le, f'{OUTPUT_DIR}/label_encoder.pkl')

# Train LightGBM on FULL balanced training data
print('\nTraining LightGBM on full data...')
lgbm = lgb.LGBMClassifier(
    objective='multiclass', metric='multi_logloss', num_class=n_classes,
    n_estimators=100, learning_rate=0.05, max_depth=8, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, is_unbalance=False,
    class_weight='balanced', device='cpu', random_state=RANDOM_STATE,
    n_jobs=-1, verbose=-1
)
lgbm.fit(X_train_bal, y_train_bal)
joblib.dump(lgbm, f'{OUTPUT_DIR}/lgbm_model.pkl')
print(f'LightGBM saved. Classes: {lgbm.n_classes_}')

# Train Bagging on FULL balanced training data
print('Training Bagging on full data...')
n_features = X_train_bal.shape[1]
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        criterion='gini', max_depth=10,
        min_samples_split=20, min_samples_leaf=10,
        class_weight='balanced', random_state=RANDOM_STATE
    ),
    n_estimators=50, max_samples=0.8,
    max_features=max(1, int(np.sqrt(n_features))),
    bootstrap=True, n_jobs=-1, random_state=RANDOM_STATE
)
bagging.fit(X_train_bal, y_train_bal)
joblib.dump(bagging, f'{OUTPUT_DIR}/bagging_model.pkl')
print(f'Bagging saved. Classes: {bagging.n_classes_}')

# Now train meta-learner using cross-validation (like v4 notebook)
print('\nRunning cross-validation for meta-learner...')
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
Xa = X_train_bal.values
ya = y_train_bal.values
Xta = X_test_scaled.values

oof_lgbm = np.zeros((len(Xa), n_classes))
oof_bagging = np.zeros((len(Xa), n_classes))
test_lgbm_accum = np.zeros((len(Xta), n_classes))
test_bagging_accum = np.zeros((len(Xta), n_classes))

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
    test_lgbm_accum += lgbm_fold.predict_proba(Xta)

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
    test_bagging_accum += bagging_fold.predict_proba(Xta)

    print(f'  Fold {fold}/{N_FOLDS} done')

test_lgbm_avg = test_lgbm_accum / N_FOLDS
test_bagging_avg = test_bagging_accum / N_FOLDS

from sklearn.linear_model import LogisticRegression
M_train = np.hstack([oof_lgbm, oof_bagging])
M_test = np.hstack([test_lgbm_avg, test_bagging_avg])
print(f'Meta features: Train {M_train.shape}, Test {M_test.shape}')

meta = LogisticRegression(
    max_iter=1000, random_state=RANDOM_STATE,
    class_weight='balanced', solver='lbfgs', n_jobs=-1
)
meta.fit(M_train, ya)
joblib.dump(meta, f'{OUTPUT_DIR}/meta_learner.pkl')

# Final evaluation
y_pred = meta.predict(M_test)
from sklearn.metrics import accuracy_score, f1_score
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
print(f'\nTest accuracy: {acc:.4f}, F1: {f1:.4f}')
print('All models saved to saved_model/')
