import glob, os, gc, warnings, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

CSV_FOLDER = './MachineLearningCVE'
LABEL_COL = 'Label'
RANDOM_STATE = 42
TEST_SIZE = 0.30
SAMPLES_PER_CLASS = 3

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
class_names = le.classes_
n_classes = len(class_names)

X = df.drop(columns=[LABEL_COL, 'label_enc']).select_dtypes(include=[np.number])
y = df['label_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')

feature_cols = X.columns.tolist()
print(f'Number of features: {len(feature_cols)}')

lines = []
for cls_idx in range(n_classes):
    cls_name = class_names[cls_idx]
    mask = (y_test == cls_idx)
    indices = np.where(mask)[0]
    print(f'{cls_name:35s}: {len(indices)} test samples')
    if len(indices) == 0:
        continue
    selected = np.random.choice(indices, size=min(SAMPLES_PER_CLASS, len(indices)), replace=False)
    for idx in selected:
        row = X_test.iloc[idx]
        vals = ','.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in row.values)
        lines.append(f'{cls_name}|{vals}')

print(f'\nTotal test vectors: {len(lines)}')

with open('cicids_test_vectors.txt', 'w') as f:
    f.write('# CICIDS 2017 test vectors (held-out 30% test split)\n')
    f.write('# Each line: LABEL|78 comma-separated values (raw, unscaled)\n')
    f.write('# Model never saw these during training.\n')
    f.write(f'# Features ({len(feature_cols)}): ')
    f.write(', '.join(feature_cols[:10]) + '...\n')
    f.write('# ========================================================\n')
    for line in lines:
        f.write(line + '\n')

print(f'Saved to cicids_test_vectors.txt')
