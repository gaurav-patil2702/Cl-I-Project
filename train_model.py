from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'sample_network_traffic.csv'
MODEL_DIR = ROOT / 'models'
REPORT_DIR = ROOT / 'reports'
FEATURES = ['flow_duration_us', 'src_bytes', 'dst_bytes', 'packet_count', 'packet_rate_per_sec', 'failed_connections', 'unique_dst_ports', 'dst_port', 'protocol_code']
TARGET = 'label'

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATA_PATH}')
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan).drop_duplicates()
    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError('Missing dataset columns: ' + ', '.join(missing))
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
    df = df[df[TARGET].isin(['normal', 'anomalous'])].copy()
    X = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    y = df[TARGET].map({'normal': 0, 'anomalous': 1})
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError('Training requires at least two examples of each class.')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    supervised = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(n_estimators=250, class_weight='balanced', random_state=42, min_samples_leaf=2, n_jobs=-1))
    ])
    supervised.fit(X_train, y_train)
    pred = supervised.predict(X_test)
    proba = supervised.predict_proba(X_test)[:, list(supervised.classes_).index(1)]
    metrics = {
        'model': 'Random Forest Classifier',
        'test_rows': int(len(y_test)),
        'accuracy': float(accuracy_score(y_test, pred)),
        'precision_anomalous': float(precision_score(y_test, pred, zero_division=0)),
        'recall_anomalous': float(recall_score(y_test, pred, zero_division=0)),
        'f1_anomalous': float(f1_score(y_test, pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        'classification_report': classification_report(y_test, pred, labels=[0, 1], target_names=['normal', 'anomalous'], output_dict=True, zero_division=0),
        'feature_importances': {name: float(value) for name, value in zip(FEATURES, supervised.named_steps['model'].feature_importances_)},
    }
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_test, proba))
    except ValueError:
        metrics['roc_auc'] = None
    joblib.dump(supervised, MODEL_DIR / 'random_forest_pipeline.joblib')
    with (REPORT_DIR / 'evaluation_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print('Training complete.')
    print(json.dumps({k: v for k, v in metrics.items() if k != 'classification_report'}, indent=2))
    print(f'Saved models to: {MODEL_DIR}')
    print(f'Saved metrics to: {REPORT_DIR / "evaluation_metrics.json"}')

if __name__ == '__main__':
    main()
