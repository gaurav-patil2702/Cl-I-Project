# Intelligent Network Anomaly Detection and Traffic Behaviour Analysis

**Computer Laboratory-I (CL-I) Mini Project**  
Combines Machine Learning, data cleaning, feature preparation, model evaluation, aggregation, and visualization.

## What this project does

- Trains a **Random Forest classifier** to predict `normal` vs `anomalous` network-flow records.
- Trains an **Isolation Forest** model to flag unusual records without using labels during training.
- Provides a **Streamlit dashboard** with:
  - Dataset overview and class distribution
  - Manual network-flow input and prediction
  - Batch CSV upload, predictions, and downloadable results
  - Evaluation metrics and confusion matrix
- Demonstrates preprocessing of missing and infinite values.

> **Important data note:** `data/sample_network_traffic.csv` is a synthetic demonstration dataset generated for this project. It is not real captured traffic and its evaluation scores must not be represented as real-world cybersecurity performance. For a research-quality submission, use a documented public dataset such as CICIDS2017 or UNSW-NB15 and adapt the feature columns in `train_model.py` and `app.py`.

## Requirements

- Python 3.10 or newer recommended
- pip
- Internet connection only if installing dependencies or downloading a real dataset

## Run on Windows

Open Command Prompt or PowerShell in the extracted project folder.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

If PowerShell blocks activation, you can use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

### 2. Install libraries

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train the models

```powershell
python train_model.py
```

This saves the trained pipelines under `models/` and evaluation metrics under `reports/`.

### 4. Start the dashboard

```powershell
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in your browser.

## Run on macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Manual input prediction

Open the **Manual Prediction** tab and enter numeric values for all nine features:

| Feature | Meaning |
|---|---|
| `flow_duration_us` | Flow duration in microseconds |
| `src_bytes` | Bytes in the source-to-destination direction |
| `dst_bytes` | Bytes in the destination-to-source direction |
| `packet_count` | Total packets in the flow |
| `packet_rate_per_sec` | Packets per second |
| `failed_connections` | Count of failed connection attempts |
| `unique_dst_ports` | Number of distinct destination ports represented by the flow/session |
| `dst_port` | Destination port number |
| `protocol_code` | Example numeric protocol code: TCP 6, UDP 17, ICMP 1 |

These are demonstration features. They must be extracted consistently from the selected dataset or network-flow tool. Do not enter arbitrary values and treat the prediction as a reliable security finding.

## Batch CSV format

The uploaded CSV must contain the nine feature columns listed above. A `label` column may be present, but it is not used for prediction. The app outputs the original columns plus:
- `random_forest_prediction`
- `anomaly_probability_estimate`
- `isolation_forest_prediction`

## Model details

### Random Forest
Supervised classifier trained on labelled examples. The label mapping is `normal = 0`, `anomalous = 1`. Evaluation uses a held-out stratified test set and reports accuracy, precision, recall, F1-score, ROC AUC when defined, and a confusion matrix.

### Isolation Forest
Unsupervised anomaly detector fitted without labels. It returns `1` for inlier and `-1` for outlier; the app maps these to `normal` and `anomalous` for display. Its contamination setting is a tunable assumption, not a discovered ground truth.

## Suggested report sections

1. Abstract
2. Introduction and problem statement
3. Objectives and scope
4. Dataset description and limitations
5. System architecture
6. Data preprocessing and feature engineering
7. ML algorithms and methodology
8. Evaluation metrics and results
9. Dashboard screenshots and discussion
10. Limitations, future scope, and conclusion

## Ethical and technical limitations

- This is a learning prototype, not a production intrusion detection system.
- It does not capture live packets, inspect encrypted content, block connections, or replace professional security tools.
- An anomaly is not necessarily an attack, and a low anomaly score does not guarantee safety.
- Before claiming real-world results, train and test with a properly documented real dataset, check class imbalance, avoid data leakage, and report limitations.
