from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sample_network_traffic.csv"
MODEL_DIR = ROOT / "models"
REPORT_PATH = ROOT / "reports" / "evaluation_metrics.json"

FEATURES = [
    "flow_duration_us", "src_bytes", "dst_bytes", "packet_count",
    "packet_rate_per_sec", "failed_connections", "unique_dst_ports",
    "dst_port", "protocol_code"
]

FRIENDLY = {
    "flow_duration_us": "Flow duration (microseconds)",
    "src_bytes": "Source-to-destination bytes",
    "dst_bytes": "Destination-to-source bytes",
    "packet_count": "Total packet count",
    "packet_rate_per_sec": "Packets per second",
    "failed_connections": "Failed connection attempts",
    "unique_dst_ports": "Unique destination ports",
    "dst_port": "Destination port",
    "protocol_code": "Protocol code (TCP=6, UDP=17, ICMP=1)"
}

st.set_page_config(page_title="Network Anomaly Intelligence", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_model():
    rf_path = MODEL_DIR / "random_forest_pipeline.joblib"
    if not rf_path.exists():
        return None
    return joblib.load(rf_path)


def clean_features(df):
    data = df.copy()
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    data = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def status_text(pred):
    return "Anomalous" if int(pred) == 1 else "Normal"


@st.cache_data(show_spinner=False)
def load_demo_data():
    return pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan)


try:
    rf_model = load_model()
except Exception as exc:
    st.error(f"Could not load trained models: {exc}")
    st.info("Run `python train_model.py` and restart the application.")
    st.stop()

st.title("🛡️ Intelligent Network Anomaly Detection")

if rf_model is None:
    st.error("Models have not been trained yet. Run `python train_model.py` from the project folder, then restart Streamlit.")
    st.stop()

section = st.radio(
    "Select module",
    ["Overview", "Manual Prediction", "CSV Batch Analysis", "Model Evaluation", "About & Data"],
    horizontal=True,
    label_visibility="collapsed",
    key="active_module",
)
# Only the selected module executes, avoiding rendering every chart on each interaction.

if section == "Overview":
    st.subheader("Traffic overview")
    if DATA_PATH.exists():
        df = load_demo_data()
        total = len(df)
        anomalous = int((df.get("label", pd.Series(dtype=str)).astype(str).str.lower() == "anomalous").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Demo records", f"{total:,}")
        c2.metric("Normal records", f"{total - anomalous:,}")
        c3.metric("Anomalous-labelled records", f"{anomalous:,}")

        if "label" in df.columns:
            fig, ax = plt.subplots(figsize=(5, 2.5))
            sns.countplot(data=df, x="label", order=["normal", "anomalous"], ax=ax)
            ax.set_title("Synthetic dataset class distribution", fontsize=10)
            ax.set_xlabel("Label", fontsize=9)
            ax.set_ylabel("Record count", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)
            fig.tight_layout()
            left, middle, right = st.columns([1, 2, 1])
            with middle:
                st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.subheader("Sample records")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("Demo CSV not found in the data folder.")

if section == "Manual Prediction":
    st.subheader("Enter one network flow")
    st.write("Enter numeric features from a known flow record. Use realistic values and the same units as the dataset.")
    with st.form("manual_prediction_form"):
        col1, col2, col3 = st.columns(3)
        values = {}
        defaults = {
            "flow_duration_us": 1200.0, "src_bytes": 25000.0, "dst_bytes": 18000.0,
            "packet_count": 65, "packet_rate_per_sec": 54.0, "failed_connections": 0,
            "unique_dst_ports": 1, "dst_port": 443, "protocol_code": 6
        }
        for i, feature in enumerate(FEATURES):
            target_col = [col1, col2, col3][i % 3]
            values[feature] = target_col.number_input(
                FRIENDLY[feature], min_value=0.0, value=float(defaults[feature]),
                step=1.0, help="Illustrative default only; change this to the observed feature value."
            )
        submitted = st.form_submit_button("Analyze traffic", type="primary")

    if submitted:
        row = pd.DataFrame([values], columns=FEATURES)
        rf_pred = int(rf_model.predict(row)[0])
        rf_proba = rf_model.predict_proba(row)[0]
        anomaly_probability = float(rf_proba[list(rf_model.classes_).index(1)]) if 1 in rf_model.classes_ else float("nan")
        if rf_pred == 1:
            st.error("Random Forest result: ANOMALOUS")
        else:
            st.success("Random Forest result: NORMAL")
        st.metric("Model anomaly probability estimate", f"{anomaly_probability:.1%}")
        # st.caption("The probability estimate reflects the supervised model and synthetic training data; it is not a calibrated real-world risk score.")

        st.subheader("Random Forest prediction probabilities")
        left, middle, right = st.columns([1, 2, 1])
        with middle:
            fig, ax = plt.subplots(figsize=(5, 2.6))
            probabilities = [1 - anomaly_probability, anomaly_probability]
            bars = ax.barh(["Normal", "Anomalous"], probabilities, color=["#3b82f6", "#ef4444"])
            ax.set_xlim(0, 1.15)
            ax.set_xlabel("Estimated probability", fontsize=9)
            ax.set_title("Random Forest class probabilities", fontsize=10)
            ax.tick_params(labelsize=9)
            for bar, probability in zip(bars, probabilities):
                ax.text(probability + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{probability:.1%}", va="center", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.dataframe(row, use_container_width=True)
        st.download_button(
            "Download this prediction as CSV",
            data=row.assign(
                random_forest_prediction=status_text(rf_pred),
                anomaly_probability_estimate=anomaly_probability
            ).to_csv(index=False).encode("utf-8"),
            file_name="network_flow_prediction.csv", mime="text/csv"
        )

if section == "CSV Batch Analysis":
    st.subheader("Analyze multiple flows from CSV")
    st.write("Upload a CSV with all required numeric feature columns listed below. The target label is optional.")
    st.code(", ".join(FEATURES))
    uploaded = st.file_uploader("Upload network-flow CSV", type=["csv"], key="batch_csv")
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            X = clean_features(raw)
            rf_preds = rf_model.predict(X).astype(int)
            rf_proba = rf_model.predict_proba(X)
            anomaly_idx = list(rf_model.classes_).index(1) if 1 in rf_model.classes_ else None
            anomaly_prob = rf_proba[:, anomaly_idx] if anomaly_idx is not None else np.zeros(len(X))
            result = raw.copy()
            result["random_forest_prediction"] = np.where(rf_preds == 1, "anomalous", "normal")
            result["anomaly_probability_estimate"] = anomaly_prob
            st.success(f"Analyzed {len(result):,} rows.")
            st.metric("Random Forest flagged anomalous", int((rf_preds == 1).sum()))
            st.dataframe(result.head(100), use_container_width=True)
            counts = result["random_forest_prediction"].value_counts().reindex(["normal", "anomalous"], fill_value=0)
            fig, ax = plt.subplots(figsize=(5, 2.5))
            counts.plot(kind="bar", ax=ax)
            ax.set_title("Predictions in uploaded file (Random Forest)", fontsize=10)
            ax.set_xlabel("Predicted class", fontsize=9)
            ax.set_ylabel("Record count", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)
            fig.tight_layout()
            left, middle, right = st.columns([1, 2, 1])
            with middle:
                st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.download_button(
                "Download all predictions", data=result.to_csv(index=False).encode("utf-8"),
                file_name="network_traffic_predictions.csv", mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Could not analyze this file: {exc}")

if section == "Model Evaluation":
    st.subheader("Held-out evaluation")
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.caption("Metrics are calculated on a held-out split of the included synthetic demo dataset. They are not evidence of real-world detection performance.")
        rf = metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{rf.get('accuracy', 0):.3f}")
        c2.metric("Precision", f"{rf.get('precision_anomalous', 0):.3f}")
        c3.metric("Recall", f"{rf.get('recall_anomalous', 0):.3f}")
        c4.metric("F1-score", f"{rf.get('f1_anomalous', 0):.3f}")
        cm = np.array(rf.get("confusion_matrix", [[0, 0], [0, 0]]))
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Anomalous"],
                    yticklabels=["Normal", "Anomalous"], ax=ax)
        ax.set_xlabel("Predicted label", fontsize=9)
        ax.set_ylabel("Actual label", fontsize=9)
        ax.set_title("Random Forest confusion matrix", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        fig.tight_layout()
        left, middle, right = st.columns([1, 2, 1])
        with middle:
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.json({
            "test_rows": rf.get("test_rows"),
            "random_forest_roc_auc": rf.get("roc_auc")
        })
    else:
        st.warning("Evaluation report not found. Run `python train_model.py` first.")

if section == "About & Data":
    st.subheader("Project scope")
    st.markdown("""
    **Objective:** demonstrate network-flow preprocessing, supervised Random Forest classification,
    model evaluation, data aggregation, and interactive visual analytics.

    **Included features**
    - Manual input prediction
    - Batch CSV prediction and downloadable output
    - Random Forest supervised classifier
    - Confusion matrix, precision, recall, F1-score and accuracy
    - Data overview and prediction distribution chart

    **Limitations**
    - The bundled dataset is synthetic and only demonstrates software functionality.
    - It does not inspect live packets or automatically block traffic.
    - A prediction is not proof of malicious activity.
    - For an academic evaluation using real data, train and evaluate using a documented public dataset such as CICIDS2017 or UNSW-NB15, adapting the feature mapping and training code to that dataset.
    """)
    st.subheader("Required feature columns")
    st.dataframe(pd.DataFrame({"column": FEATURES, "meaning": [FRIENDLY[x] for x in FEATURES]}), use_container_width=True)
