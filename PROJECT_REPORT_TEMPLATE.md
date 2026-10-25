# Project Report Template

## Title
**Intelligent Network Anomaly Detection and Traffic Behaviour Analysis Using Machine Learning**

## Abstract
This project demonstrates a machine-learning workflow for analyzing network-flow records and identifying unusual traffic patterns. It combines data cleaning, feature preparation, supervised classification, unsupervised anomaly detection, evaluation metrics, and an interactive Streamlit dashboard. The application supports manual feature entry and batch CSV analysis. The included dataset is synthetic and is used only to demonstrate the end-to-end implementation. Real-world performance must be evaluated separately using a documented public network intrusion dataset.

## Problem Statement
Manual inspection of large volumes of network-flow records can be difficult. This project explores whether machine learning and visualization can help flag records whose features resemble known anomalous traffic or differ from the patterns learned by an anomaly detector.

## Objectives
1. Clean and prepare network-flow data.
2. Explore traffic features and class distributions.
3. Train a Random Forest supervised classifier.
4. Train an Isolation Forest unsupervised anomaly detector.
5. Compare predictions using precision, recall, F1-score, and confusion matrices.
6. Build a dashboard for manual input, batch prediction, and visualization.
7. Document limitations and responsible interpretation of results.

## Methodology
1. Load data and inspect columns, types, missing values, duplicates, and extreme values.
2. Replace infinite values with missing values and impute numeric missing values in the pipeline.
3. Split labelled data into training and held-out testing partitions.
4. Train Random Forest on labelled examples.
5. Train Isolation Forest without using labels in its fitting process.
6. Evaluate the models on the held-out test set.
7. Deploy the fitted pipelines in a Streamlit interface.
8. Accept a manual record or a compatible CSV and return predictions.

## Tools and Technologies
Python, Pandas, NumPy, Scikit-learn, Joblib, Matplotlib, Seaborn, Streamlit.

## Results
Insert the actual values from `reports/evaluation_metrics.json` after running `python train_model.py`. State clearly that the included data is synthetic. Do not invent performance claims or present synthetic-data scores as real-world results.

## Limitations
The included sample is synthetic. The project does not perform live packet capture, automatic blocking, or production security monitoring. The results are dependent on the dataset, feature definitions, model configuration, and evaluation procedure.

## Future Scope
- Retrain and evaluate on CICIDS2017 or UNSW-NB15.
- Add feature mapping for the selected public dataset.
- Add explainability such as permutation importance or SHAP.
- Evaluate robustness to class imbalance and data drift.
- Add authorized, privacy-conscious flow collection only if permitted and within project scope.

## Conclusion
Summarize what the implemented workflow demonstrates, the evaluation results actually obtained, and the limitations that remain.
