# UEBA for Threat Detection Using Quantum Machine Learning

## Overview

This repository contains the implementation of a Quantum Machine Learning (QML)-based User and Entity Behavior Analytics (UEBA) framework for advanced threat detection in cybersecurity, as described in the dissertation titled *"UEBA for Threat Detection Using Quantum Machine Learning"* (see `docs/dissertation.docx`). The framework leverages quantum computing principles to enhance anomaly detection, behavioral profiling, alert prioritization, and risk scoring, addressing limitations of classical machine learning (ML) in scalability, accuracy, and real-time processing.

The project is structured around four core components:
1. **Anomaly Detection**: Identifies unusual patterns in network traffic using a Quantum Autoencoder (QAE).
2. **Behavioral Profiling**: Clusters user and entity behaviors to establish baselines for anomaly detection.
3. **Alert Policy**: Prioritizes alerts based on severity (Critical, High, Medium) using a Quantum Neural Network (QNN).
4. **Risk Scoring**: Quantifies the risk of detected anomalies to guide response prioritization.

The implementation uses Python with PyTorch for quantum circuit simulation, achieving accuracies of 94–96% and AUC values of 0.977–0.998 across components, as detailed in the dissertation (Chapter 4).

## Repository Structure

## Prerequisites

- **Python**: 3.10 or higher
- **Hardware**: CUDA-enabled GPU recommended for faster computation
- **Dataset**: CIC-UNSW-NB15 (subset provided in `data/raw/CICFlowMeter_out.csv`)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ueba-qml-threat-detection.git
   cd ueba-qml-threat-detection
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes:
   - `torch>=2.0.0`
   - `numpy>=1.26.0`
   - `pandas>=2.2.0`
   - `scikit-learn>=1.5.0`
   - `joblib>=1.4.0`
   - `matplotlib>=3.9.0`

4. **Download Dataset**:
   

- **UNSW-NB15 Dataset for Anomaly Detection**  
  Source: https://www.unb.ca/cic/datasets/cic-unsw-nb15.html  
  Filename: `unsw-nb15.csv`

- **LANL Dataset for Behavioral Profiling**  
  Source: https://csr.lanl.gov/data/auth/  
  Filename: `lanl_auth_data.csv`

- **Alerting and Policy Enforcement**  
  Source: https://data.mendeley.com/datasets/p6tym3fghz/1  
  Filename: `alerting_policy_data.csv`

- **Risk Scoring**  
  Source: https://www.kaggle.com/datasets/pengr252/ueba-user-and-entity-behavior-analytics  
  Filename: `risk_scoring_data.csv`

After downloading, run the preprocessing notebooks (e.g., `Anomaly_Detection/2_data_preprocessing.ipynb`) to generate processed data in the `processed/` directory.

## Usage

Each component can be run independently or sequentially using the provided scripts. The `run_all.sh` script executes all components in order: preprocessing, training, and evaluation.

### Running All Components
```bash
bash run_all.sh
```

This script:
1. Preprocesses the dataset for each component.
2. Trains the respective QML models (QAE or QNN).
3. Evaluates models and generates visualizations (saved in `models/*/plots/`).

### Running Individual Components
Each component has its own directory with scripts for preprocessing, training, and evaluation. For example, to run the Anomaly Detection component:


Similar commands apply to `behavioral_profiling`, `alert_policy`, and `risk_scoring`. Check each directory's `README.md` for specific arguments.

### Jupyter Notebooks
Explore the components interactively using the provided notebooks:
```bash
cd notebooks
jupyter notebook
```

## Component Details

### 1. Anomaly Detection
- **Purpose**: Detects unusual network traffic patterns (e.g., insider threats, malware) using a Quantum Autoencoder.
- **Methodology**:
  - Preprocesses CIC-UNSW-NB15 dataset (47 features reduced to 31, normalized).
  - Applies PCA to reduce to 2 dimensions for 2-qubit compatibility.
  - Uses a QAE with ZZFeatureMap encoding and RealAmplitudes ansatz (3 layers).
  - Trains for 200 epochs with MSE loss, batch size 4.
  - Evaluates anomalies via reconstruction error (threshold-based).
- **Performance**: Accuracy 95%, AUC 0.988
  
### 2. Behavioral Profiling
- **Purpose**: Clusters user and entity behaviors into three groups to establish baselines for anomaly detection.
- **Methodology**:
  - Preprocesses data to extract behavioral features (e.g., login patterns, file access).
  - Uses a QNN with variational quantum circuits for clustering.
  - Trains for 200 epochs with cross-entropy loss.
  - Evaluates using precision, recall, and confusion matrix.
- **Performance**: Accuracy 96%, AUC 0.979–0.998

  
### 3. Alert Policy
- **Purpose**: Classifies alerts into Critical, High, and Medium severity for efficient incident response.
- **Methodology**:
  - Preprocesses alert data to extract severity-related features.
  - Uses a QNN with parameterized quantum circuits for classification.
  - Trains for 200 epochs with cross-entropy loss.
  - Evaluates using ROC and precision-recall curves.
- **Performance**: Accuracy 94%, AUC 0.977–0.998

  
### 4. Risk Scoring
- **Purpose**: Assigns risk scores (Low, Medium, High) to detected anomalies for prioritization.
- **Methodology**:
  - Preprocesses data to focus on risk-relevant features.
  - Uses a QAE to compute reconstruction errors as risk scores.
  - Normalizes scores to [0,1] using the 95th percentile.
  - Trains for 200 epochs with MSE loss.
  - Evaluates using risk score histograms and ROC curves.
- **Performance**: Accuracy 96%, AUC 0.981–0.998.

## Visualizations

Each component generates visualizations saved in `models/*/plots/`:
- **Training Loss Curves**: Monitors convergence.
- **Risk Score Histograms**: Shows distribution of anomaly/risk scores.
- **ROC Curves**: Evaluates discriminative power.
- **Precision-Recall Curves**: Assesses performance on imbalanced data.
- **Confusion Matrices**: Details classification performance.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure code follows PEP 8 standards and includes tests.

## Limitations

- **Simulation-Based**: Models are tested in simulated quantum environments (PyTorch-based), not on real quantum hardware, due to current NISQ limitations 
- **Dataset**: Uses a subset of CIC-UNSW-NB15; real-world data may introduce noise or concept drift.
- **Interpretability**: QNNs and QAEs lack full interpretability, a challenge for cybersecurity applications.

## Future Work

- Validate models with real-world datasets.
- Improve QML interpretability using hybrid quantum-classical approaches.
- Deploy on quantum hardware as it matures.
- Extend to other cybersecurity domains (e.g., fraud detection).



## Contact

For questions or issues, please open an issue on GitHub or contact [Your Email].
