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

```
├── data/
│   ├── raw/
│   │   └── CICFlowMeter_out.csv        # Sample dataset (CIC-UNSW-NB15 subset)
│   ├── processed/
│   │   └── raw_traffic_data.csv        # Preprocessed data for visualization
├── src/
│   ├── anomaly_detection/
│   │   ├── quantum_autoencoder.py      # Quantum Autoencoder implementation
│   │   ├── preprocess.py               # Data preprocessing for anomaly detection
│   │   ├── train.py                    # Training script
│   │   └── evaluate.py                 # Evaluation and anomaly scoring
│   ├── behavioral_profiling/
│   │   ├── quantum_clustering.py       # QNN for behavioral clustering
│   │   ├── preprocess.py               # Data preprocessing for profiling
│   │   ├── train.py                    # Training script
│   │   └── evaluate.py                 # Evaluation and clustering metrics
│   ├── alert_policy/
│   │   ├── quantum_alert_classifier.py # QNN for alert severity classification
│   │   ├── preprocess.py               # Data preprocessing for alerts
│   │   ├── train.py                    # Training script
│   │   └── evaluate.py                 # Evaluation and alert prioritization
│   ├── risk_scoring/
│   │   ├── quantum_risk_scorer.py      # Quantum Autoencoder for risk scoring
│   │   ├── preprocess.py               # Data preprocessing for risk scoring
│   │   ├── train.py                    # Training script
│   │   └── evaluate.py                 # Evaluation and risk score visualization
│   ├── utils/
│   │   ├── quantum_gates.py            # Quantum gate definitions (RX, RY, RZ, CNOT)
│   │   ├── logging.py                  # Logging configuration
│   │   └── visualization.py            # Plotting functions (ROC, PR curves, etc.)
├── models/
│   ├── anomaly_detection/
│   │   ├── qml_anomaly_model.pth       # Trained QAE model
│   │   ├── qml_anomaly_pca.pkl         # PCA transformer
│   │   ├── qml_anomaly_scaler.pkl      # MinMax scaler
│   │   └── qml_anomaly_metadata.json   # Model metadata
│   ├── behavioral_profiling/           # Similar structure for profiling model
│   ├── alert_policy/                   # Similar structure for alert model
│   ├── risk_scoring/                   # Similar structure for risk model
├── notebooks/
│   ├── data_exploration.ipynb          # Exploratory data analysis
│   ├── anomaly_detection.ipynb         # Demo for anomaly detection
│   ├── behavioral_profiling.ipynb      # Demo for behavioral profiling
│   ├── alert_policy.ipynb              # Demo for alert policy
│   ├── risk_scoring.ipynb              # Demo for risk scoring
├── docs/
│   └── dissertation.docx               # Full dissertation document
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
└── run_all.sh                         # Script to run all components
```

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
   - The repository includes a sample subset of the CIC-UNSW-NB15 dataset (`CICFlowMeter_out.csv`).
   - For the full dataset, download from [CIC-UNSW-NB15](https://www.unb.ca/cic/datasets/cic-unsw-nb15.html) and place in `data/raw/`.

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

```bash
cd src/anomaly_detection
python preprocess.py --input ../../data/raw/CICFlowMeter_out.csv --output ../../data/processed/raw_traffic_data.csv
python train.py --data ../../data/processed/raw_traffic_data.csv --output ../../models/anomaly_detection/
python evaluate.py --model ../../models/anomaly_detection/qml_anomaly_model.pth --data ../../data/processed/raw_traffic_data.csv
```

Similar commands apply to `behavioral_profiling`, `alert_policy`, and `risk_scoring`. Check each directory's `README.md` for specific arguments.

### Jupyter Notebooks
Explore the components interactively using the provided notebooks:
```bash
cd notebooks
jupyter notebook
```
Open `anomaly_detection.ipynb`, `behavioral_profiling.ipynb`, `alert_policy.ipynb`, or `risk_scoring.ipynb` for step-by-step demos.

## Component Details

### 1. Anomaly Detection
- **Purpose**: Detects unusual network traffic patterns (e.g., insider threats, malware) using a Quantum Autoencoder.
- **Methodology**:
  - Preprocesses CIC-UNSW-NB15 dataset (47 features reduced to 31, normalized).
  - Applies PCA to reduce to 2 dimensions for 2-qubit compatibility.
  - Uses a QAE with ZZFeatureMap encoding and RealAmplitudes ansatz (3 layers).
  - Trains for 200 epochs with MSE loss, batch size 4.
  - Evaluates anomalies via reconstruction error (threshold-based).
- **Performance**: Accuracy 95%, AUC 0.988 (see dissertation, Section 4.2.1).
- **Files**:
  - `src/anomaly_detection/quantum_autoencoder.py`: QAE implementation.
  - `models/anomaly_detection/qml_anomaly_model.pth`: Trained model.
  - `notebooks/anomaly_detection.ipynb`: Demo notebook.

### 2. Behavioral Profiling
- **Purpose**: Clusters user and entity behaviors into three groups to establish baselines for anomaly detection.
- **Methodology**:
  - Preprocesses data to extract behavioral features (e.g., login patterns, file access).
  - Uses a QNN with variational quantum circuits for clustering.
  - Trains for 200 epochs with cross-entropy loss.
  - Evaluates using precision, recall, and confusion matrix.
- **Performance**: Accuracy 96%, AUC 0.979–0.998 (see dissertation, Section 4.2.2).
- **Files**:
  - `src/behavioral_profiling/quantum_clustering.py`: QNN implementation.
  - `models/behavioral_profiling/`: Trained model and metadata.
  - `notebooks/behavioral_profiling.ipynb`: Demo notebook.

### 3. Alert Policy
- **Purpose**: Classifies alerts into Critical, High, and Medium severity for efficient incident response.
- **Methodology**:
  - Preprocesses alert data to extract severity-related features.
  - Uses a QNN with parameterized quantum circuits for classification.
  - Trains for 200 epochs with cross-entropy loss.
  - Evaluates using ROC and precision-recall curves.
- **Performance**: Accuracy 94%, AUC 0.977–0.998 (see dissertation, Section 4.2.3).
- **Files**:
  - `src/alert_policy/quantum_alert_classifier.py`: QNN implementation.
  - `models/alert_policy/`: Trained model and metadata.
  - `notebooks/alert_policy.ipynb`: Demo notebook.

### 4. Risk Scoring
- **Purpose**: Assigns risk scores (Low, Medium, High) to detected anomalies for prioritization.
- **Methodology**:
  - Preprocesses data to focus on risk-relevant features.
  - Uses a QAE to compute reconstruction errors as risk scores.
  - Normalizes scores to [0,1] using the 95th percentile.
  - Trains for 200 epochs with MSE loss.
  - Evaluates using risk score histograms and ROC curves.
- **Performance**: Accuracy 96%, AUC 0.981–0.998 (see dissertation, Section 4.2.4).
- **Files**:
  - `src/risk_scoring/quantum_risk_scorer.py`: QAE implementation.
  - `models/risk_scoring/`: Trained model and metadata.
  - `notebooks/risk_scoring.ipynb`: Demo notebook.

## Visualizations

Each component generates visualizations saved in `models/*/plots/`:
- **Training Loss Curves**: Monitors convergence.
- **Risk Score Histograms**: Shows distribution of anomaly/risk scores.
- **ROC Curves**: Evaluates discriminative power.
- **Precision-Recall Curves**: Assesses performance on imbalanced data.
- **Confusion Matrices**: Details classification performance.

Example: `models/anomaly_detection/plots/evaluation_plots.png` contains a consolidated dashboard (see dissertation, Figures 4.3, 4.4).

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure code follows PEP 8 standards and includes tests.

## Limitations

- **Simulation-Based**: Models are tested in simulated quantum environments (PyTorch-based), not on real quantum hardware, due to current NISQ limitations (see dissertation, Section 4.3.8).
- **Dataset**: Uses a subset of CIC-UNSW-NB15; real-world data may introduce noise or concept drift.
- **Interpretability**: QNNs and QAEs lack full interpretability, a challenge for cybersecurity applications.

## Future Work

- Validate models with real-world datasets.
- Improve QML interpretability using hybrid quantum-classical approaches.
- Deploy on quantum hardware as it matures.
- Extend to other cybersecurity domains (e.g., fraud detection).

See dissertation (Chapter 5) for detailed recommendations.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

- Supervisor: [Supervisor's Full Name and Title] for guidance.
- [Your University] for academic support.
- Open-source communities (e.g., Qiskit, PyTorch) for tools and resources.

For more details, refer to the dissertation in `docs/dissertation.docx`.

## Contact

For questions or issues, please open an issue on GitHub or contact [Your Email].