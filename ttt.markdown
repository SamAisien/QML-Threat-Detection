# UEBA for Threat Detection Using Quantum Machine Learning

## Overview

This repository implements a Quantum Machine Learning (QML)-based User and Entity Behavior Analytics (UEBA) framework for threat detection in cybersecurity, as detailed in the dissertation *"UEBA for Threat Detection Using Quantum Machine Learning"* (see `docs/dissertation.docx`). The framework uses quantum computing principles to enhance threat detection through four components: Anomaly Detection, Behavioral Profiling, Alert Policy, and Risk Scoring, achieving accuracies of 94–96% and AUC values of 0.977–0.998 (Chapter 4).

### Components
1. **Anomaly Detection**: Identifies unusual patterns using a Quantum Autoencoder (QAE).
2. **Behavioral Profiling**: Clusters behaviors to establish baselines using a Quantum Neural Network (QNN).
3. **Alert Policy**: Prioritizes alerts by severity (Critical, High, Medium) using a QNN.
4. **Risk Scoring**: Quantifies anomaly risk levels (Low, Medium, High) using a QAE.

## Repository Structure

The structure is based on the workflow shown in the dissertation figures (Chapter 3, Figures 3.1–3.36).

```
├── data/
│   ├── raw/
│   │   └── CICFlowMeter_out.csv        # Raw dataset (CIC-UNSW-NB15 subset)
│   ├── processed/
│   │   ├── anomaly_detection.csv       # Preprocessed data for anomaly detection
│   │   ├── behavioral_profiling.csv    # Preprocessed data for behavioral profiling
│   │   ├── alert_policy.csv            # Preprocessed data for alert policy
│   │   └── risk_scoring.csv            # Preprocessed data for risk scoring
├── models/
│   ├── anomaly_detection/
│   │   ├── qml_anomaly_model.pth       # Trained Quantum Autoencoder model
│   │   ├── qml_anomaly_pca.pkl         # PCA transformer
│   │   ├── qml_anomaly_scaler.pkl      # MinMax scaler
│   │   ├── qml_anomaly_metadata.json   # Model metadata (Figure 3.8)
│   │   └── plots/
│   │       ├── evaluation_plots.png    # Consolidated dashboard (Figure 3.9)
│   │       └── training_loss.png       # Training loss curve
│   ├── behavioral_profiling/
│   │   ├── qml_profiling_model.pth     # Trained QNN model
│   │   ├── qml_profiling_metadata.json # Model metadata
│   │   └── plots/
│   │       ├── evaluation_plots.png    # Consolidated dashboard (Figure 3.18)
│   │       └── training_loss.png       # Training loss curve
│   ├── alert_policy/
│   │   ├── qml_alert_model.pth         # Trained QNN model
│   │   ├── qml_alert_metadata.json     # Model metadata
│   │   └── plots/
│   │       ├── evaluation_plots.png    # Consolidated dashboard (Figure 3.27)
│   │       └── training_loss.png       # Training loss curve
│   ├── risk_scoring/
│   │   ├── qml_risk_model.pth          # Trained Quantum Autoencoder model
│   │   ├── qml_risk_metadata.json      # Model metadata
│   │   └── plots/
│   │       ├── evaluation_plots.png    # Consolidated dashboard (Figure 3.36)
│   │       └── training_loss.png       # Training loss curve
├── src/
│   ├── anomaly_detection/
│   │   ├── environment_setup.py        # Environment setup (Figure 3.1)
│   │   ├── preprocess.py               # Data preprocessing (Figure 3.2)
│   │   ├── quantum_gates.py            # Quantum gate definitions (Figure 3.3)
│   │   ├── quantum_circuit.py          # Quantum circuit design (Figure 3.4)
│   │   ├── quantum_autoencoder.py      # QAE implementation (Figure 3.5)
│   │   ├── train.py                    # Training script (Figure 3.6)
│   │   ├── evaluate.py                 # Evaluation script (Figure 3.7)
│   │   └── save_model.py               # Model saving (Figure 3.8)
│   ├── behavioral_profiling/
│   │   ├── preprocess.py               # Data preprocessing (Figure 3.11)
│   │   ├── quantum_circuit.py          # Quantum circuit design (Figure 3.13)
│   │   ├── quantum_clustering.py       # QNN implementation (Figure 3.14)
│   │   ├── train.py                    # Training script (Figure 3.15)
│   │   ├── evaluate.py                 # Evaluation script (Figure 3.16)
│   │   └── visualization.py            # Visualization script (Figure 3.18)
│   ├── alert_policy/
│   │   ├── preprocess.py               # Data preprocessing (Figure 3.20)
│   │   ├── quantum_circuit.py          # Quantum circuit design (Figure 3.22)
│   │   ├── quantum_alert_classifier.py # QNN implementation (Figure 3.23)
│   │   ├── train.py                    # Training script (Figure 3.24)
│   │   ├── evaluate.py                 # Evaluation script (Figure 3.25)
│   │   └── visualization.py            # Visualization script (Figure 3.27)
│   ├── risk_scoring/
│   │   ├── preprocess.py               # Data preprocessing (Figure 3.30)
│   │   ├── quantum_circuit.py          # Quantum circuit design (Figure 3.31)
│   │   ├── quantum_autoencoder.py      # QAE implementation (Figure 3.32)
│   │   ├── train.py                    # Training script (Figure 3.33)
│   │   ├── risk_scoring.py             # Risk scoring (Figure 3.34)
│   │   ├── evaluate.py                 # Evaluation script (Figure 3.35)
│   │   └── visualization.py            # Visualization script (Figure 3.36)
├── notebooks/
│   ├── data_exploration.ipynb          # Exploratory data analysis
│   ├── anomaly_detection.ipynb         # Demo for anomaly detection
│   ├── behavioral_profiling.ipynb      # Demo for behavioral profiling
│   ├── alert_policy.ipynb              # Demo for alert policy
│   ├── risk_scoring.ipynb              # Demo for risk scoring
├── docs/
│   └── dissertation.docx               # Full dissertation document
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── run_all.sh                          # Script to run all components
```

## Prerequisites

- **Python**: 3.10 or higher
- **Hardware**: CUDA-enabled GPU recommended
- **Dataset**: CIC-UNSW-NB15 (subset in `data/raw/CICFlowMeter_out.csv`)

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
   - A subset of CIC-UNSW-NB15 is provided in `data/raw/CICFlowMeter_out.csv`.
   - For the full dataset, download from [CIC-UNSW-NB15](https://www.unb.ca/cic/datasets/cic-unsw-nb15.html) and place in `data/raw/`.

## Usage

Each component can be run independently or sequentially using the `run_all.sh` script, following the workflow in the dissertation figures.

### Running All Components
```bash
bash run_all.sh
```

This script:
1. Preprocesses data for each component, saving outputs in `data/processed/`.
2. Trains models, saving them in `models/*/`.
3. Evaluates models and generates visualizations in `models/*/plots/`.

### Running Individual Components
Each component’s directory contains scripts corresponding to the dissertation figures. For example, for Anomaly Detection:

```bash
cd src/anomaly_detection
python environment_setup.py  # Sets up environment (Figure 3.1)
python preprocess.py --input ../../data/raw/CICFlowMeter_out.csv --output ../../data/processed/anomaly_detection.csv  # Preprocessing (Figure 3.2)
python train.py --data ../../data/processed/anomaly_detection.csv --output ../../models/anomaly_detection/  # Training (Figure 3.6)
python evaluate.py --model ../../models/anomaly_detection/qml_anomaly_model.pth --data ../../data/processed/anomaly_detection.csv  # Evaluation (Figure 3.7)
python save_model.py --model ../../models/anomaly_detection/qml_anomaly_model.pth  # Save model (Figure 3.8)
```

Repeat for `behavioral_profiling`, `alert_policy`, and `risk_scoring`.

### Jupyter Notebooks
Explore interactively:
```bash
cd notebooks
jupyter notebook
```
Open `anomaly_detection.ipynb`, `behavioral_profiling.ipynb`, `alert_policy.ipynb`, or `risk_scoring.ipynb`.

## Component Details

### 1. Anomaly Detection
- **Purpose**: Detects unusual patterns using a Quantum Autoencoder.
- **Workflow** (Figures 3.1–3.9):
  - Environment setup with Python 3.10, PyTorch, etc. (`environment_setup.py`, Figure 3.1).
  - Preprocesses CIC-UNSW-NB15 dataset (47 features to 31, normalized) (`preprocess.py`, Figure 3.2).
  - Defines quantum gates (RX, RY, RZ, CNOT) (`quantum_gates.py`, Figure 3.3).
  - Designs quantum circuit with ZZFeatureMap (`quantum_circuit.py`, Figure 3.4).
  - Implements QAE (`quantum_autoencoder.py`, Figure 3.5).
  - Trains for 200 epochs with MSE loss (`train.py`, Figure 3.6).
  - Evaluates using reconstruction error (`evaluate.py`, Figure 3.7).
  - Saves model (`save_model.py`, Figure 3.8).
- **Performance**: Accuracy 95%, AUC 0.988 (Section 4.2.1).
- **Files**:
  - `src/anomaly_detection/`: Scripts as above.
  - `models/anomaly_detection/qml_anomaly_model.pth`: Trained model.
  - `notebooks/anomaly_detection.ipynb`: Demo.

### 2. Behavioral Profiling
- **Purpose**: Clusters behaviors into three groups using a QNN.
- **Workflow** (Figures 3.11–3.18):
  - Preprocesses data for behavioral features (`preprocess.py`, Figure 3.11).
  - Designs quantum circuit (`quantum_circuit.py`, Figure 3.13).
  - Implements QNN (`quantum_clustering.py`, Figure 3.14).
  - Trains for 200 epochs (`train.py`, Figure 3.15).
  - Evaluates with precision, recall (`evaluate.py`, Figure 3.16).
  - Visualizes results (`visualization.py`, Figure 3.18).
- **Performance**: Accuracy 96%, AUC 0.979–0.998 (Section 4.2.2).
- **Files**:
  - `src/behavioral_profiling/`: Scripts as above.
  - `models/behavioral_profiling/qml_profiling_model.pth`: Trained model.
  - `notebooks/behavioral_profiling.ipynb`: Demo.

### 3. Alert Policy
- **Purpose**: Classifies alerts into Critical, High, Medium using a QNN.
- **Workflow** (Figures 3.20–3.27):
  - Preprocesses alert data (`preprocess.py`, Figure 3.20).
  - Designs quantum circuit (`quantum_circuit.py`, Figure 3.22).
  - Implements QNN (`quantum_alert_classifier.py`, Figure 3.23).
  - Trains for 200 epochs (`train.py`, Figure 3.24).
  - Evaluates with ROC curves (`evaluate.py`, Figure 3.25).
  - Visualizes results (`visualization.py`, Figure 3.27).
- **Performance**: Accuracy 94%, AUC 0.977–0.998 (Section 4.2.3).
- **Files**:
  - `src/alert_policy/`: Scripts as above.
  - `models/alert_policy/qml_alert_model.pth`: Trained model.
  - `notebooks/alert_policy.ipynb`: Demo.

### 4. Risk Scoring
- **Purpose**: Assigns risk scores (Low, Medium, High) using a QAE.
- **Workflow** (Figures 3.30–3.36):
  - Preprocesses data (`preprocess.py`, Figure 3.30).
  - Designs quantum circuit (`quantum_circuit.py`, Figure 3.31).
  - Implements QAE (`quantum_autoencoder.py`, Figure 3.32).
  - Trains for 200 epochs (`train.py`, Figure 3.33).
  - Computes risk scores (`risk_scoring.py`, Figure 3.34).
  - Evaluates with ROC curves (`evaluate.py`, Figure 3.35).
  - Visualizes results (`visualization.py`, Figure 3.36).
- **Performance**: Accuracy 96%, AUC 0.981–0.998 (Section 4.2.4).
- **Files**:
  - `src/risk_scoring/`: Scripts as above.
  - `models/risk_scoring/qml_risk_model.pth`: Trained model.
  - `notebooks/risk_scoring.ipynb`: Demo.

## Visualizations

Visualizations are saved in `models/*/plots/`, matching the dissertation figures:
- **Training Loss Curves**: Monitors convergence.
- **Risk Score Histograms**: Shows score distributions.
- **ROC Curves**: Evaluates discriminative power.
- **Precision-Recall Curves**: Assesses performance.
- **Confusion Matrices**: Details classification performance.

Example: `models/anomaly_detection/plots/evaluation_plots.png` (Figure 3.9).

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please follow PEP 8 standards and include tests.

## Limitations

- **Simulation-Based**: Tested in simulated quantum environments (PyTorch), not on real hardware (Section 4.3.8).
- **Dataset**: Uses a CIC-UNSW-NB15 subset; real-world data may introduce noise.
- **Interpretability**: QNNs and QAEs lack full interpretability.

## Future Work

- Validate with real-world datasets.
- Improve QML interpretability.
- Deploy on quantum hardware.
- Extend to other cybersecurity domains (Chapter 5).

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

- Supervisor: [Supervisor's Full Name and Title] for guidance.
- [Your University] for academic support.
- Open-source communities (e.g., Qiskit, PyTorch) for tools.

For more details, refer to `docs/dissertation.docx`.

## Contact

For questions, open an issue on GitHub or contact [Your Email].