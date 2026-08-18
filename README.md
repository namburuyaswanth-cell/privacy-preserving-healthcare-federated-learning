# Privacy-Preserving Healthcare Device Authentication via Edge-AI and Federated Learning

A final-year B.Tech project focused on privacy-preserving healthcare AI using Federated Learning, Edge-AI, secure device/hospital authentication, and medical image classification.

## Overview

The project proposes a healthcare framework in which sensitive medical data remains at local hospital/client nodes while model updates are collaboratively aggregated through Federated Learning.

The design combines:

- Federated Learning (FL)
- Edge-AI
- Centralized Authorization Authority (CAA)
- HMAC-SHA256 token authentication
- OSDSTran (Optimized Spatial Densely Connected Swin Transformer)
- Prairie Dog Optimization Algorithm (PDOA)
- FedAvg aggregation
- Chest X-ray disease classification

## Problem

Centralized healthcare AI can require sensitive patient data to be transferred to a central location. The project addresses privacy, unauthorized participation, computational cost, and model-performance concerns by keeping raw data local and allowing only authenticated participants to contribute model updates.

## Proposed Workflow

1. Hospitals/devices register with the Centralized Authorization Authority.
2. Authentication tokens are generated using HMAC-SHA256.
3. Medical images are preprocessed locally.
4. Local models are trained at participating hospitals.
5. OSDSTran is described in the report as the proposed classification architecture.
6. PDOA is used for hyperparameter optimization.
7. Only model updates are sent to the federated server.
8. Validated updates are aggregated using FedAvg.
9. The global model is redistributed to participating clients.
10. The model produces disease-classification results.

## Reported Results

According to the supplied project report:

| Metric | Proposed OSDSTran |
|---|---:|
| Accuracy | 98.88% |
| Precision | 98.65% |
| Recall | 98.75% |
| F1-score | 98.81% |
| AUC | 0.9883 |
| PRC | 0.9881 |
| Average computation time | 10.85 s |
| Latency | 0.008 sec/packet |
| Response time | approximately 5.5–6.1 s |

The report compares the proposed model against VGG, ResNet, Xception, and ViT.

## Repository Contents

```text
Privacy-Preserving-Healthcare-EdgeAI-FederatedLearning/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── docs/
│   ├── Project_Report.pdf
│   ├── Project_Presentation.pptx
│   ├── architecture.md
│   ├── results.md
│   ├── publication-status.md
│   └── diagrams/
├── src/
│   └── federated_cnn_demo.py
├── notebooks/
│   └── README.md
└── dataset/
    └── README.md
```

## Important Implementation Note

The supplied report describes OSDSTran + PDOA as the proposed research methodology. However, the implementation chapter included in the supplied report contains a TensorFlow CNN implementation with Conv2D/MaxPooling layers, HMAC verification, client splitting, and FedAvg. The repository therefore labels the included code as a **federated CNN demonstration** rather than falsely presenting it as a complete OSDSTran implementation.

The supplied report should remain the authoritative source for the project's reported research results.

## Dataset

The report references the NIH Chest X-ray dataset on Kaggle. The dataset itself is intentionally not included in this repository because of its size and licensing/distribution considerations. See `dataset/README.md`.

## Running the Included Demonstration

The included source is based on the implementation chapter of the supplied report and expects:

```text
dataset/
├── NORMAL/
└── PNEUMONIA/
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python src/federated_cnn_demo.py
```

For a real run, set a secret key through the environment:

```bash
# Windows PowerShell
$env:HMAC_SECRET_KEY="replace-with-your-own-secret"

# Linux/macOS
export HMAC_SECRET_KEY="replace-with-your-own-secret"
```

Do not commit real secret keys to GitHub.

## Academic Documents

- [Project Report](docs/Project_Report.pdf)
- [Project Presentation](docs/Project_Presentation.pptx)
- [Architecture Notes](docs/architecture.md)
- [Results Summary](docs/results.md)
- [Publication Status](docs/publication-status.md)

## Future Scope

The supplied report proposes blockchain-based security enhancements, larger and more diverse medical datasets, reduced computational and communication cost, integration with real-time monitoring/wearable devices, and multi-disease healthcare applications.

## Disclaimer

This repository is an academic final-year project. It is not a clinical diagnostic system and should not be used for medical diagnosis or treatment decisions.

## Frontend and Backend

The repository includes a responsive web dashboard built around the project workflow.

```text
backend/
├── app.py
└── requirements.txt

frontend/
├── index.html
├── css/style.css
└── js/app.js
```

### Backend

The Flask backend exposes:

- `GET /api/health`
- `GET /api/project`
- `GET /api/federated/status`
- `POST /api/authenticate`
- `POST /api/predict`

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

Set an HMAC secret before starting:

```powershell
$env:HMAC_SECRET_KEY="your-development-secret"
```

or on Linux/macOS:

```bash
export HMAC_SECRET_KEY="your-development-secret"
```

Start the application:

```bash
python backend/app.py
```

Then open `http://127.0.0.1:5000`.

### Frontend

The dashboard provides project metrics, federated client status, an HMAC authentication demonstration, and a chest X-ray upload interface.

Real image inference is enabled only when a trained Keras model is placed at:

```text
models/healthcare_model.keras
```

The supplied project files did not include a trained model file, so the backend explicitly reports when the model is unavailable instead of generating a fake prediction.
