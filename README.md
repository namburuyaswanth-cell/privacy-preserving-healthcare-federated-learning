# Privacy-Preserving Healthcare Device Authentication via Edge-AI and Federated Learning

> A privacy-aware healthcare AI project combining Edge-AI, Federated Learning, HMAC-SHA256 device authentication, and CNN-based chest X-ray classification.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-CNN-green)
![Federated Learning](https://img.shields.io/badge/Federated%20Learning-FedAvg-purple)
![Security](https://img.shields.io/badge/Security-HMAC--SHA256-red)

---

## 📌 Project Overview

Healthcare AI applications often work with highly sensitive medical data. Centralizing medical images for model training can introduce privacy and security concerns.

This project explores a **privacy-preserving healthcare AI workflow** in which medical images can remain at local healthcare/edge clients while machine-learning models are trained and model updates are collaboratively aggregated through a Federated Learning workflow.

The system also includes **HMAC-SHA256-based healthcare device authentication** to demonstrate how participating clients can be authenticated before taking part in the workflow.

The project combines:

- Edge-AI
- Federated Learning
- HMAC-SHA256 authentication
- FedAvg aggregation
- CNN-based medical image classification
- Chest X-ray classification
- Model evaluation and visualization
- Web-based healthcare demonstration dashboard

---

# 🎯 Objectives

The major objectives of this project are:

1. Develop a privacy-aware healthcare AI workflow.
2. Demonstrate Federated Learning for collaborative model training.
3. Keep sensitive medical images at local client/edge nodes during the intended workflow.
4. Implement HMAC-SHA256-based device authentication.
5. Develop a CNN-based chest X-ray classification model.
6. Evaluate the model using standard machine-learning metrics.
7. Provide a simple web dashboard for demonstrating authentication and image prediction.

---

# 🏗️ System Architecture

The overall workflow can be represented as:

```text
                  ┌─────────────────────────┐
                  │   Hospital / Edge       │
                  │        Client           │
                  │                         │
                  │  Local Medical Images   │
                  │  Local Model Training   │
                  └────────────┬────────────┘
                               │
                               │ Authentication
                               ▼
                  ┌─────────────────────────┐
                  │   CAA Authentication    │
                  │                         │
                  │     HMAC-SHA256         │
                  └────────────┬────────────┘
                               │
                               │ Authenticated
                               │ Model Updates
                               ▼
                  ┌─────────────────────────┐
                  │    Federated Server     │
                  │                         │
                  │    FedAvg Aggregation   │
                  └────────────┬────────────┘
                               │
                               │ Global Model
                               ▼
                  ┌─────────────────────────┐
                  │   Participating Clients │
                  │                         │
                  │  Updated Global Model   │
                  └─────────────────────────┘

🔐 Privacy by Design

The intended Federated Learning workflow avoids directly centralizing raw medical images for collaborative training.

Instead:

                              Medical Data
                                   ↓
                      Local Hospital / Edge Client
                                   ↓
                         Local Model Training
                                   ↓
                              Model Update
                                   ↓
                             Authentication
                                   ↓
                            Federated Server
                                   ↓
                           FedAvg Aggregation
                                   ↓
                             Global Model

This demonstrates the basic concept of collaborative machine learning while reducing the need to directly share raw training images.


🔑 Device Authentication

The project uses HMAC-SHA256 in the supplied authentication workflow.

A healthcare device provides a device identifier, which is processed by the authentication mechanism before the client participates in the demonstration workflow.

Example:
                           Hospital / Device ID
                                     ↓
                                HMAC-SHA256
                                     ↓
                            Authentication Token
                                     ↓
                            Authenticated Client

The web dashboard provides a simple interface for testing device authentication.


🧠 Machine Learning Pipeline

The medical-image classification component uses a Convolutional Neural Network (CNN).

The dataset contains two classes:
dataset/
├── NORMAL/
└── PNEUMONIA/

The general machine-learning workflow is:

                  Chest X-ray Images
                         ↓
                 Image Preprocessing
                         ↓
                   Dataset Split
                         ↓
                    CNN Training
                         ↓
              Federated Training Workflow
                         ↓
                 Model Aggregation
                         ↓
                     Evaluation
                         ↓
                     Prediction



📊 Model Performance

The current experimental run produced the following results:

Metric	Result
Accuracy	87.50%
Precision	88.89%
Recall	84.21%
F1 Score	86.49%

These values represent the current local experimental result and may vary depending on dataset split, training configuration, random initialization, and execution environment.



📈 Evaluation Results

The project generates evaluation visualizations including a confusion matrix and federated accuracy plot.

- Confusion Matrix

- Federated Accuracy



🖥️ Healthcare Demonstration Dashboard

The project includes a web-based dashboard named HealthFL.

The dashboard demonstrates:

- Healthcare device authentication
- Federated network status
- Privacy-by-design concept
- System architecture
- Chest X-ray image upload
- Model prediction
- Prediction probability
- Evaluation metrics

Example workflow:

Open HealthFL Dashboard
        ↓
Check Server Status
        ↓
Authenticate Healthcare Device
        ↓
Select Chest X-ray
        ↓
Upload Image
        ↓
CNN Model Prediction
        ↓
Display Prediction



📁 Project Structure

privacy-preserving-healthcare-federated-learning/
│
├── backend/
│   └── app.py
│
├── dataset/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── docs/
│
├── frontend/
│   └── index.html
│
├── models/
│   └── healthcare_model.keras
│
├── notebooks/
│
├── results/
│   ├── confusion_matrix.png
│   └── federated_accuracy.png
│
├── src/
│   └── federated_cnn_demo.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

The dataset, virtual environment, uploaded images, and trained model files are excluded from version control where appropriate. The trained model is generated locally after training.



🛠️ Technologies Used :

Programming
- Python
- HTML
- CSS
- JavaScript

Machine Learning
- TensorFlow
- Keras
- Scikit-learn
- NumPy
- OpenCV
- Matplotlib

AI / ML Concepts
- Convolutional Neural Networks
- Federated Learning
- Federated Averaging (FedAvg)
- Edge-AI
- Medical Image Classification
- Model Evaluation

Security
- HMAC-SHA256
- Healthcare Device Authentication

Development Tools
- Git
- GitHub
- Python Virtual Environment
- Visual Studio Code / Command Prompt



🚀 Installation and Setup --

1. Clone the Repository:

       git clone https://github.com/namburuyaswanth-cell/privacy-preserving-healthcare-federated-learning.git

Move into the project directory:

       cd privacy-preserving-healthcare-federated-learning

2. Create a Python Virtual Environment

The project was tested using Python 3.11.

On Windows:

       py -3.11 -m venv venv

Activate the environment:

       venv\Scripts\activate

You should see:

       (venv)

at the beginning of your terminal prompt.


📦 3. Install Dependencies

Run:

       pip install -r requirements.txt


🧪 4. Run the Federated CNN Experiment

From the project root directory:

       python src\federated_cnn_demo.py

The script performs the training/evaluation workflow and generates evaluation outputs.

The trained model is saved locally under:

       models/healthcare_model.keras

Evaluation files are generated under:

       results/


🌐 5. Run the Healthcare Dashboard

First, set the HMAC secret for local development.

Windows CMD:

        set HMAC_SECRET_KEY=my-local-development-secret-123

Then start the backend:

        python backend\app.py

The Flask development server should start at:

        http://127.0.0.1:5000

Open the address in your browser.



🔐 6. Test Device Authentication

Open the HealthFL dashboard.

Navigate to the Device Authentication section.

Enter a device ID such as:

        Hospital-04

Click:

        Authenticate

A successful authentication response contains information such as:

        authenticated: true
        device_id: Hospital-04
        token_type: HMAC-SHA256



🩻 7. Test Chest X-ray Prediction

After starting the dashboard:

1. Navigate to Chest X-ray prediction.
2. Click Choose File.
3. Select a PNG, JPG, or JPEG chest X-ray image.
4. Click Upload & Predict.
5. The application processes the image using the trained model.
6. The predicted class and probability are displayed.

Possible classes:

-- NORMAL
-- PNEUMONIA

The prediction functionality requires the locally generated model:

    -- models/healthcare_model.keras



🔬 Experimental Workflow

The project was developed and tested through the following stages:


Dataset Preparation
        ↓
Image Preprocessing
        ↓
CNN Model Development
        ↓
Federated Learning Workflow
        ↓
Model Aggregation
        ↓
Model Evaluation
        ↓
Model Saving
        ↓
Flask Backend
        ↓
HealthFL Frontend
        ↓
Device Authentication
        ↓
Chest X-ray Prediction



📋 Key Learning Outcomes --

Through this project, the following concepts were explored:

- Data preprocessing for medical images
- CNN-based image classification
- Federated Learning concepts
- FedAvg model aggregation
- Edge-AI architecture
- Secure device authentication
- HMAC-SHA256
- Model evaluation
- Confusion matrix analysis
- Precision, Recall and F1 Score
- Flask backend development
- Frontend-backend integration
- Git and GitHub project management



🔮 Future Improvements --

The project can be extended with:

- Differential Privacy
- Secure Aggregation
- Improved CNN and Transformer architectures
- Larger and more diverse medical datasets
- Advanced hyperparameter optimization
- Real multi-client federated training
- Client-level model monitoring
- Experiment tracking
- Cloud/edge deployment
- Containerized deployment
- Improved authentication and key management


⚠️ Disclaimer

This project is an academic final-year project developed for educational and research demonstration purposes.

It is not intended for:

- Clinical diagnosis
- Real-world medical decision-making
- Production healthcare deployment
- Handling real patient information
- Replacing professional medical advice

The reported machine-learning results are experimental and should not be interpreted as clinical performance.



⭐ Project Highlights --

Privacy + AI + Security + Healthcare

This project demonstrates how multiple areas of computer science can be combined into a single practical system:


                   HEALTHCARE AI
                        │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
    Edge-AI     Federated    Security
                 Learning       │
        │           │       HMAC-SHA256
        │           │           │
        └───────────┼───────────┘
                    ▼
             Medical Image AI
                    │
                    ▼
                 HealthFL
              Web Dashboard
