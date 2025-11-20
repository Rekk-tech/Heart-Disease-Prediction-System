# 🫀 Heart Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)

> **An intelligent heart disease prediction system using Machine Learning and Ensemble Methods**

## 🎯 Overview

This project builds a comprehensive heart disease prediction system using 10 different machine learning algorithms on the Cleveland Heart Disease dataset. The system achieves an average AUC of 0.94 with hyperparameter optimization and detailed experiment tracking.

**🌐 Live Demo:** https://heart-disease-prediction-systems.streamlit.app/


---

## 🚀 Quick Start

### Windows
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the application
streamlit run app\streamlit_app.py

# Access at http://localhost:8501
```

### Linux/Mac
```bash
# Activate virtual environment
source venv/bin/activate

# Run the application
./run.sh

# Access at http://localhost:8501
```

---

## 📁 Project Structure

```
heart-disease-diagnosis-main/
├── 📱 app/
│   ├── streamlit_app.py          # Main web interface
│   └── model_functions.py        # Feature engineering classes
├── 🔧 src/
│   ├── pipeline.py               # Main ML pipeline
│   ├── model_functions.py        # Feature transformers
│   └── utils/
│       └── app_utils.py          # Utility functions
├── 📜 scripts/
│   ├── experiment_manager.py     # Experiment management
│   └── train_models.py           # Training and optimization
├── 📊 data/
│   ├── raw/                      # Raw data
│   ├── processed/                # Processed data
│   └── patient_history.json      # Patient history
├── 🤖 models/
│   └── saved_models/latest/      # Trained models
├── 🧪 experiments/
│   ├── experiment_log.json       # Logs of 40+ experiments
│   ├── logs/                     # Training logs
│   └── results/                  # Results and predictions
├── 📓 notebooks/                 # Jupyter notebooks
├── 📈 results/                   # Analysis results
└── ⚙️ .streamlit/                # Streamlit configuration
```

---

## 🎯 Algorithms Evaluated

The system uses 10 machine learning algorithms:

1. **Logistic Regression** - Basic linear model
2. **Random Forest** - Ensemble of decision trees
3. **K-Nearest Neighbors** - Instance-based learning
4. **Decision Tree** - Single decision tree
5. **AdaBoost** - Adaptive boosting
6. **Gradient Boosting** - Sequential ensemble
7. **XGBoost** - Extreme gradient boosting
8. **LightGBM** - Light gradient boosting
9. **Support Vector Machine** - Support vector machine
10. **Ensemble Voting** - Meta-classifier

---

## 🏆 Results

| Model                  | Accuracy | Precision | Recall | F1-Score | AUC    | Status |
| ---------------------- | -------- | --------- | ------ | -------- | ------ | ------ |
| 🥇 **Gradient Boosting** | **91.8%** | **89.7%** | **92.9%** | **91.2%** | **95.5%** | ✅ Best |
| 🥈 K-Nearest Neighbors  | 90.2%    | 86.7%     | 92.9%  | 89.7%    | 95.4%  | ✅ Excellent |
| 🥉 XGBoost              | 90.2%    | 86.7%     | 92.9%  | 89.7%    | 94.4%  | ✅ Very Good |
| Logistic Regression     | 88.5%    | 83.9%     | 92.9%  | 88.1%    | 95.7%  | ✅ Good |
| LightGBM               | 86.9%    | 83.3%     | 89.3%  | 86.2%    | 94.7%  | ✅ Good |
| AdaBoost               | 85.2%    | 80.6%     | 89.3%  | 84.8%    | 94.3%  | ✅ Good |
| Random Forest          | 83.6%    | 82.1%     | 82.1%  | 82.1%    | 93.6%  | ✅ Stable |
| Support Vector Machine | 83.6%    | 82.1%     | 82.1%  | 82.1%    | 95.6%  | ✅ Reliable |
| Decision Tree          | 83.6%    | 82.1%     | 82.1%  | 82.1%    | 88.6%  | ✅ Baseline |
| **Ensemble Average**   | **87.0%** | **84.1%** | **87.1%** | **85.5%** | **94.0%** | 🎯 **Target** |

🏆 **Overall Results:** Average AUC 94.0% | Best Model: Gradient Boosting

---

## Application Features

### 🩺 1. Patient Diagnosis

- 📝 Input form with clinical parameter validation
- 🔮 Real-time predictions from 10 models
- 🗳️ Majority voting with confidence scores
- 📊 Visual risk assessment
- 💊 Personalized treatment recommendations

### 📈 2. Model Analysis

- 📋 Comprehensive performance metrics
- 🔄 Cross-validation vs test set comparison
- ⚙️ Detailed model configuration
- 🎯 Confusion matrix and ROC curves

### 🔍 3. Feature Importance Analysis

- 🧠 SHAP-style feature contribution
- 📊 Model-specific importance ranking
- 🏥 Clinical interpretation guidance
- 📉 Input contribution visualization

### 🧪 4. Experiment Tracking

- 📚 Hyperparameter search history (40+ experiments)
- 🔄 Reproducible experiment logs
- 🔧 Performance comparison tools
- 📊 HTML/PDF report export

### 📋 5. History & Reporting

- 🗃️ Patient prediction history storage
- 📄 Automated PDF report generation
- 💾 CSV/Excel data export
- 📈 Usage statistics

---


### Setup

```bash
# Clone repository
git clone https://github.com/Rekk-tech/Heart-Disease-Prediction-System.git
cd Heart-Disease-Prediction-System

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎮 Usage

### Local Deployment

```bash
# Windows
streamlit run app/streamlit_app.py




### Model Training

```bash
# Run hyperparameter tuning
python scripts/train_models.py

# Results saved in experiments/
```

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter lab notebooks/

# Available notebooks:
# - 01_AdaBoost_Model.ipynb
# - 02_Create_Datasets.ipynb
# - 03_Deploy_Streamlit.ipynb
```

---



## ⚠️ Limitations & Disclaimer

🎓 **FOR EDUCATIONAL/RESEARCH PURPOSES ONLY**

This system is NOT designed for clinical use. Always consult qualified healthcare professionals for diagnosis and treatment.

## 📄 License

📚 **For educational and research use only.** See individual package licenses for dependencies.

---


- 🐙 **GitHub:** https://github.com/Rekk-tech/Heart-Disease-Prediction-System
- 🌐 **Demo:** https://heart-disease-prediction-systems.streamlit.app/






