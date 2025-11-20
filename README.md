# 🫀 Heart Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)

> **Hệ thống dự đoán bệnh tim thông minh sử dụng Machine Learning và Ensemble Methods**

## 🎯 Tổng quan

Dự án này xây dựng một hệ thống dự đoán bệnh tim toàn diện sử dụng 10 thuật toán machine learning khác nhau trên bộ dữ liệu Cleveland Heart Disease. Hệ thống đạt AUC trung bình 0.94 với tối ưu hóa siêu tham số và theo dõi thí nghiệm chi tiết.

**🌐 Demo trực tuyến:** https://heart-disease-prediction-systems.streamlit.app/

**👥 Nhóm phát triển:** Dũng, Anh, Vinh, Hằng, Huy | AIO2025 VietAI

---

## 🚀 Khởi chạy nhanh

### Windows
```powershell
# Kích hoạt môi trường
.\venv\Scripts\Activate.ps1

# Chạy ứng dụng
streamlit run app\streamlit_app.py

# Truy cập tại http://localhost:8502
```

### Linux/Mac
```bash
# Kích hoạt môi trường
source venv/bin/activate

# Chạy ứng dụng
./run.sh

# Truy cập tại http://localhost:8502
```

---

## 📁 Cấu trúc dự án

```
heart-disease-diagnosis-main/
├── 📱 app/
│   ├── streamlit_app.py          # Giao diện web chính
│   └── model_functions.py        # Feature engineering classes
├── 🔧 src/
│   ├── pipeline.py               # Pipeline ML chính
│   ├── model_functions.py        # Feature transformers
│   └── utils/
│       └── app_utils.py          # Hàm tiện ích
├── 📜 scripts/
│   ├── experiment_manager.py     # Quản lý thí nghiệm
│   └── train_models.py           # Huấn luyện và tối ưu
├── 📊 data/
│   ├── raw/                      # Dữ liệu gốc
│   ├── processed/                # Dữ liệu đã xử lý
│   └── patient_history.json      # Lịch sử bệnh nhân
├── 🤖 models/
│   └── saved_models/latest/      # Models đã huấn luyện
├── 🧪 experiments/
│   ├── experiment_log.json       # Log 40+ thí nghiệm
│   ├── logs/                     # Log huấn luyện
│   └── results/                  # Kết quả và dự đoán
├── 📓 notebooks/                 # Jupyter notebooks
├── 📈 results/                   # Kết quả phân tích
└── ⚙️ .streamlit/                # Cấu hình Streamlit
```

---

## Methodology

### Dataset

**Source:** Cleveland Heart Disease Dataset (UCI Machine Learning Repository)  
**Samples:** 303 patients  
**Features:** 13 clinical attributes  
**Target:** Binary classification (0 = Healthy, 1 = Disease)

### 🎯 Thuật toán được đánh giá

Hệ thống sử dụng 10 thuật toán machine learning:

1. **Logistic Regression** - Mô hình tuyến tính cơ bản
2. **Random Forest** - Ensemble cây quyết định
3. **K-Nearest Neighbors** - Học dựa trên láng giềng
4. **Decision Tree** - Cây quyết định đơn
5. **AdaBoost** - Adaptive boosting
6. **Gradient Boosting** - Sequential ensemble
7. **XGBoost** - Extreme gradient boosting
8. **LightGBM** - Light gradient boosting
9. **Support Vector Machine** - Máy vector hỗ trợ
10. **Ensemble Voting** - Meta-classifier tổng hợp

### Hyperparameter Optimization

- **Framework:** Optuna (Tree-structured Parzen Estimator)
- **Trials:** 100 per model
- **Validation:** 5-fold stratified cross-validation
- **Metric:** F1-score (macro average)

### Evaluation

- **Cross-validation AUC:** Performance during training
- **Test AUC:** Held-out test set performance
- **Majority Voting:** Final prediction from ensemble

---

## Application Features

## Results

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

🏆 **Kết quả tổng thể:** AUC trung bình 94.0% | Mô hình tốt nhất: Gradient Boosting

---

## Application Features

### 🩺 1. Chẩn đoán bệnh nhân

- 📝 Form nhập liệu với validation thông số lâm sàng
- 🔮 Dự đoán real-time từ 10 models
- 🗳️ Majority voting với điểm tin cậy
- 📊 Visualize đánh giá rủi ro
- 💊 Đề xuất cá nhân hóa

### 📈 2. Phân tích mô hình

- 📋 Metrics hiệu suất toàn diện
- 🔄 So sánh cross-validation vs test set
- ⚙️ Chi tiết cấu hình mô hình
- 🎯 Confusion matrix và ROC curves

### 🔍 3. Phân tích tầm quan trọng

- 🧠 SHAP-style feature contribution
- 📊 Ranking tầm quan trọng theo mô hình
- 🏥 Hướng dẫn diễn giải lâm sàng
- 📉 Input contribution visualization

### 🧪 4. Theo dõi thí nghiệm

- 📚 Lịch sử tìm kiếm siêu tham số (40+ experiments)
- 🔄 Log thí nghiệm có thể tái tạo
- 🔧 Tools so sánh hiệu suất
- 📊 Export báo cáo HTML/PDF

### 📋 5. Lịch sử & Báo cáo

- 🗃️ Lưu trữ dự đoán bệnh nhân
- 📄 Tạo báo cáo PDF tự động
- 💾 Xuất dữ liệu CSV/Excel
- 📈 Thống kê sử dụng

---

## 💻 Cài đặt

### Yêu cầu hệ thống

- 🐍 Python 3.10+ (khuyến nghị 3.11)
- 📦 pip package manager
- 💾 8GB RAM (khuyến nghị 16GB)
- 💿 2GB ổ cứng trống

### Cài đặt

```bash
# Clone repository
git clone https://github.com/Rekk-tech/Heart-Disease-Prediction-System.git
cd Heart-Disease-Prediction-System

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows)
venv\Scripts\activate

# Kích hoạt môi trường (Linux/Mac)
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

---

## 🎮 Sử dụng

### Triển khai local

```bash
# Windows
streamlit run app/streamlit_app.py

# Linux/Mac
./run.sh
```

### Triển khai đám mây

🌐 **Streamlit Cloud:**
- **URL:** https://heart-disease-prediction-systems.streamlit.app/
- **Auto-deploy:** Tự động khi push code
- **Uptime:** 24/7 khả dụng
- **SSL:** HTTPS bảo mật

### Huấn luyện mô hình

```bash
# Chạy hyperparameter tuning
python scripts/train_models.py

# Kết quả lưu tại experiments/
```

### Jupyter Notebooks

```bash
# Khởi động Jupyter
jupyter lab notebooks/

# Notebooks có sẵn:
# - 01_AdaBoost_Model.ipynb
# - 02_Create_Datasets.ipynb  
# - 03_Deploy_Streamlit.ipynb
```

---

## Technical Details

### 🔧 Dependencies chính

- **🤖 ML:** scikit-learn, XGBoost, LightGBM, joblib
- **🖥️ UI:** Streamlit 1.25+, Plotly, matplotlib
- **🔬 Optimization:** Optuna (TPE sampling)
- **📊 Data:** pandas, numpy, scipy
- **📄 Reports:** reportlab, SHAP
- **🔒 Utils:** pathlib, datetime, json

### 🔄 Tính tái tạo

- 🌱 Fixed random seed (42) cho tất cả experiments
- 📝 Log hyperparameter hoàn chỉnh (40+ experiments)
- 🏷️ Versioned model artifacts
- ⚙️ Experiment manager với metadata

---

## ⚠️ Giới hạn & Tuyên bố miễn trừ

🎓 **CHỈ DÀNH CHO MỤC ĐÍCH GIÁO DỤC/NGHIÊN CỨU**

Hệ thống này KHÔNG được thiết kế cho sử dụng lâm sàng. Luôn tham khảo ý kiến bác sĩ chuyên khoa cho chẩn đoán và điều trị.

**🚨 Hạn chế đã biết:**

- 📊 Kích thước dataset nhỏ (n=303)
- 🏥 Giới hạn ở dân số Cleveland clinic
- 🔬 Chưa có validation cohort ngoài
- ⏰ Thiếu tính năng: xu hướng thời gian
- 🌍 Chưa validation trên dân số Việt Nam

---

## 🙏 Lời cảm ơn

- **🏛️ UCI Machine Learning Repository** - Cung cấp Cleveland Heart Disease dataset
- **🌟 Open-source communities** - scikit-learn, Streamlit, Optuna, Plotly
- **🎓 VietAI AIO2025** - Hỗ trợ học tập và mentoring
- **👨‍🏫 Instructors & Mentors** - Hướng dẫn và phản hồi quý báu

---

## 📄 License

📚 **Sử dụng giáo dục và nghiên cứu.** Xem license của từng package dependencies.

---

## 👥 Đóng góp

Chào mừng contributions! Vui lòng:

1. 🍴 Fork repository
2. 🌟 Tạo feature branch
3. 💻 Commit changes
4. 📤 Push và tạo Pull Request

---

## 📞 Liên hệ

**Nhóm phát triển:** AIO2025 VietAI Learning Team

- 📧 **Email:** [Contact through GitHub]
- 🐙 **GitHub:** https://github.com/Rekk-tech/Heart-Disease-Prediction-System
- 🌐 **Demo:** https://heart-disease-prediction-systems.streamlit.app/

---

⭐ **Nếu dự án hữu ích, hãy cho chúng tôi một star!** ⭐
