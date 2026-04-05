# 🎓 Student Dropout Prediction using Machine Learning

This project focuses on predicting student academic outcomes—specifically identifying potential **dropouts**—using various Machine Learning classification algorithms. The goal is to provide a data-driven approach to early intervention in educational environments.

## 📌 Project Overview
The pipeline covers the entire data science lifecycle, from rigorous data preprocessing and feature selection based on correlation to model optimization using Grid Search.

### Key Features:
* **Exploratory Data Analysis (EDA):** Correlation matrix analysis to identify the most impactful features.
* **Data Preprocessing:** Handling missing values, categorical encoding (LabelEncoder), and feature scaling with MinMaxScaler.
* **Feature Selection:** Automated removal of low-correlation features to improve model efficiency.
* **Model Implementation:** Comparison between k-Nearest Neighbors (kNN), Random Forest (RF), and Support Vector Machines (SVM).
* **Hyperparameter Tuning:** Systematic optimization using `GridSearchCV` to maximize F1-Score.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** * `scikit-learn`: Model training and evaluation.
  * `pandas`: Data manipulation and analysis.
  * `matplotlib`: Visualization of results (Confusion Matrices).
  * `joblib`: Model persistence (saving/loading .pkl files).

## 📁 Project Structure
```text
├── data/               # Original dataset files
├── models/             # Saved .pkl models and scalers
├── pre_processed/      # Cleaned and processed CSV files
└── scripts/
    ├── pre_processing.py      # Data cleaning and scaling
    ├── knn_classifier.py      # kNN implementation
    ├── rf_classifier.py       # Random Forest implementation
    ├── svm_classifier.py      # SVM implementation
    └── grid_search_opt.py     # Hyperparameter tuning scripts
