# Health-OK: Disease Prediction using ML and Deep Learning

Health-OK is an advanced medical diagnostics and risk assessment system powered by Machine Learning (ML) and Deep Learning (DL) models. The project utilizes high-quality, curated datasets from Kaggle to train, validate, and test models designed to predict and classify various critical health conditions.

---

## 🚀 Supported Disease Prediction Models

The system incorporates both classical Machine Learning models (for structured/tabular clinical data) and Deep Learning architectures (for imaging, sequence, and complex tabular data).

### 1. Cardiovascular Disease (Heart Disease) Prediction
* **Dataset:** Cleveland Heart Disease Dataset (Kaggle)
* **Objective:** Classify whether a patient has heart disease based on clinical features (age, sex, chest pain type, cholesterol, blood pressure, etc.).
* **Models Used:** 
  * Logistic Regression (Baseline)
  * Random Forest Classifier
  * XGBoost Classifier
* **Key Metrics:** Accuracy, Precision, Recall (Sensitivity), ROC-AUC.

### 2. Diabetes Risk Assessment
* **Dataset:** Pima Indians Diabetes Dataset (Kaggle)
* **Objective:** Predict the onset of diabetes based on diagnostic measurements (glucose level, blood pressure, insulin, BMI, age, etc.).
* **Models Used:**
  * Support Vector Machines (SVM) with RBF Kernel
  * K-Nearest Neighbors (KNN)
  * Multi-Layer Perceptron (MLP) Neural Networks
* **Key Metrics:** F1-Score, Sensitivity, Specificity.

### 3. Breast Cancer Classification
* **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset (Kaggle)
* **Objective:** Classify tumor biopsies as Benign or Malignant using cell nucleus features computed from digitized images.
* **Models Used:**
  * Support Vector Classifier (SVC)
  * Random Forest
  * Artificial Neural Networks (ANN)
* **Key Metrics:** Precision-Recall AUC, False Negative Rate (critical to minimize in medical diagnosis).

### 4. Kidney Disease (Chronic Kidney Disease - CKD)
* **Dataset:** Chronic Kidney Disease Dataset (Kaggle)
* **Objective:** Detect the presence of CKD using blood test values, urinalysis, and physical symptoms.
* **Models Used:**
  * Decision Trees
  * Gradient Boosting (LightGBM)
* **Key Metrics:** Accuracy, Confusion Matrix.

---

## 🧠 Deep Learning Architectures

For complex datasets and image classification tasks, we implement:
* **Artificial Neural Networks (ANN):** Multi-layered dense networks for complex non-linear tabular data patterns.
* **Convolutional Neural Networks (CNN):** Custom CNNs and transfer learning (e.g., ResNet, VGG) for medical imaging prediction tasks (such as pneumonia detection from chest X-rays or skin cancer classification).

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM
* **Deep Learning:** TensorFlow / Keras or PyTorch
* **Model Serialization:** Joblib, Pickle

---

## 📈 Evaluation Approach
Medical diagnostics require high sensitivity (recall) to minimize false negatives (failing to diagnose a sick patient). All models are cross-validated and tuned using Hyperparameter Optimization (GridSearchCV/RandomizedSearchCV) with a heavy focus on:
* **Recall / Sensitivity:** Ensuring true positive cases are not missed.
* **F1-Score:** Balancing precision and recall.
* **ROC-AUC Score:** Measuring model discrimination capacity.
