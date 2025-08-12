# GlucoPredict: Diabetes Prediction using Machine Learning

GlucoPredict is a machine learning project that predicts the likelihood of diabetes based on patient health indicators.  
It uses **scikit-learn** for model building, includes **data preprocessing**, **exploratory data analysis (EDA)**, and evaluates model performance using multiple metrics.

---

## 📌 Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA) with visualizations
- Machine learning classification model using scikit-learn
- Performance evaluation with accuracy, precision, recall, and confusion matrix
- Well-documented Jupyter Notebook for easy reproducibility

---

## 📂 Dataset
The dataset contains medical records with the following features:
- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

**Target Variable:**  
`Outcome` — (1 = Diabetic, 0 = Non-diabetic)

> This dataset is available publicly as the [PIMA Indians Diabetes Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

---

## ⚙️ Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
requirements.txt

nginx
pandas
numpy
matplotlib
seaborn
scikit-learn
🚀 Usage
Clone the repository

bash
git clone https://github.com/sammysamss/GlucoSense-AI.git
Navigate to the project folder

bash
cd glucopredict
Open the Jupyter Notebook

bash
jupyter notebook Project_3_Diabetes_Prediction.ipynb
Run all cells to reproduce preprocessing, training, and evaluation results.

