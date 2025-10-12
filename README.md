# 🧠 Mental Health in Technology-related Jobs

A case study applying unsupervised machine learning techniques — including **k-Means clustering**, **Principal Component Analysis (PCA)**, and **Chi-Square feature selection** — to analyze mental health survey data from professionals in the tech industry.

---

## 📘 Overview

This repository contains the full implementation of the case study *"Mental Health in Technology-related Jobs"* by **Chiara Supit** (B.Sc. in Applied Artificial Intelligence, IU International University of Applied Sciences).

The study explores how unsupervised learning and feature engineering can be applied to identify key factors influencing mental health conditions among employees in technology-related occupations. Using the **OSMI Mental Health in Tech Survey 2016** dataset, the analysis highlights the role of employment type, company size, gender, and job role in shaping workplace mental wellbeing.

---

## 🧩 Objectives

- To preprocess and clean complex survey data containing missing and categorical values.  
- To apply **k-Means clustering** for grouping employees based on mental health patterns.  
- To use **PCA** for visualizing high-dimensional data in two-dimensional space.  
- To identify key features influencing cluster separation using **Chi-Square tests**.  
- To provide HR and management insights for better workplace mental health interventions.

---

## 📊 Methodology

### **1. Data Exploration and Cleaning**
- Dataset: [OSMI Mental Health in Tech Survey 2016](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016)
- Steps:
  - Handling missing values (mode imputation, threshold filtering)
  - Text normalization and encoding (One-Hot Encoding)
  - Feature selection using Variance Threshold

### **2. Clustering and Evaluation**
- Unsupervised clustering with **k-Means**
- Determining optimal *k* using **Elbow** and **Silhouette** methods

### **3. Dimensionality Reduction**
- Visualization via **Principal Component Analysis (PCA)**

### **4. Feature Importance**
- Identifying significant predictors using **Chi-Square feature selection**

---

## 📈 Results Summary

- Optimal number of clusters: **k = 2**
- Top distinguishing factors: employment type, gender, company size, and job role
- PCA visualization indicated moderate separability between employee groups
- Insights suggest tailored HR strategies for:
  - Self-employed and freelance professionals  
  - Employees in large organizations  
  - Women in technical roles  
  - Front-end developers facing high stress workloads  

---

## 🧰 Technologies Used

- **Python 3.x**
- **pandas**, **NumPy**, **scikit-learn**
- **Matplotlib**, **Yellowbrick**
- **tabulate**

---

## 📂 Repository Structure

```
mental-health-in-tech/
│
├── data/                 # Dataset (OSMI 2016) or data import script
├── scripts/              # Python scripts for data cleaning and modeling
├── figures/              # Flowcharts, plots, and PCA visualizations
├── results/              # Generated output tables or CSVs
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🧾 Citation

If you reference this work in an academic context, please cite as:

> Supit, C. (2025). *Mental Health in Technology-related Jobs: A Case Study on Unsupervised Learning and Feature Engineering*. IU International University of Applied Sciences.

---

## 🌐 Author

**Chiara Supit**  
B.Sc. in Applied Artificial Intelligence  
IU International University of Applied Sciences  
📅 October 2025  

---

## 🧠 License

This project is licensed under the **MIT License** – feel free to use and modify the code with proper attribution.

---

## 🔗 Related Resources

- [OSMI Mental Health in Tech Survey 2016 – Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016)  
- [Scikit-learn Feature Selection Documentation](https://scikit-learn.org/stable/modules/feature_selection.html)  
- [IEEE Paper: Cluster Quality Analysis Using Silhouette Score](https://doi.org/10.1109/DSAA49011.2020.00096)
