# ✈️ Flight Delay Prediction – Machine Learning Pipeline (ATL Airport, 2021)

**Author:** [Ricardos Abi Akar](https://github.com/ricardos-ai)  
**Role:** AI & Machine Learning Engineer  
**Project Title:** *FlightFlow Dynamics – Predicting Flight Delays using Machine Learning*  

---

## 🧠 Project Overview

This project develops a **complete end‑to‑end machine learning system** for predicting flight delays and early arrivals at **Hartsfield–Jackson Atlanta International Airport (ATL)** using large‑scale U.S. flight data for **2021**.  
It integrates **advanced data preprocessing, feature engineering, unsupervised learning (PCA, K‑Means)**, and **supervised ensemble models (XGBoost, LightGBM, Random Forest)** — demonstrating technical maturity in **AI engineering and MLOps‑ready design**.

---

## 🗂️ Repository Structure

```
flight-delay-prediction-ml-pipeline/
│
├── notebooks/
│   ├── ATL_Flight_Delays_ML_Pipeline_2021_Lite.ipynb      # Clean version (code + markdown, GitHub previewable)
│   └── ATL_Flight_Delays_ML_Pipeline_2021_Full.ipynb      # Full version (includes all outputs and visualizations)
│
├── report/
│   └── FlightFlow_Dynamics_Report.pdf                      # Detailed technical report and business analysis
│
├── .gitignore
└── README.md
```

---

## 📊 Dataset

**Source:** [Bureau of Transportation Statistics – U.S. Domestic Flights, 2021](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp)

The dataset contains over **6,311,000 records** and **61 variables** describing domestic flight operations in the United States for the year **2021**.  
It includes attributes such as departure/arrival times, distances, delays, taxi times, and airline/operator metadata.

> 🧩 To reproduce this project, download the original dataset directly from the BTS website and place it under a local `data/` directory.

---

## ⚙️ Technical Stack

| Category | Tools & Libraries |
|-----------|-------------------|
| **Language** | Python 3.10 |
| **Core Libraries** | pandas, numpy, matplotlib, seaborn, scipy |
| **Machine Learning** | scikit‑learn, xgboost, lightgbm |
| **Unsupervised Learning** | PCA (Principal Component Analysis), K‑Means clustering |
| **Supervised Learning** | XGBoost, RandomForest, Gradient Boosting, LightGBM |
| **Feature Engineering** | Custom metrics (EfficiencyRatio, MeanEfficiencyRatio_ByOriginAirline), aggregated route‑level statistics |
| **Validation & Tuning** | Stratified K‑Fold Cross‑Validation, RandomizedSearchCV, manual parameter tuning |
| **Explainability** | SHAP (Shapley Additive Explanations), permutation feature importance, XGBoost feature gain/weight analysis |
| **Statistical Tests** | Pearson correlation, Chi‑square test, ANOVA, Bartlett’s sphericity, KMO sampling adequacy |
| **Visualization** | seaborn, matplotlib, correlation heatmaps, PCA scree/circle plots, SHAP beeswarm/waterfall |
| **Environment** | Jupyter Notebook / Google Colab |

---

## 🧩 Methodology

### 1. **Exploratory Data Analysis (EDA)**
- Built a custom reusable **`Analysis` class** for univariate and bivariate analysis.
- Investigated missing values, modality distribution, and high‑cardinality categorical features.
- Dropped redundant or correlated variables to reduce dimensionality and improve computational efficiency.

### 2. **Feature Engineering**
- Engineered **Efficiency Ratio** (`Distance / AirTime`) to measure operational performance.
- Computed aggregated airline‑route efficiency metrics (mean delay, mean efficiency, etc.).
- Encoded categorical variables using dummy and label encoding as appropriate.

### 3. **Unsupervised Learning**
- Performed **PCA** to reduce dimensionality and reveal underlying flight behavior patterns.
- Conducted **Bartlett’s test** and **KMO** to validate PCA suitability.
- Applied **K‑Means (k=2)** with elbow and silhouette analysis → identified long‑haul vs short‑haul delay clusters.

### 4. **Supervised Learning**
- Modeled **15 delay groups** using multiple classifiers: Random Forest, LightGBM, Gradient Boosting, and **XGBoost**.
- Performed **hyperparameter tuning** (learning rate, tree depth, subsample, min_child_weight, etc.).
- Used **Stratified K‑Fold Cross‑Validation** for balanced model evaluation.
- Achieved final test accuracy ≈ **0.70**, F1 ≈ **0.68**, with strong performance (AUC 0.8‑0.9) for specific delay groups.

### 5. **Explainability & Model Interpretation**
- Employed **SHAP** for local/global interpretability and feature impact visualization.
- Compared **gain**, **weight**, and **permutation importance** methods.
- Identified key drivers: `WheelsOff`, `DepHour`, `EfficiencyRatio`, and airline‑route‑level efficiency.

### 6. **Operational Insights**
- Long‑haul flights (Cluster 0) show higher average delays → require targeted scheduling strategies.
- Airlines with recurrent inefficiencies (e.g., high EfficiencyRatio) can be penalized via gate pricing.
- Predictive models support **dynamic gate assignment**, **staff optimization**, and **profit‑driven concession management**.

---

## 🧩 Results Summary

| Metric | Value |
|---------|--------|
| **Model Type** | Tuned XGBoost Classifier |
| **Accuracy (Test)** | 0.70 |
| **F1‑Score (Test)** | 0.68 |
| **ROC‑AUC (selected classes)** | 0.8 – 0.9 |
| **Best Predictors** | WheelsOff, EfficiencyRatio, DepHour, MeanEfficiency_ByOriginAirline |

> The final model successfully differentiates delay categories and provides interpretable insights for airport operation planning.

---

## 🧩 Files Description

| File | Description |
|------|-------------|
| `ATL_Flight_Delays_ML_Pipeline_2021_Lite.ipynb` | Code‑only notebook (small, optimized for web preview). |
| `ATL_Flight_Delays_ML_Pipeline_2021_Full.ipynb` | Full notebook including all visual outputs and plots (large, for offline use). |
| `FlightFlow_Dynamics_Report.pdf` | Full technical and business report detailing methodology, analysis, and findings. |

---

## 🧩 Reproducibility Instructions

1. Clone the repository:
   ```bash
   git clone git@github.com:ricardos-ai/flight-delay-prediction-ml-pipeline.git
   cd flight-delay-prediction-ml-pipeline
   ```

2. Download the 2021 U.S. flight dataset from the [BTS Official Source](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp) and place it under a local `data/` directory.

3. Run the notebook:
   ```bash
   jupyter notebook notebooks/ATL_Flight_Delays_ML_Pipeline_2021_Lite.ipynb
   ```

---

## 📈 Key Takeaways

- Demonstrates **AI engineer–level mastery** of data preprocessing, feature engineering, unsupervised and supervised ML.  
- Combines **statistical rigor** (Bartlett, KMO, ANOVA) with **explainable AI** (SHAP).  
- Designed for **real‑world scalability and operational decision‑support**.  
- Clear code structure with modular analysis class and reproducible pipeline.

---

## 📚 References

- Bureau of Transportation Statistics (2021) – *On‑Time Performance Data*  
- Henriques & Feiteira (2018) – *Predictive Modelling: Flight Delays and Associated Factors*  
- Neyshabouri & Sherry (2014) – *Airport Surface Operations: ATL Case Study*  
- Yablonsky et al. (2014) – *Flight Delay Performance at ATL*

---

## 🧩 License

This project is released under the [MIT License](LICENSE).  
You may freely use, modify, and distribute this work with proper attribution.

---

**Developed by Ricardos Abi Akar – AI Engineer**  
*Turning predictive models into operational intelligence.*  
