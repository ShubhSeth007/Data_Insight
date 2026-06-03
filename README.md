# 📊 DataInsight: Automated EDA Dashboard

[![Streamlit App]([https://datainsight-jzekgvb9oq5nphtxsowfdq.streamlit.app/])

An enterprise-grade, fully automated Exploratory Data Analysis (EDA) web application built in Python. This tool ingests raw tabular datasets and instantly generates interactive, production-ready visualizations alongside deep statistical profiles, bridging the gap between raw data and actionable data storytelling.

🔗 **[Live Demo Link]([https://datainsight-jzekgvb9oq5nphtxsowfdq.streamlit.app/])**

---

## 🎯 Core Features

### 📋 1. Smart Data Ingestion & Overview
* Automatically categorizes attributes into Numerical and Categorical feature sets.
* Tracks structural high-level metadata (total records, dimensions, structural splits).
* Generates a dynamic **Missing Data Matrix** heatmap to visualize spatial data gaps.

### 📈 2. Automated Numerical Profiling
* Maps distributions utilizing unified **Histograms and Kernel Density Estimates (KDE)**.
* Measures **Skewness** and provides instant data transformation alerts (e.g., log transforms).
* Pinpoints statistical anomalies using **Box Plots** and automated **Interquartile Range (IQR)** boundaries.

### 📊 3. Categorical Engine & Cardinality Control
* Handles high-cardinality attributes cleanly by self-truncating plots to focus on dominant categories.
* Visualizes localized class imbalances to mitigate predictive modeling bias early.

### 🔗 4. Bivariate Analysis & Multicollinearity Screening
* Renders a highly styled, masked lower-triangle **Correlation Matrix Heatmap**.
* Automatically runs programmatic flags for highly correlated feature pairings ($|r| > 0.7$) to protect against multicollinearity.
* Includes a fully custom **Bivariate Scatter Visualizer** with embedded regression trendlines (`sns.regplot`).

---

## 🧰 Tech Stack

* **Interface & Deployment:** Streamlit (Streamlit Community Cloud)
* **Data Processing & Analytics:** Pandas, NumPy, SciPy (Statistical tracking)
* **Data Visualization:** Matplotlib, Seaborn

---

## 🚀 Local Installation & Setup

Want to run this project locally on your machine? Follow these quick steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/ShubhSeth007/Data_Insight.git)
   cd YOUR_REPO_NAME
   ```

2. **Install dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

---

## 👤 Author
* **Your Name** - [Your GitHub Profile](https://github.com/ShubhSeth007)
