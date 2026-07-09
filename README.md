# Employee Attrition Prediction

A machine learning-powered web application that predicts employee attrition risk using HR data. Built with Streamlit and scikit-learn (Logistic Regression). This tool helps HR teams and managers identify employees at risk of leaving, understand the key drivers of attrition, and take proactive retention measures.

## Problem Statement

Employee attrition (voluntary turnover) is a significant challenge for organizations. Losing skilled employees leads to:
- **High replacement costs** — recruitment, onboarding, training
- **Loss of institutional knowledge** and productivity dips
- **Decreased team morale** and increased workload on remaining staff

Traditional HR analytics rely on manual reporting and lagging indicators. This project provides a **data-driven, predictive approach** to identify at-risk employees early so that retention strategies can be targeted effectively.

## Solution Overview

1. **Data Analysis** — Exploratory Data Analysis (EDA) on 1,470 employee records with 35+ features
2. **Model Comparison** — Tested Logistic Regression, Random Forest, and Gradient Boosting; selected Logistic Regression for best Recall
3. **Web Application** — Streamlit frontend with file upload, predictions, interactive dashboards, and PDF export

### Key Business Insights from EDA

| Finding | Insight |
|---------|---------|
| **Sales department** has highest attrition (~20.6%) | Prioritize retention programs for sales teams |
| **Sales Representatives** at ~40% attrition | Most at-risk role — needs immediate attention |
| **Low work-life balance (rating 1)** → 31% attrition | Improve WLB policies to retain employees |
| **First 2 years** have highest turnover risk | Strengthen onboarding and early-career engagement |
| **Lower income** correlates with leaving | Compensation review may help retention |

## Features

- **Upload & Analyze** — Upload CSV/Excel employee data; system auto-detects columns even with different naming conventions
- **ML Predictions** — Logistic Regression model (balanced class weights) predicts attrition probability for each employee
- **Risk Categorization** — Low (<30%), Medium (30-60%), High (>60%) risk tiers with color-coded indicators
- **Interactive Dashboard** — Charts by department, job role, age group, tenure, compensation, satisfaction, and work-life balance
- **Employee Explorer** — Filter by department/role/gender/risk, search by ID or name, and drill into individual profiles with prediction details
- **PDF Export** — Generate a comprehensive PDF report with KPIs, charts, demographics, and risk breakdown

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit, Plotly |
| Backend | Python, scikit-learn |
| Model | Logistic Regression (class_weight='balanced') |
| Report Generation | fpdf2, Kaleido (plotly.to_image) |
| Data Processing | Pandas, NumPy |
| Serialization | joblib |
| Visualization | Plotly Express, Matplotlib, Seaborn |

## Model Performance

Three classification algorithms were compared on the IBM HR Analytics dataset (80/20 stratified split). **Recall** was prioritized as the primary metric because catching at-risk employees is more valuable than avoiding false alarms.

| Model | Precision | Recall | F1 Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.34 | **0.62** | **0.44** | **0.80** |
| Random Forest | 0.57 | 0.09 | 0.15 | 0.77 |
| Gradient Boosting | 0.59 | 0.21 | 0.31 | 0.79 |

### Feature Importance (Top 10)

| Feature | Coefficient |
|---------|------------|
| JobRole_Laboratory Technician | +0.80 |
| OverTime_Yes | +0.77 |
| BusinessTravel_Travel_Frequently | +0.72 |
| JobLevel | +0.66 |
| TotalWorkingYears | -0.66 |
| JobRole_Sales Representative | +0.55 |
| BusinessTravel_Travel_Rarely | +0.51 |
| EducationField_Life Sciences | -0.51 |
| YearsSinceLastPromotion | +0.50 |
| Department_Sales | +0.48 |

Positive coefficients = higher attrition risk. Negative = lower risk.

## Project Structure

```
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .gitignore
├── README.md
├── assets/
│   └── style.css             # Custom professional light theme
├── components/
│   └── dashboard.py          # Dashboard rendering (KPIs, charts)
├── model/
│   ├── attrition_model.pkl   # Trained Logistic Regression model
│   ├── scaler.pkl            # Fitted StandardScaler
│   ├── feature_columns.pkl   # Expected feature column names
│   └── train_model.py        # Model training pipeline
├── utils/
│   ├── preprocessing.py      # Column detection, normalization, encoding
│   ├── prediction.py         # Prediction wrapper
│   └── report.py             # PDF report generation
├── notebooks/
│   └── analysis.ipynb        # Full EDA + model comparison notebook
├── charts/                    # Visualizations from analysis
├── dataset/
│   └── employees.csv          # Sample HR dataset (IBM HR Analytics)
├── dashboard/
│   └── Employees_dataanalysis.pbix  # Power BI companion dashboard
└── presentation/
    └── Employee_Attrition_Prediction_Presentation.pptx
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Usage

1. **Upload** — Upload a CSV or Excel file containing employee records. The system automatically recognizes columns even with different naming conventions (e.g., "Department", "Dept", "BusinessUnit" all work).
2. **Run Analysis** — Click "Run Full Analysis" to process the data. The ML model generates predictions, attrition probabilities, and risk categories for each employee.
3. **Explore Dashboard** — View attrition trends by department, job role, age, tenure, compensation, satisfaction, and work-life balance.
4. **Review Employees** — Filter and search the workforce, then drill into individual profiles to see their predicted risk and contributing factors.
5. **Export** — Download a comprehensive PDF report or the enriched dataset with predictions.

### Sample Dataset

The repository includes `dataset/employees.csv` (IBM HR Analytics Employee Attrition dataset) for testing. It contains 1,470 records with 35 attributes.

## Deployment

### Streamlit Cloud (Free)

1. Push this repository to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with GitHub
3. Click **"New app"**, select your repo, set main file to `app.py`
4. Click **Deploy** — the platform auto-installs dependencies and launches the app

### Alternative Platforms

- **Hugging Face Spaces** — Create a Space with Streamlit SDK, push code
- **Render** — Use a Web Service with `streamlit run app.py` as start command
- **Docker** — Build a container using the provided `requirements.txt`

## License

MIT
