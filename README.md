# Employee Attrition Prediction

A machine learning-powered web application that predicts employee attrition risk using HR data. Built with **Streamlit** and **scikit-learn (Logistic Regression)**, this tool helps HR teams and managers identify employees at risk of leaving, understand the key drivers of attrition, and take proactive retention measures.

---

## Overview

Employee attrition (voluntary turnover) is a significant challenge for organizations. This project provides a **data-driven, predictive approach** to identify at-risk employees early so that retention strategies can be targeted effectively. The system analyzes 30+ HR features — demographics, job role, compensation, satisfaction scores, and tenure — to classify attrition risk as **Low / Medium / High**.

---

## Problem Statement

Losing skilled employees leads to:

- **High replacement costs** — recruitment, onboarding, training
- **Loss of institutional knowledge** and productivity dips
- **Decreased team morale** and increased workload on remaining staff

Traditional HR analytics rely on manual reporting and lagging indicators. There is a need for an **automated, ML-powered tool** that can proactively flag at-risk employees and surface actionable insights.

---

## Solution

1. **Data Analysis** — Exploratory Data Analysis (EDA) on 1,470 employee records with 35+ features from the IBM HR Analytics dataset
2. **Model Comparison** — Tested Logistic Regression, Random Forest, and Gradient Boosting; selected **Logistic Regression** for best Recall (0.62)
3. **Web Application** — Streamlit frontend with file upload, predictions, interactive dashboards, and PDF export

### Key Business Insights from EDA

| Finding | Insight |
|---|---|
| **Sales department** has highest attrition (~20.6%) | Prioritize retention programs for sales teams |
| **Sales Representatives** at ~40% attrition | Most at-risk role — needs immediate attention |
| **Low work-life balance (rating 1)** → 31% attrition | Improve WLB policies to retain employees |
| **First 2 years** have highest turnover risk | Strengthen onboarding and early-career engagement |
| **Lower income** correlates with leaving | Compensation review may help retention |

---

## Key Features

- **Upload & Analyze** — Upload CSV/Excel employee data; system auto-detects columns even with different naming conventions
- **ML Predictions** — Logistic Regression model (balanced class weights) predicts attrition probability for each employee
- **Risk Categorization** — Low (<30%), Medium (30-60%), High (>60%) risk tiers with color-coded indicators
- **Interactive Dashboard** — Charts by department, job role, age group, tenure, compensation, satisfaction, and work-life balance
- **Employee Explorer** — Filter by department/role/gender/risk, search by ID or name, drill into individual profiles with prediction details
- **PDF Export** — Generate a comprehensive PDF report with KPIs, charts, demographics, and risk breakdown
- **Smart Column Detection** — Recognizes columns regardless of naming variations (e.g., "Department", "Dept", "BusinessUnit" all work)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Streamlit Frontend                    │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Upload  │  │ Dashboard │  │Employees │  │  Export  │  │
│  │  Module  │  │  Module   │  │  Module  │  │  Module  │  │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │              │             │              │        │
│  ┌────┴──────────────┴─────────────┴──────────────┴────┐   │
│  │                  App Controller (app.py)            │   │
│  └────────────────────────┬───────────────────────────┘   │
└───────────────────────────┼───────────────────────────────┘
                            │
┌───────────────────────────┼───────────────────────────────┐
│                  Python Backend Layer                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐  │
│  │ Preprocessing  │  │   Prediction   │  │    PDF     │  │
│  │   (utils/)     │──│    Engine      │  │   Report   │  │
│  │ Column mapping │  │  (utils/)      │  │  (utils/)  │  │
│  │   Encoding     │  │ Inference      │  │  fpdf2     │  │
│  └───────┬────────┘  └────────┬───────┘  └────────────┘  │
│          │                    │                            │
│  ┌───────┴────────────────────┴───────────────────────┐   │
│  │              Trained Model (model/)                 │   │
│  │  ┌──────────────┐ ┌──────────┐ ┌───────────────┐   │   │
│  │  │  attrition_  │ │  scaler │ │ feature_cols  │   │   │
│  │  │  model.pkl   │ │  .pkl   │ │    .pkl       │   │   │
│  │  └──────────────┘ └──────────┘ └───────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Dashboard Components                     │   │
│  │  Plotly charts: KPI cards, bar charts, pie charts,   │   │
│  │  box plots, histograms, line charts                   │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
```

**Data Flow:**

1. User uploads CSV/Excel → preprocessing layer normalizes columns
2. Encoded data passed to trained Logistic Regression model
3. Model outputs prediction (0/1) and probability score
4. Probabilities mapped to risk categories (Low/Medium/High)
5. Results stored in session state for dashboard, explorer, and export
6. Charts rendered via Plotly; PDF generated via fpdf2

---

## Filesystem Architecture

```
Employee-Attrition-Prediction/
├── app.py                          # Main Streamlit application entry point
├── requirements.txt                # Python dependencies
├── .gitignore
├── README.md
├── assets/
│   └── style.css                   # Custom professional light theme
├── components/
│   └── dashboard.py                # Dashboard rendering (KPIs, charts, Plotly figures)
├── model/
│   ├── attrition_model.pkl         # Trained Logistic Regression model
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── feature_columns.pkl         # Expected feature column names
│   └── train_model.py              # Model training pipeline
├── utils/
│   ├── preprocessing.py            # Column detection, normalization, encoding
│   ├── prediction.py               # Prediction wrapper
│   └── report.py                   # PDF report generation (fpdf2)
├── notebooks/
│   └── analysis.ipynb              # Full EDA + model comparison notebook
├── charts/                         # Static visualizations from analysis
├── dataset/
│   └── employees.csv               # Sample HR dataset (IBM HR Analytics)
├── dashboard/
│   └── Employees_dataanalysis.pbix # Power BI companion dashboard
├── .devcontainer/                  # Dev container configuration
└── presentation/
    └── Employee_Attrition_Prediction_Presentation.pptx
```

---

## Project Workflow

```
┌─────────────┐    ┌───────────────┐    ┌─────────────────┐
│ Upload CSV  │───>│ Auto-detect   │───>│ Run Analysis    │
│ or Excel    │    │ Columns       │    │ (Predict +      │
│ File        │    │ (30+ fields)  │    │  Dashboard)     │
└─────────────┘    └───────────────┘    └────────┬────────┘
                                                  │
                         ┌────────────────────────┼────────────────────┐
                         ▼                        ▼                    ▼
                  ┌──────────────┐      ┌────────────────┐   ┌──────────────┐
                  │ Interactive  │      │ Employee       │   │  Export      │
                  │ Dashboard    │      │ Explorer       │   │  (PDF/CSV)   │
                  │ (KPI charts) │      │ (Filter/Search)│   │              │
                  └──────────────┘      └────────────────┘   └──────────────┘
```

**Step-by-step:**

1. **Upload** — Upload CSV or Excel; system auto-maps columns using fuzzy matching
2. **Analyze** — Click "Run Full Analysis" to predict attrition probability per employee
3. **Dashboard** — Explore interactive charts (department, role, tenure, compensation, satisfaction)
4. **Employees** — Filter by department/role/risk, search by ID/name, view individual profiles
5. **Export** — Download enriched CSV or comprehensive PDF report

---

## GUI Overview

The application has a **5-tab navigation** interface:

| Tab | Function |
|---|---|
| **Upload Data** | File upload, column detection, data summary, run analysis |
| **Dashboard** | KPI cards (total employees, attrition rate), interactive Plotly charts by department/role/age/tenure/compensation/satisfaction |
| **Employees** | Filter by department/job role/gender/risk category, search by keyword, select individual employee to view profile + prediction details |
| **Export** | Generate PDF report or download enriched CSV with predictions |
| **How To Use** | Step-by-step guide, risk category explanation, model information |

The UI features a professional light theme with custom CSS, color-coded risk indicators (green/yellow/red), and responsive card-based layout.

---

## Technology Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit 1.32.0 |
| **Visualization** | Plotly Express 5.18, Matplotlib 3.8, Seaborn |
| **Machine Learning** | scikit-learn 1.4 (Logistic Regression) |
| **Data Processing** | Pandas 2.2, NumPy 1.26 |
| **Model Serialization** | joblib 1.3 |
| **PDF Generation** | fpdf2 2.7, Kaleido 0.2 |
| **Excel Support** | openpyxl 3.1 |
| **CSS Styling** | Custom (Inter font, light theme) |
| **Development** | Jupyter Notebook, VS Code devcontainer |

---

## Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Steps

```bash
# Clone the repository
git clone https://github.com/Janhavi-K99/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction

# (Optional) Create a virtual environment
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Usage

1. **Upload** — Upload a CSV or Excel file containing employee records
2. **Run Analysis** — Click "Run Full Analysis" to process the data
3. **Explore Dashboard** — View attrition trends by department, job role, age, tenure, compensation, satisfaction, and work-life balance
4. **Review Employees** — Filter and search the workforce, drill into individual profiles
5. **Export** — Download a comprehensive PDF report or enriched dataset with predictions

### Sample Dataset

The repository includes `dataset/employees.csv` (IBM HR Analytics Employee Attrition dataset) with 1,470 records and 35 attributes for testing.

---

## Supported Commands / Operations

| Operation | Description |
|---|---|
| **Upload CSV** | Upload `.csv` files with employee data |
| **Upload Excel** | Upload `.xlsx` files with employee data |
| **Run Full Analysis** | Execute ML predictions + dashboard generation |
| **Generate Dashboard** | Render analytics-only dashboard (no predictions) |
| **Filter Employees** | Filter by Department, Job Role, Gender, Risk Category |
| **Search Employees** | Free-text search across all fields |
| **View Employee Detail** | Select employee to see profile + prediction |
| **Download PDF Report** | Export comprehensive PDF with charts and KPIs |
| **Download CSV** | Export enriched dataset with predictions |

---

## Implementation Details

### Model Training (`model/train_model.py`)

- **Algorithm**: Logistic Regression with `class_weight='balanced'` to handle class imbalance
- **Features**: 45+ one-hot encoded features from 30+ original columns
- **Preprocessing**: StandardScaler normalization, dummy encoding for categoricals
- **Training**: 80/20 stratified split; 2,000 max iterations for convergence
- **Artifacts**: Model, scaler, and feature column list saved as `.pkl` files

### Model Performance

| Model | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | 0.34 | **0.62** | **0.44** | **0.80** |
| Random Forest | 0.57 | 0.09 | 0.15 | 0.77 |
| Gradient Boosting | 0.59 | 0.21 | 0.31 | 0.79 |

Recall was prioritized as the primary metric — catching at-risk employees is more valuable than avoiding false alarms.

### Feature Importance (Top 10)

| Feature | Coefficient | Impact |
|---|---|---|
| JobRole_Laboratory Technician | +0.80 | Higher risk |
| OverTime_Yes | +0.77 | Higher risk |
| BusinessTravel_Travel_Frequently | +0.72 | Higher risk |
| JobLevel | +0.66 | Higher risk |
| TotalWorkingYears | -0.66 | Lower risk |
| JobRole_Sales Representative | +0.55 | Higher risk |
| BusinessTravel_Travel_Rarely | +0.51 | Higher risk |
| EducationField_Life Sciences | -0.51 | Lower risk |
| YearsSinceLastPromotion | +0.50 | Higher risk |
| Department_Sales | +0.48 | Higher risk |

### Column Detection (`utils/preprocessing.py`)

The system uses fuzzy string matching to recognize columns regardless of naming variations:
- Normalizes column names (removes spaces, underscores, hyphens)
- Uses alias dictionaries for common variations (e.g., "Department", "Dept", "BusinessUnit", "Division")
- Falls back to substring matching if exact match fails
- Requires at least 10 of 30 core columns to enable predictions

### Prediction Engine (`utils/prediction.py`)

- Loads model, scaler, and feature columns from `.pkl` files
- Encodes input data to match training feature space (adds missing columns as 0)
- Returns prediction (0/1), probability (0.0-1.0), and risk category
- Risk thresholds: Low (<0.3), Medium (0.3-0.6), High (>0.6)

### PDF Report (`utils/report.py`)

- Uses fpdf2 for PDF generation
- Includes cover page, executive summary, KPI tables
- Renders Plotly charts as static PNG images (via Kaleido)
- Sections: Attrition by Department, by Job Role, Demographics, Compensation & Tenure, Work-Life & Satisfaction, Risk Breakdown, High-Risk Employees list

### Dashboard Components (`components/dashboard.py`)

- KPI metric cards with color-coded indicators
- Categorical attrition bar charts (department, job role)
- Box plots for income vs attrition
- Line charts for tenure vs attrition
- Pie charts for demographics (gender, education, marital status)
- Histogram for age distribution
- Risk distribution bar chart

---

## Challenges Faced

1. **Class Imbalance** — The dataset has only ~16% attrition cases. Used `class_weight='balanced'` in Logistic Regression to improve recall.
2. **Column Name Variability** — Real-world HR datasets use inconsistent naming. Built a fuzzy matching system with alias dictionaries to handle variations.
3. **Feature Engineering** — Converting 30+ mixed-type columns (numeric + categorical) into a consistent feature space for inference. Required robust encoding and missing column handling.
4. **PDF Chart Rendering** — Generating Plotly charts as static images for PDF export required integrating Kaleido/Orca, which added dependency complexity.
5. **Model Generalization** — The model trained on IBM HR data may not generalize perfectly to other organizations. Designed the system to work with partial data and provide graceful degradation.

---

## Learning Outcomes

1. **End-to-end ML pipeline** — From data preprocessing to model deployment in a web app
2. **Model evaluation** — Understanding trade-offs between precision, recall, and F1 in imbalanced classification
3. **Streamlit development** — Building multi-page apps with session state, custom CSS, and interactive components
4. **Column detection & fuzzy matching** — Handling real-world data inconsistencies
5. **PDF generation** — Creating structured reports with embedded charts using fpdf2 and Plotly/Kaleido
6. **HR analytics domain knowledge** — Understanding key drivers of employee attrition and how to translate ML outputs into business insights

---

## Future Enhancements

- [ ] **Retrain on custom data** — Allow users to upload labeled historical data to retrain the model
- [ ] **Additional ML models** — Add XGBoost, LightGBM, or Neural Network options
- [ ] **SHAP/LIME explanations** — Provide feature-level explanations for individual predictions
- [ ] **Multi-language support** — Internationalize the UI
- [ ] **API endpoint** — Expose prediction as a REST API
- [ ] **Database integration** — Connect to HR databases (PostgreSQL, Snowflake) for live data
- [ ] **Automated retraining** — Schedule periodic model retraining as new data arrives
- [ ] **A/B testing module** — Track retention intervention effectiveness

---

## Author / Contact

**Janhavi K.**

- GitHub: [@Janhavi-K99](https://github.com/Janhavi-K99)
- Project Repository: [Employee-Attrition-Prediction](https://github.com/Janhavi-K99/Employee-Attrition-Prediction)

---

## License

This project is licensed under the MIT License.
