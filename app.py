import gradio as gr
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime
from utils.preprocessing import (
    MODEL_REQUIRED_COLS, get_available_model_cols, can_predict, detect_groups, find_column
)
from utils.prediction import predict_attrition
from utils.report import generate_report
from components.dashboard import render_dashboard_to_html, COLORS


CUSTOM_CSS = """
.app-header { display: flex; align-items: center; justify-content: space-between; padding: 16px 0; margin-bottom: 24px; border-bottom: 1px solid #E4E7EB; }
.header-left { display: flex; align-items: center; gap: 14px; }
.logo-mark { width: 42px; height: 42px; background: rgba(37,99,235,0.08); border: 1px solid rgba(37,99,235,0.2); border-radius: 12px; display: flex; align-items: center; justify-content: center; }
.app-title { font-size: 22px; font-weight: 700; color: #1F2937; letter-spacing: -0.3px; }
.app-subtitle { font-size: 13px; color: #9CA3AF; margin-top: 1px; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; letter-spacing: 0.3px; }
.badge-primary { background: rgba(37,99,235,0.08); color: #2563EB; border: 1px solid rgba(37,99,235,0.2); }
.badge-success { background: rgba(5,150,105,0.08); color: #059669; border: 1px solid rgba(5,150,105,0.2); }
.col-tag { display: inline-block; padding: 2px 8px; background: rgba(37,99,235,0.08); border: 1px solid rgba(37,99,235,0.15); border-radius: 4px; font-size: 11px; color: #4B5563; font-family: 'SF Mono', 'Fira Code', monospace; margin: 2px; }
.detail-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #E4E7EB; }
.detail-label { font-size: 13px; color: #9CA3AF; }
.detail-value { font-size: 13px; color: #1F2937; font-weight: 500; }
.stat-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-top: 16px; }
.stat-card { background: #fff; border: 1px solid #E4E7EB; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.stat-num { font-size: 32px; font-weight: 700; color: #1F2937; }
.stat-label { font-size: 12px; color: #9CA3AF; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.8px; }
.alert { padding: 12px 16px; border-radius: 8px; font-size: 13px; margin: 8px 0; }
.alert-success { background: rgba(5,150,105,0.08); border: 1px solid rgba(5,150,105,0.2); color: #059669; }
.alert-warning { background: rgba(217,119,6,0.08); border: 1px solid rgba(217,119,6,0.2); color: #D97706; }
.alert-danger { background: rgba(220,38,38,0.08); border: 1px solid rgba(220,38,38,0.2); color: #DC2626; }
.about-card { background: #fff; border: 1px solid #E4E7EB; border-radius: 12px; padding: 24px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.about-card-title { font-size: 15px; font-weight: 600; color: #1F2937; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #E4E7EB; }
.step-num { width: 24px; height: 24px; border-radius: 50%; background: #2563EB; color: #fff; display: inline-flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; margin-right: 12px; flex-shrink: 0; }
"""

HEADER_HTML = f"""
<div class="app-header">
    <div class="header-left">
        <div class="logo-mark">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="#2563EB" stroke-width="2"/><path d="M2 17L12 22L22 17" stroke="#2563EB" stroke-width="2"/><path d="M2 12L12 17L22 12" stroke="#60A5FA" stroke-width="2"/></svg>
        </div>
        <div>
            <div class="app-title">AttritionIQ</div>
            <div class="app-subtitle">Employee Attrition Intelligence Platform</div>
        </div>
    </div>
    <div class="header-right">
        <span class="badge badge-primary">v2.0</span>
        <span class="badge badge-success">ML Powered</span>
    </div>
</div>
"""

ABOUT_HTML = f"""
<div style="display:flex;gap:24px;">
<div style="flex:1.4;">
<div class="about-card">
<div class="about-card-title">How It Works</div>
<p style="font-size:14px;color:#1F2937;margin:0 0 12px;line-height:1.6;">
AttritionIQ uses a machine learning model trained on historical HR data to predict which employees are at risk of leaving. The model analyzes patterns across <strong>30+ features</strong> including demographics, job role, compensation, tenure, satisfaction scores, and performance metrics.
</p>
</div>

<div class="about-card">
<div class="about-card-title">Step-by-Step Guide</div>
<div style="display:flex;flex-direction:column;gap:10px;">
    <div style="display:flex;gap:12px;align-items:start;">
        <div class="step-num">1</div>
        <div><strong style="color:#1F2937;">Upload Data</strong><p style="font-size:13px;color:#9CA3AF;margin:2px 0 0;">Upload a CSV or Excel file containing employee records.</p></div>
    </div>
    <div style="display:flex;gap:12px;align-items:start;">
        <div class="step-num">2</div>
        <div><strong style="color:#1F2937;">Run Analysis</strong><p style="font-size:13px;color:#9CA3AF;margin:2px 0 0;">Click "Run Full Analysis" to process data. The ML model predicts attrition probability.</p></div>
    </div>
    <div style="display:flex;gap:12px;align-items:start;">
        <div class="step-num">3</div>
        <div><strong style="color:#1F2937;">Explore Dashboard</strong><p style="font-size:13px;color:#9CA3AF;margin:2px 0 0;">View interactive charts showing attrition trends.</p></div>
    </div>
    <div style="display:flex;gap:12px;align-items:start;">
        <div class="step-num">4</div>
        <div><strong style="color:#1F2937;">Review Employees</strong><p style="font-size:13px;color:#9CA3AF;margin:2px 0 0;">Filter, search, and drill into individual employee profiles.</p></div>
    </div>
    <div style="display:flex;gap:12px;align-items:start;">
        <div class="step-num">5</div>
        <div><strong style="color:#1F2937;">Export Report</strong><p style="font-size:13px;color:#9CA3AF;margin:2px 0 0;">Generate a comprehensive PDF report with all charts and KPIs.</p></div>
    </div>
</div>
</div>
</div>

<div style="flex:1;">
<div class="about-card">
<div class="about-card-title">Understanding the Results</div>
<p style="font-size:13px;font-weight:600;color:#1F2937;margin:0 0 8px;">Risk Categories</p>
<div style="display:flex;flex-direction:column;gap:8px;margin-bottom:16px;">
    <div style="display:flex;gap:8px;align-items:center;">
        <div style="width:12px;height:12px;border-radius:50%;background:#059669;flex-shrink:0;"></div>
        <div><span style="font-weight:600;font-size:13px;color:#059669;">Low Risk</span><span style="font-size:12px;color:#9CA3AF;margin-left:6px;">(Prob &lt; 30%)</span></div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;">
        <div style="width:12px;height:12px;border-radius:50%;background:#D97706;flex-shrink:0;"></div>
        <div><span style="font-weight:600;font-size:13px;color:#D97706;">Medium Risk</span><span style="font-size:12px;color:#9CA3AF;margin-left:6px;">(Prob 30%-60%)</span></div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;">
        <div style="width:12px;height:12px;border-radius:50%;background:#DC2626;flex-shrink:0;"></div>
        <div><span style="font-weight:600;font-size:13px;color:#DC2626;">High Risk</span><span style="font-size:12px;color:#9CA3AF;margin-left:6px;">(Prob &gt; 60%)</span></div>
    </div>
</div>
</div>

<div class="about-card">
<div class="about-card-title">Model Information</div>
<div style="display:flex;flex-direction:column;gap:6px;">
    <div class="detail-row"><span class="detail-label">Algorithm</span><span class="detail-value">Logistic Regression</span></div>
    <div class="detail-row"><span class="detail-label">Features Used</span><span class="detail-value">30+ employee attributes</span></div>
    <div class="detail-row"><span class="detail-label">Output</span><span class="detail-value">Prediction + Probability</span></div>
    <div class="detail-row"><span class="detail-label">Risk Tiers</span><span class="detail-value">Low / Medium / High</span></div>
    <div class="detail-row"><span class="detail-label">Min. Columns</span><span class="detail-value">10 for predictions</span></div>
</div>
</div>
</div>
</div>
"""


def get_empty_state():
    return {
        "df_raw": None,
        "df_with_preds": None,
        "filename": None,
        "groups": {},
        "col_map": {},
        "can_pred": False,
    }


def serialize_df(df):
    if df is None:
        return None
    return df.to_json()


def deserialize_df(json_str):
    if json_str is None:
        return None
    return pd.read_json(json_str)


def _build_col_map_html(groups):
    if not groups:
        return '<p style="font-size:13px;color:#9CA3AF;">No recognizable columns detected.</p>'
    rows_html = ""
    for std_name, actual in sorted(groups.items(), key=lambda x: x[0]):
        needed = "prediction" if std_name in MODEL_REQUIRED_COLS else "analytics"
        badge = '<span class="badge badge-primary" style="font-size:10px;">Prediction</span>' if needed == "prediction" else '<span class="badge badge-success" style="font-size:10px;">Analytics</span>'
        rows_html += f"<tr><td style='padding:4px 8px;color:#1F2937;'>{std_name}</td><td style='padding:4px 8px;color:#9CA3AF;'>{actual}</td><td style='padding:4px 8px;'>{badge}</td></tr>"
    return f'<div style="margin-top:12px;"><p style="font-size:14px;font-weight:600;color:#1F2937;margin-bottom:8px;">Column Mapping</p><table style="width:100%;border-collapse:collapse;">{rows_html}</table></div>'


def _build_alert_html(can_pred, n_found, n_required):
    if can_pred:
        return f'<div class="alert alert-success">Sufficient columns detected for attrition prediction ({n_found}/{n_required} required).</div>'
    return f'<div class="alert alert-warning">Only {n_found}/{n_required} model columns detected. Dashboard will show available analytics only.</div>'


def on_upload(file, state_json):
    state = dict(state_json) if state_json else get_empty_state()
    if file is None:
        col_html = '<p style="font-size:13px;color:#9CA3AF;">Upload a file to see column mapping</p>'
        alert_html = ""
        return state, col_html, alert_html, gr.Button(interactive=False), "", gr.DataFrame(), ""

    try:
        ext = os.path.splitext(file.name)[1].lower()
        if ext == ".csv":
            df_raw = pd.read_csv(file.name)
        else:
            df_raw = pd.read_excel(file.name)

        state["df_raw"] = serialize_df(df_raw)
        state["df_with_preds"] = None
        state["filename"] = os.path.basename(file.name)

        groups = dict(detect_groups(df_raw))
        col_map = get_available_model_cols(df_raw)
        can_pred = can_predict(col_map)
        state["groups"] = groups
        state["col_map"] = col_map
        state["can_pred"] = can_pred

        col_html = _build_col_map_html(groups)
        alert_html = _build_alert_html(can_pred, len(col_map), len(MODEL_REQUIRED_COLS))

        return state, col_html, alert_html, gr.Button(interactive=True), "", gr.DataFrame(), ""

    except Exception as e:
        return state, f'<div class="alert alert-danger">Error: {str(e)}</div>', "", gr.Button(interactive=False), "", gr.DataFrame(), ""


def on_analyze(state_json):
    state = dict(state_json) if state_json else get_empty_state()
    df_raw = deserialize_df(state.get("df_raw"))
    if df_raw is None:
        return state, '<div class="alert alert-danger">No data uploaded.</div>', "", gr.DataFrame(), gr.DataFrame(), ""

    can_pred = state.get("can_pred", False)
    try:
        if can_pred:
            preds, probs, risks = predict_attrition(df_raw)
            df_out = df_raw.copy()
            df_out["Prediction"] = preds
            df_out["Attrition_Probability"] = probs
            df_out["Risk_Category"] = risks
            state["df_with_preds"] = serialize_df(df_out)
            msg = '<div class="alert alert-success">Analysis complete! Navigate to Dashboard or Employees tabs.</div>'
        else:
            df_out = df_raw.copy()
            state["df_with_preds"] = serialize_df(df_out)
            msg = '<div class="alert alert-success">Dashboard ready! Navigate to view (no predictions available).</div>'
    except Exception as e:
        state["df_with_preds"] = serialize_df(df_raw.copy())
        msg = f'<div class="alert alert-warning">Prediction error: {str(e)}. Showing analytics only.</div>'

    return state, msg, "", gr.DataFrame(), gr.DataFrame(), ""


def get_display_df(state_json):
    state = dict(state_json) if state_json else get_empty_state()
    df = deserialize_df(state.get("df_with_preds") or state.get("df_raw"))
    if df is None:
        return None
    preferred = ["EmployeeNumber", "EmployeeId", "Age", "Department", "JobRole", "Gender", "MonthlyIncome", "YearsAtCompany"]
    if "Prediction" in df.columns:
        preferred += ["Prediction", "Attrition_Probability", "Risk_Category"]
    display = [c for c in preferred if c in df.columns]
    extra = [c for c in df.columns if c not in display and not c.startswith("_")]
    display = display + extra[:min(8, len(extra))]
    return df[display].reset_index(drop=True) if display else df.reset_index(drop=True)


def get_employees_tab(state_json, dept_val, role_val, gender_val, risk_val, search_val):
    state = dict(state_json) if state_json else get_empty_state()
    df = deserialize_df(state.get("df_with_preds") or state.get("df_raw"))
    if df is None:
        return (
            gr.Dropdown(choices=["All"], value="All"),
            gr.Dropdown(choices=["All"], value="All"),
            gr.Dropdown(choices=["All"], value="All"),
            gr.Dropdown(choices=["All"], value="All"),
            gr.DataFrame(),
            "No data uploaded yet.",
            gr.Dropdown(choices=[], value=None, interactive=False),
            "",
        )

    groups = state.get("groups", {})
    has_pred = "Prediction" in df.columns

    dept_col = groups.get("Department")
    role_col = groups.get("JobRole")
    gender_col = groups.get("Gender")
    risk_col = "Risk_Category" if has_pred else None

    def get_choices(col, current_val):
        if col and col in df.columns:
            opts = ["All"] + sorted(df[col].dropna().unique().tolist())
            return gr.Dropdown(choices=opts, value=current_val if current_val in opts else "All")
        return gr.Dropdown(choices=["All"], value="All")

    filtered = df.copy()

    if dept_col and dept_val != "All" and dept_col in filtered.columns:
        filtered = filtered[filtered[dept_col] == dept_val]
    if role_col and role_val != "All" and role_col in filtered.columns:
        filtered = filtered[filtered[role_col] == role_val]
    if gender_col and gender_val != "All" and gender_col in filtered.columns:
        filtered = filtered[filtered[gender_col] == gender_val]
    if risk_col and risk_val != "All" and risk_col in filtered.columns:
        filtered = filtered[filtered[risk_col] == risk_val]

    if search_val:
        mask = filtered.astype(str).apply(lambda x: x.str.contains(search_val, case=False, na=False)).any(axis=1)
        filtered = filtered[mask]

    preferred = ["EmployeeNumber", "EmployeeId", "Age", "Department", "JobRole", "Gender", "MonthlyIncome", "YearsAtCompany"]
    if has_pred:
        preferred += ["Prediction", "Attrition_Probability", "Risk_Category"]
    display = [c for c in preferred if c in filtered.columns]
    extra = [c for c in filtered.columns if c not in display and not c.startswith("_")]
    display = display + extra[:min(8, len(extra))]

    filtered_display = filtered[display].reset_index(drop=True) if display else filtered.reset_index(drop=True)
    count_text = f"{len(filtered):,} employees match"

    id_col = groups.get("EmployeeNumber")
    if id_col is None or id_col not in filtered.columns:
        for c in filtered.columns:
            if "employee" in str(c).lower() and any(x in str(c).lower() for x in ["id", "number", "code"]):
                id_col = c
                break
    has_id = id_col and id_col in filtered.columns and len(filtered) > 0
    if has_id:
        emp_choices = [(f"{id_col}: {v}", v) for v in filtered[id_col].unique()]
        return (
            get_choices(dept_col, dept_val),
            get_choices(role_col, role_val),
            get_choices(gender_col, gender_val),
            get_choices(risk_col, risk_val),
            gr.DataFrame(filtered_display),
            count_text,
            gr.Dropdown(choices=emp_choices, value=emp_choices[0][1] if emp_choices else None, interactive=True),
            "",
        )

    return (
        get_choices(dept_col, dept_val),
        get_choices(role_col, role_val),
        get_choices(gender_col, gender_val),
        get_choices(risk_col, risk_val),
        gr.DataFrame(filtered_display),
        count_text,
        gr.Dropdown(choices=[], value=None, interactive=False),
        "",
    )


def get_employee_detail(state_json, emp_val):
    state = dict(state_json) if state_json else get_empty_state()
    df = deserialize_df(state.get("df_with_preds") or state.get("df_raw"))
    if df is None or emp_val is None:
        return "Select an employee to view details."

    groups = state.get("groups", {})
    id_col = groups.get("EmployeeNumber")
    if id_col is None or id_col not in df.columns:
        id_col = None
        for c in df.columns:
            if "employee" in str(c).lower() and any(x in str(c).lower() for x in ["id", "number", "code"]):
                id_col = c
                break
    if id_col is None or id_col not in df.columns:
        return "Employee ID column not found."

    row = df[df[id_col] == emp_val]
    if len(row) == 0:
        return "Employee not found."
    row = row.iloc[0]

    has_pred = "Prediction" in df.columns
    parts = ['<div style="display:flex;gap:16px;">']

    parts.append('<div class="about-card" style="flex:1;"><div class="about-card-title">Profile</div>')
    for f in ["Age", "Gender", "Department", "JobRole", "MaritalStatus", "EducationField", "MonthlyIncome", "YearsAtCompany"]:
        fc = find_column(df, [f])
        if fc and fc in row.index:
            parts.append(f'<div class="detail-row"><span class="detail-label">{f}</span><span class="detail-value">{row[fc]}</span></div>')
    parts.append('</div>')

    if has_pred and "Prediction" in row.index:
        parts.append(f'<div class="about-card" style="flex:1;"><div class="about-card-title">Prediction</div>')
        pred_val = int(row["Prediction"])
        prob_val = row["Attrition_Probability"]
        risk_val = row["Risk_Category"]
        label = "Likely to Leave" if pred_val == 1 else "Likely to Stay"
        color = COLORS["danger"] if pred_val == 1 else COLORS["success"]
        parts.append(f'<div class="detail-row"><span class="detail-label">Status</span><span class="detail-value" style="color:{color};font-weight:600;">{label}</span></div>')
        parts.append(f'<div class="detail-row"><span class="detail-label">Probability</span><span class="detail-value">{prob_val:.1%}</span></div>')
        parts.append(f'<div class="detail-row"><span class="detail-label">Risk Level</span><span class="detail-value" style="color:{color};">{risk_val}</span></div>')

        extra_cols = [c for c in df.columns if c not in ["Prediction", "Attrition_Probability", "Risk_Category", id_col] and not c.startswith("_")]
        extra_cols = extra_cols[:10]
        for c in extra_cols:
            if c in row.index:
                parts.append(f'<div class="detail-row"><span class="detail-label">{c}</span><span class="detail-value">{row[c]}</span></div>')
        parts.append('</div>')

    parts.append('</div>')
    return ''.join(parts)


def generate_pdf(state_json):
    state = dict(state_json) if state_json else get_empty_state()
    df = deserialize_df(state.get("df_with_preds") or state.get("df_raw"))
    if df is None:
        return None
    try:
        pdf_bytes = generate_report(df)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf_bytes)
        tmp.close()
        return tmp.name
    except Exception as e:
        return None


def generate_csv(state_json):
    state = dict(state_json) if state_json else get_empty_state()
    df = deserialize_df(state.get("df_with_preds") or state.get("df_raw"))
    if df is None:
        return None
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(csv_bytes)
    tmp.close()
    return tmp.name


def get_export_stats(state_json):
    state = dict(state_json) if state_json else get_empty_state()
    df = deserialize_df(state.get("df_with_preds"))
    if df is None or "Prediction" not in df.columns:
        return ""
    total = len(df)
    att = int(df["Prediction"].sum())
    stay = total - att
    stats = f"""
    <div class="stat-grid">
        <div class="stat-card"><div class="stat-num">{total:,}</div><div class="stat-label">Total Employees</div></div>
        <div class="stat-card"><div class="stat-num" style="color:{COLORS['danger']};">{att:,}</div><div class="stat-label">Predicted to Leave</div></div>
        <div class="stat-card"><div class="stat-num" style="color:{COLORS['success']};">{stay:,}</div><div class="stat-label">Predicted to Stay</div></div>
    </div>
    """
    rc = df["Risk_Category"].value_counts()
    rows = "".join(
        f'<tr><td style="padding:6px 8px;color:#1F2937;">{cat}</td><td style="padding:6px 8px;color:#1F2937;">{rc.get(cat, 0):,}</td><td style="padding:6px 8px;color:#9CA3AF;">{rc.get(cat, 0)/total*100:.1f}%</td></tr>'
        for cat in ["High Risk", "Medium Risk", "Low Risk"]
    )
    stats += f"""
    <div class="about-card" style="margin-top:16px;">
        <div class="about-card-title">Risk Breakdown</div>
        <table style="width:100%;"><tr><th style="text-align:left;color:#9CA3AF;">Category</th><th style="text-align:left;color:#9CA3AF;">Count</th><th style="text-align:left;color:#9CA3AF;">Percentage</th></tr>{rows}</table>
    </div>
    """
    return stats


def get_dashboard(state_json):
    state = dict(state_json) if state_json else get_empty_state()
    df = deserialize_df(state.get("df_with_preds") or state.get("df_raw"))
    if df is None:
        return "Upload and analyze data first."
    return render_dashboard_to_html(df)


INITIAL_STATE = get_empty_state()

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="blue"), title="AttritionIQ") as app:
    state = gr.State(INITIAL_STATE)

    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        with gr.Tab("Upload Data"):
            gr.Markdown("### Upload Employee Data")
            file_input = gr.File(file_types=[".csv", ".xlsx"], label="Upload CSV or Excel file")
            col_map_output = gr.HTML('<p style="font-size:13px;color:#9CA3AF;">Upload a file to see column mapping</p>')
            alert_output = gr.HTML("")
            analyze_btn = gr.Button("Run Full Analysis (Predict + Dashboard)", variant="primary", interactive=False)
            status_output = gr.HTML("")

            gr.Markdown("### Supported Columns")
            gr.HTML("""
            <p style="font-size:13px;color:#9CA3AF;margin:0 0 8px;">The system automatically detects columns even with different naming conventions.</p>
            """)

            cats = [
                ("Personal Info", ["Age", "Gender", "MaritalStatus", "Education", "EducationField", "DistanceFromHome"]),
                ("Job Details", ["Department", "JobRole", "JobLevel", "JobInvolvement", "OverTime"]),
                ("Compensation", ["MonthlyIncome", "DailyRate", "HourlyRate", "MonthlyRate", "PercentSalaryHike", "StockOptionLevel"]),
                ("Tenure & Experience", ["YearsAtCompany", "TotalWorkingYears", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "NumCompaniesWorked"]),
                ("Satisfaction", ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"]),
                ("Performance", ["PerformanceRating", "TrainingTimesLastYear"]),
                ("Other", ["BusinessTravel", "EmployeeNumber"]),
            ]
            cat_html = ""
            for cat_name, cols_list in cats:
                cat_html += f'<p style="font-size:12px;font-weight:600;color:#60A5FA;margin:8px 0 4px;">{cat_name}</p>'
                cat_html += '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;">'
                for c in cols_list:
                    cat_html += f'<span class="col-tag">{c}</span>'
                cat_html += '</div>'
            gr.HTML(cat_html)

            file_input.change(
                on_upload,
                inputs=[file_input, state],
                outputs=[state, col_map_output, alert_output, analyze_btn, status_output, gr.DataFrame(visible=False), gr.DataFrame(visible=False)]
            )
            analyze_btn.click(
                on_analyze,
                inputs=[state],
                outputs=[state, status_output, col_map_output, gr.DataFrame(visible=False), gr.DataFrame(visible=False), gr.DataFrame(visible=False)]
            )

        with gr.Tab("Dashboard"):
            dashboard_output = gr.HTML("Upload and analyze data first.")
            analyze_btn.click(
                get_dashboard,
                inputs=[state],
                outputs=[dashboard_output]
            )
            state.change(
                get_dashboard,
                inputs=[state],
                outputs=[dashboard_output]
            )

        with gr.Tab("Employees"):
            with gr.Row():
                dept_filter = gr.Dropdown(label="Department", choices=["All"], value="All")
                role_filter = gr.Dropdown(label="Job Role", choices=["All"], value="All")
                gender_filter = gr.Dropdown(label="Gender", choices=["All"], value="All")
                risk_filter = gr.Dropdown(label="Risk Category", choices=["All"], value="All")
            search_input = gr.Textbox(label="", placeholder="Search by Employee ID, Name, or any value...")
            emp_count = gr.Markdown("Upload data to get started.")
            emp_table = gr.DataFrame()
            emp_selector = gr.Dropdown(label="Select Employee", choices=[], value=None, interactive=False)
            emp_detail = gr.HTML("Select an employee to view details.")

            all_emp_inputs = [state, dept_filter, role_filter, gender_filter, risk_filter, search_input]
            all_emp_outputs = [dept_filter, role_filter, gender_filter, risk_filter, emp_table, emp_count, emp_selector, emp_detail]

            for comp in [dept_filter, role_filter, gender_filter, risk_filter, search_input]:
                comp.change(get_employees_tab, inputs=all_emp_inputs, outputs=all_emp_outputs)

            state.change(get_employees_tab, inputs=all_emp_inputs, outputs=all_emp_outputs)

            analyze_btn.click(get_employees_tab, inputs=all_emp_inputs, outputs=all_emp_outputs)

            emp_selector.change(get_employee_detail, inputs=[state, emp_selector], outputs=[emp_detail])

        with gr.Tab("Export"):
            gr.Markdown("### Export Report")
            gr.Markdown("Download a comprehensive PDF report with KPIs, charts, demographics, and risk analysis.")
            with gr.Row():
                pdf_btn = gr.Button("Generate PDF Report", variant="primary")
                csv_btn = gr.Button("Download Enriched Data (CSV)")
            pdf_output = gr.File(label="Download PDF Report")
            csv_output = gr.File(label="Download CSV")
            export_stats = gr.HTML("Upload and analyze data first.")

            pdf_btn.click(generate_pdf, inputs=[state], outputs=[pdf_output])
            csv_btn.click(generate_csv, inputs=[state], outputs=[csv_output])

            state.change(get_export_stats, inputs=[state], outputs=[export_stats])
            analyze_btn.click(get_export_stats, inputs=[state], outputs=[export_stats])

        with gr.Tab("How To Use"):
            gr.HTML(ABOUT_HTML)

    gr.Markdown("---")
    gr.HTML('<div style="text-align:center;padding:8px 0;"><p style="font-size:12px;color:#9CA3AF;">AttritionIQ &bull; Employee Attrition Intelligence Platform</p></div>')


if __name__ == "__main__":
    app.launch()
