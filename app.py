import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
from utils.preprocessing import (
    MODEL_REQUIRED_COLS, get_available_model_cols, can_predict, detect_groups, find_column
)
from utils.prediction import predict_attrition
from utils.report import generate_report
from components.dashboard import render_dashboard, COLORS

st.set_page_config(page_title="AttritionIQ | Employee Intelligence", layout="wide", page_icon="")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(f"""
<div class="app-header">
    <div class="header-left">
        <div class="logo-mark">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none"><path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="{COLORS['primary']}" stroke-width="2"/><path d="M2 17L12 22L22 17" stroke="{COLORS['primary']}" stroke-width="2"/><path d="M2 12L12 17L22 12" stroke="{COLORS['primary_light']}" stroke-width="2"/></svg>
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
""", unsafe_allow_html=True)

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_with_preds" not in st.session_state:
    st.session_state.df_with_preds = None
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False
if "data_summary" not in st.session_state:
    st.session_state.data_summary = {}
if "page" not in st.session_state:
    st.session_state.page = "upload"

nav_items = [
    ("upload", "Upload Data", "Upload employee dataset"),
    ("dashboard", "Dashboard", "View analytics & insights"),
    ("employees", "Employees", "Search & filter workforce"),
    ("export", "Export", "Download results"),
    ("about", "How To Use", "Learn about the platform")
]

cols = st.columns([1, 1, 1, 1, 1])
for i, (key, label, tip) in enumerate(nav_items):
    active = st.session_state.page == key
    with cols[i]:
        if st.button(label, key=f"nav_{key}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state.page = key
            st.rerun()

st.markdown(f'<div class="nav-indicator" id="nav-{st.session_state.page}"></div>', unsafe_allow_html=True)

page = st.session_state.page

if page == "upload":
    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.markdown('<div class="card"><div class="card-title">Upload Employee Data</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["csv", "xlsx"], label_visibility="collapsed")
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded_file)
                else:
                    df_raw = pd.read_excel(uploaded_file)
                st.session_state.df_raw = df_raw
                st.session_state.uploaded = False
                st.session_state.df_with_preds = None
                st.markdown(f"""
                <div class="alert alert-success">
                    <span class="alert-icon"></span>
                    <span>Loaded <strong>{uploaded_file.name}</strong> — {len(df_raw):,} rows x {len(df_raw.columns)} columns</span>
                </div>
                """, unsafe_allow_html=True)
                groups = dict(detect_groups(df_raw))
                col_map = get_available_model_cols(df_raw)
                st.session_state.data_summary = {
                    "filename": uploaded_file.name,
                    "rows": len(df_raw),
                    "cols": len(df_raw.columns),
                    "detected": groups,
                    "model_cols": col_map,
                    "can_predict": can_predict(col_map)
                }
                st.markdown('<div class="card-title" style="margin-top:16px;">Column Mapping</div>', unsafe_allow_html=True)
                if groups:
                    rows_html = ""
                    for std_name, actual in sorted(groups.items(), key=lambda x: x[0]):
                        needed = "prediction" if std_name in MODEL_REQUIRED_COLS else "analytics"
                        badge = '<span class="badge badge-success">Prediction</span>' if needed == "prediction" else '<span class="badge badge-info">Analytics</span>'
                        rows_html += f"<tr><td style='padding:4px 8px;color:{COLORS['text']};'>{std_name}</td><td style='padding:4px 8px;color:{COLORS['text_muted']};'>{actual}</td><td style='padding:4px 8px;'>{badge}</td></tr>"
                    st.markdown(f"<table style='width:100%;border-collapse:collapse;'>{rows_html}</table>", unsafe_allow_html=True)
                else:
                    st.info("No recognizable columns detected. The dashboard will show basic statistics.")
                can_pred = can_predict(col_map)
                if can_pred:
                    st.markdown(f"""
                    <div class="alert alert-success">
                        <span class="alert-icon"></span>
                        <span>Sufficient columns detected for attrition prediction ({len(col_map)}/{len(MODEL_REQUIRED_COLS)} required). Click <strong>Run Analysis</strong> below.</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    missing_ct = len(MODEL_REQUIRED_COLS) - len(col_map)
                    st.markdown(f"""
                    <div class="alert alert-warning">
                        <span class="alert-icon"></span>
                        <span>Only <strong>{len(col_map)}/{len(MODEL_REQUIRED_COLS)}</strong> model columns detected. Prediction requires ~10+ core columns. Dashboard will show available analytics only.</span>
                    </div>
                    """, unsafe_allow_html=True)
                btn_label = "Run Full Analysis (Predict + Dashboard)" if can_pred else "Generate Dashboard (No Predictions)"
                if st.button(btn_label, use_container_width=True):
                    with st.spinner("Processing data..." if not can_pred else "Running predictions..."):
                        if can_pred:
                            try:
                                preds, probs, risks = predict_attrition(df_raw)
                                df_out = df_raw.copy()
                                df_out["Prediction"] = preds
                                df_out["Attrition_Probability"] = probs
                                df_out["Risk_Category"] = risks
                                st.session_state.df_with_preds = df_out
                                st.session_state.uploaded = True
                                st.success("Analysis complete! Navigate to Dashboard.")
                            except Exception as e:
                                st.error(f"Prediction error: {str(e)}. Showing analytics only.")
                                st.session_state.df_with_preds = df_raw.copy()
                                st.session_state.uploaded = True
                        else:
                            st.session_state.df_with_preds = df_raw.copy()
                            st.session_state.uploaded = True
                            st.success("Dashboard ready! Navigate to view.")
            except Exception as e:
                st.markdown(f"""
                <div class="alert alert-danger">
                    <span class="alert-icon"></span>
                    <span>Error: {str(e)}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:48px 20px;">
                <div style="width:64px;height:64px;border-radius:16px;background:var(--accent-bg,rgba(37,99,235,0.08));display:flex;align-items:center;justify-content:center;margin:0 auto 16px;border:1px solid rgba(37,99,235,0.15);">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                </div>
                <p style="font-size:16px;font-weight:500;color:var(--text-primary,#1F2937);margin:0 0 4px;">Upload your employee dataset</p>
                <p style="font-size:13px;color:var(--text-muted,#9CA3AF);margin:0;">Supports CSV and Excel formats</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">Supported Columns</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:13px;color:var(--text-muted,#9CA3AF);margin:0 0 8px;">
            The system automatically detects columns even with different naming conventions.
        </p>
        """, unsafe_allow_html=True)
        cats = [
            ("Personal Info", ["Age", "Gender", "MaritalStatus", "Education", "EducationField", "DistanceFromHome"]),
            ("Job Details", ["Department", "JobRole", "JobLevel", "JobInvolvement", "OverTime"]),
            ("Compensation", ["MonthlyIncome", "DailyRate", "HourlyRate", "MonthlyRate", "PercentSalaryHike", "StockOptionLevel"]),
            ("Tenure & Experience", ["YearsAtCompany", "TotalWorkingYears", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "NumCompaniesWorked"]),
            ("Satisfaction", ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"]),
            ("Performance", ["PerformanceRating", "TrainingTimesLastYear"]),
            ("Other", ["BusinessTravel", "EmployeeNumber"])
        ]
        for cat_name, cols_list in cats:
            st.markdown('<div style="margin-bottom:8px;">', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:12px;font-weight:600;color:{COLORS["primary_light"]};margin:0 0 4px;">{cat_name}</p>', unsafe_allow_html=True)
            tags = "".join(f'<span class="col-tag">{c}</span>' for c in cols_list)
            st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:4px;">{tags}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "dashboard":
    if st.session_state.df_with_preds is not None:
        render_dashboard(st.session_state.df_with_preds)
    elif st.session_state.df_raw is not None:
        if st.button("Run Analysis on Uploaded Data", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()
        st.info("Click the button above or go to Upload to run the analysis first.")
    else:
        st.info("Please upload a dataset first.")

elif page == "employees":
    if st.session_state.df_with_preds is not None:
        df = st.session_state.df_with_preds
        has_pred = "Prediction" in df.columns
        groups = dict(detect_groups(df))
        st.markdown('<div class="card"><div class="card-title">Filter & Search Employees</div>', unsafe_allow_html=True)
        filters = []
        fcols = st.columns(4)
        fi = 0
        for key in ["Department", "JobRole", "Gender", "Risk_Category"]:
            if key in df.columns or (key == "Risk_Category" and has_pred):
                col_name = key if key in df.columns else "Risk_Category"
                with fcols[fi % 4]:
                    opts = ["All"] + sorted(df[col_name].dropna().unique().tolist())
                    sel = st.selectbox(key.replace("_", " "), opts, key=f"f_{key}")
                    filters.append((col_name, sel))
                fi += 1
        if "Age" in groups:
            with fcols[fi % 4]:
                age_filter = st.selectbox("Age Group", ["All", "<20", "20-29", "30-39", "40-49", "50-59", "60+"], key="f_age")
                filters.append(("_age_grp", age_filter, groups["Age"]))
            fi += 1
        filtered = df.copy()
        for f in filters:
            if len(f) == 2:
                col_name, sel = f
                if sel != "All":
                    filtered = filtered[filtered[col_name] == sel]
            elif len(f) == 3:
                col_name, sel, actual_col = f
                if sel != "All":
                    bins = [0, 20, 30, 40, 50, 60, 200]
                    labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60+"]
                    filtered["_age_grp_tmp"] = pd.cut(filtered[actual_col], bins=bins, labels=labels, right=False)
                    filtered = filtered[filtered["_age_grp_tmp"] == sel]
        st.markdown(f'<p style="font-size:14px;color:{COLORS["text_muted"]};margin:0 0 8px;">{len(filtered):,} employees match</p>', unsafe_allow_html=True)
        search_term = st.text_input("", placeholder="Search by Employee ID, Name, or any value...", label_visibility="collapsed")
        if search_term:
            mask = filtered.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            filtered = filtered[mask]
            st.markdown(f'<p style="font-size:13px;color:{COLORS["text_muted"]};">Filtered to {len(filtered):,} by search</p>', unsafe_allow_html=True)
        id_cols = ["EmployeeNumber", "EmployeeId", "EmployeeID", "emp_id", "Employee_ID", "EmpID"]
        actual_id = None
        for c in id_cols:
            if c in filtered.columns:
                actual_id = c
                break
        if actual_id is None:
            for c in filtered.columns:
                if "employee" in str(c).lower() and ("id" in str(c).lower() or "number" in str(c).lower() or "code" in str(c).lower()):
                    actual_id = c
                    break
        preferred = ["EmployeeNumber", "EmployeeId", "Age", "Department", "JobRole", "Gender", "MonthlyIncome", "YearsAtCompany"]
        if has_pred:
            preferred += ["Prediction", "Attrition_Probability", "Risk_Category"]
        display = [c for c in preferred if c in filtered.columns]
        extra = [c for c in filtered.columns if c not in display and not c.startswith("_")]
        display = display + extra[:min(8, len(extra))]
        st.dataframe(filtered[display].reset_index(drop=True), use_container_width=True, height=400)
        if actual_id and len(filtered) > 0:
            st.markdown('<div class="section-divider"><span>Employee Detail</span></div>', unsafe_allow_html=True)
            ids = filtered[actual_id].unique().tolist()
            emp_sel = st.selectbox("Select Employee", ids, format_func=lambda x: f"{actual_id}: {x}")
            row = filtered[filtered[actual_id] == emp_sel].iloc[0]
            ca, cb = st.columns(2)
            with ca:
                st.markdown(f'<div class="card"><div class="card-title">Profile</div>', unsafe_allow_html=True)
                for f in ["Age", "Gender", "Department", "JobRole", "MaritalStatus", "EducationField", "MonthlyIncome", "YearsAtCompany"]:
                    fc = find_column(df, [f])
                    if fc and fc in row:
                        st.markdown(f'<div class="detail-row"><span class="detail-label">{f}</span><span class="detail-value">{row[fc]}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with cb:
                if has_pred:
                    st.markdown(f'<div class="card"><div class="card-title">Prediction</div>', unsafe_allow_html=True)
                    pred_val = row["Prediction"]
                    prob_val = row["Attrition_Probability"]
                    risk_val = row["Risk_Category"]
                    label = "Likely to Leave" if pred_val == 1 else "Likely to Stay"
                    color = COLORS["danger"] if pred_val == 1 else COLORS["success"]
                    st.markdown(f"""
                    <div class="detail-row"><span class="detail-label">Status</span><span class="detail-value" style="color:{color};font-weight:600;">{label}</span></div>
                    <div class="detail-row"><span class="detail-label">Probability</span><span class="detail-value">{prob_val:.1%}</span></div>
                    <div class="detail-row"><span class="detail-label">Risk Level</span><span class="detail-value" style="color:{color};">{risk_val}</span></div>
                    """, unsafe_allow_html=True)
                    pred_cols = [c for c in df.columns if c not in display and c not in ["Age", "Gender", "Department", "JobRole", "MaritalStatus", "EducationField", "MonthlyIncome", "YearsAtCompany"] and not c.startswith("_")]
                    pred_cols = pred_cols[:12]
                    for c in pred_cols:
                        st.markdown(f'<div class="detail-row"><span class="detail-label">{c}</span><span class="detail-value">{row[c]}</span></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif st.session_state.df_raw is not None:
        st.info("Run analysis first from the Upload tab to enable employee filtering.")
    else:
        st.info("Please upload a dataset first.")

elif page == "export":
    if st.session_state.df_with_preds is not None:
        df = st.session_state.df_with_preds
        has_pred = "Prediction" in df.columns
        st.markdown('<div class="card"><div class="card-title">Export Report</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:13px;color:var(--text-muted,#9CA3AF);margin:0 0 12px;">Download a comprehensive PDF report with KPIs, charts, demographics, and risk analysis.</p>', unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Generate PDF Report", use_container_width=True, type="primary"):
                with st.spinner("Generating report with charts and analysis..."):
                    try:
                        pdf_bytes = generate_report(df)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            "Download PDF Report",
                            pdf_bytes,
                            f"attrition_report_{ts}.pdf",
                            "application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Report generation error: {str(e)}")
        with c2:
            csv = df.to_csv(index=False).encode("utf-8")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button("Download Enriched Data (CSV)", csv, f"attrition_enriched_{ts}.csv", "text/csv", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if has_pred:
            total = len(df)
            att = int(df["Prediction"].sum())
            stay = total - att
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px;">
                <div class="stat-card"><div class="stat-num">{total:,}</div><div class="stat-label">Total Employees</div></div>
                <div class="stat-card"><div class="stat-num" style="color:{COLORS['danger']};">{att:,}</div><div class="stat-label">Predicted to Leave</div></div>
                <div class="stat-card"><div class="stat-num" style="color:{COLORS['success']};">{stay:,}</div><div class="stat-label">Predicted to Stay</div></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f'<div class="card"><div class="card-title">Risk Breakdown</div>', unsafe_allow_html=True)
            rc = df["Risk_Category"].value_counts()
            rows = "".join(
                f'<tr><td style="padding:6px 8px;color:{COLORS["text"]};">{cat}</td><td style="padding:6px 8px;color:{COLORS["text"]};">{rc.get(cat, 0):,}</td><td style="padding:6px 8px;color:{COLORS["text_muted"]};">{rc.get(cat, 0)/total*100:.1f}%</td></tr>'
                for cat in ["High Risk", "Medium Risk", "Low Risk"]
            )
            st.markdown(f"<table style='width:100%;'><tr><th style='text-align:left;color:{COLORS['text_muted']};'>Category</th><th style='text-align:left;color:{COLORS['text_muted']};'>Count</th><th style='text-align:left;color:{COLORS['text_muted']};'>Percentage</th></tr>{rows}</table>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Please upload and analyze data first.")

elif page == "about":
    st.markdown('<div class="section-divider"><span>About AttritionIQ</span></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">How It Works</div>
            <p style="font-size:14px;color:{COLORS['text']};margin:0 0 12px;line-height:1.6;">
            AttritionIQ uses a machine learning model trained on historical HR data to predict which employees are at risk of leaving. The model analyzes patterns across <strong>30+ features</strong> including demographics, job role, compensation, tenure, satisfaction scores, and performance metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Step-by-Step Guide</div>
            <div style="display:flex;flex-direction:column;gap:10px;">
                <div style="display:flex;gap:12px;align-items:start;">
                    <div style="width:24px;height:24px;border-radius:50%;background:{COLORS['primary']};color:#fff;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;flex-shrink:0;">1</div>
                    <div><strong style="color:{COLORS['text']};">Upload Data</strong><p style="font-size:13px;color:{COLORS['text_muted']};margin:2px 0 0;">Upload a CSV or Excel file containing employee records. The system automatically detects columns even with different naming conventions.</p></div>
                </div>
                <div style="display:flex;gap:12px;align-items:start;">
                    <div style="width:24px;height:24px;border-radius:50%;background:{COLORS['primary']};color:#fff;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;flex-shrink:0;">2</div>
                    <div><strong style="color:{COLORS['text']};">Run Analysis</strong><p style="font-size:13px;color:{COLORS['text_muted']};margin:2px 0 0;">Click "Run Full Analysis" to process your data. The ML model generates predictions, attrition probabilities, and risk categories for each employee.</p></div>
                </div>
                <div style="display:flex;gap:12px;align-items:start;">
                    <div style="width:24px;height:24px;border-radius:50%;background:{COLORS['primary']};color:#fff;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;flex-shrink:0;">3</div>
                    <div><strong style="color:{COLORS['text']};">Explore Dashboard</strong><p style="font-size:13px;color:{COLORS['text_muted']};margin:2px 0 0;">View interactive charts showing attrition trends by department, job role, age, tenure, compensation, and satisfaction levels.</p></div>
                </div>
                <div style="display:flex;gap:12px;align-items:start;">
                    <div style="width:24px;height:24px;border-radius:50%;background:{COLORS['primary']};color:#fff;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;flex-shrink:0;">4</div>
                    <div><strong style="color:{COLORS['text']};">Review Employees</strong><p style="font-size:13px;color:{COLORS['text_muted']};margin:2px 0 0;">Filter and search the workforce. Drill into individual employee profiles to see their predicted attrition risk and contributing factors.</p></div>
                </div>
                <div style="display:flex;gap:12px;align-items:start;">
                    <div style="width:24px;height:24px;border-radius:50%;background:{COLORS['primary']};color:#fff;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;flex-shrink:0;">5</div>
                    <div><strong style="color:{COLORS['text']};">Export Report</strong><p style="font-size:13px;color:{COLORS['text_muted']};margin:2px 0 0;">Generate a comprehensive PDF report with all charts, KPIs, and risk analysis for sharing with stakeholders.</p></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Understanding the Results</div>
            <p style="font-size:13px;font-weight:600;color:{COLORS['text']};margin:0 0 8px;">Risk Categories</p>
            <div style="display:flex;flex-direction:column;gap:8px;margin-bottom:16px;">
                <div style="display:flex;gap:8px;align-items:center;">
                    <div style="width:12px;height:12px;border-radius:50%;background:{COLORS['success']};flex-shrink:0;"></div>
                    <div><span style="font-weight:600;font-size:13px;color:{COLORS['success']};">Low Risk</span><span style="font-size:12px;color:{COLORS['text_muted']};margin-left:6px;">(Probability &lt; 30%)</span><p style="font-size:12px;color:{COLORS['text_muted']};margin:0;">Employees unlikely to leave. Standard retention practices apply.</p></div>
                </div>
                <div style="display:flex;gap:8px;align-items:center;">
                    <div style="width:12px;height:12px;border-radius:50%;background:{COLORS['warning']};flex-shrink:0;"></div>
                    <div><span style="font-weight:600;font-size:13px;color:{COLORS['warning']};">Medium Risk</span><span style="font-size:12px;color:{COLORS['text_muted']};margin-left:6px;">(Probability 30%-60%)</span><p style="font-size:12px;color:{COLORS['text_muted']};margin:0;">Monitor these employees. Consider engagement surveys and check-ins.</p></div>
                </div>
                <div style="display:flex;gap:8px;align-items:center;">
                    <div style="width:12px;height:12px;border-radius:50%;background:{COLORS['danger']};flex-shrink:0;"></div>
                    <div><span style="font-weight:600;font-size:13px;color:{COLORS['danger']};">High Risk</span><span style="font-size:12px;color:{COLORS['text_muted']};margin-left:6px;">(Probability &gt; 60%)</span><p style="font-size:12px;color:{COLORS['text_muted']};margin:0;">Immediate attention needed. Proactive retention strategies recommended.</p></div>
                </div>
            </div>
            <p style="font-size:13px;font-weight:600;color:{COLORS['text']};margin:0 0 8px;">Required Data Fields</p>
            <p style="font-size:12px;color:{COLORS['text_muted']};margin:0 0 8px;line-height:1.5;">
            The ML model works best with <strong>10+ core columns</strong> from categories like personal info, job details, compensation, tenure, satisfaction, and performance. Fewer columns mean limited predictions but basic analytics still work.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Model Information</div>
            <div style="display:flex;flex-direction:column;gap:6px;">
                <div class="detail-row"><span class="detail-label">Algorithm</span><span class="detail-value">Logistic Regression</span></div>
                <div class="detail-row"><span class="detail-label">Features Used</span><span class="detail-value">30+ employee attributes</span></div>
                <div class="detail-row"><span class="detail-label">Output</span><span class="detail-value">Prediction + Probability</span></div>
                <div class="detail-row"><span class="detail-label">Risk Tiers</span><span class="detail-value">Low / Medium / High</span></div>
                <div class="detail-row"><span class="detail-label">Min. Columns</span><span class="detail-value">10 for predictions</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:24px 0 8px;border-top:1px solid var(--border,#E4E7EB);margin-top:48px;">
    <p style="font-size:12px;color:var(--text-muted,#9CA3AF);">AttritionIQ  •  Employee Attrition Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)
