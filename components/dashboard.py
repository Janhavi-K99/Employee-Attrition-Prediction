import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from utils.preprocessing import find_column, detect_groups

COLORS = {
    "primary": "#2563EB", "primary_light": "#60A5FA", "danger": "#DC2626",
    "success": "#059669", "warning": "#D97706", "info": "#0284C7",
    "dark": "#1F2937", "card": "#FFFFFF", "border": "#E4E7EB",
    "text": "#1F2937", "text_muted": "#9CA3AF"
}

PALETTES = {
    "diverging": ["#059669", "#A7F3D0", "#FDE68A", "#FCA5A5", "#DC2626"],
    "sequential": ["#2563EB", "#60A5FA", "#93C5FD", "#A7F3D0", "#059669"],
    "categorical": ["#2563EB", "#059669", "#DC2626", "#D97706", "#0284C7", "#8B5CF6", "#14B8A6"]
}

def make_chart(fig, height=380):
    fig.update_layout(
        template="plotly_white",
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=45, b=10),
        font=dict(color=COLORS["text"], size=11),
        title_font=dict(size=14, color=COLORS["text"]),
        hovermode="x unified"
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.04)", zerolinecolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.04)", zerolinecolor="rgba(0,0,0,0.06)")
    return fig

def render_grouped_bar_chart(df, group_col, value_col, title, agg="mean", ascending=True):
    grp = df.groupby(group_col)[value_col].agg(agg).reset_index()
    grp = grp.sort_values(value_col, ascending=ascending)
    fig = px.bar(grp, x=group_col, y=value_col, color=value_col,
                 color_continuous_scale="Reds_r" if agg == "mean" else "Blues",
                 text=grp[value_col].round(2))
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    return make_chart(fig.update_layout(title=title, xaxis_title="", yaxis_title=""))

def categorical_attrition_bar(df, cat_col, title):
    if cat_col not in df.columns: return None
    grp = df.groupby(cat_col)["Prediction"].mean().reset_index()
    grp.columns = [cat_col, "Attrition Rate"]
    grp["Attrition Rate"] = (grp["Attrition Rate"] * 100).round(1)
    grp = grp.sort_values("Attrition Rate", ascending=True)
    is_long = len(grp) > 6
    fig = px.bar(grp,
                 y=cat_col if is_long else None,
                 x="Attrition Rate" if is_long else cat_col,
                 color="Attrition Rate", color_continuous_scale="Reds_r",
                 text=grp["Attrition Rate"].astype(str) + "%",
                 title=title)
    fig.update_traces(textposition="outside")
    return make_chart(fig, height=320 + max(0, len(grp) * 12))

def render_pie(df, col, title):
    if col not in df.columns: return None
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "Count"]
    colors = PALETTES["categorical"][:len(counts)]
    fig = px.pie(counts, values="Count", names=col, title=title,
                 color_discrete_sequence=colors, hole=0.4)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return make_chart(fig, 360)

def render_box(df, cat_col, num_col, title):
    if cat_col not in df.columns or num_col not in df.columns: return None
    fig = px.box(df, x=cat_col, y=num_col, color=cat_col, title=title)
    return make_chart(fig, 380)

def render_histogram(df, col, title, bins=20):
    if col not in df.columns: return None
    fig = px.histogram(df, x=col, nbins=bins, title=title,
                       color_discrete_sequence=[COLORS["primary"]])
    return make_chart(fig, 360)

def render_line(df, x_col, y_col, title):
    if x_col not in df.columns or y_col not in df.columns: return None
    grp = df.groupby(x_col)[y_col].mean().reset_index()
    fig = px.line(grp, x=x_col, y=y_col, markers=True, title=title)
    fig.update_traces(line_color=COLORS["primary"], marker=dict(color=COLORS["primary"], size=5))
    return make_chart(fig, 360)


PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>'
STACKED_DASHBOARD_CSS = """
<style>
.dashboard-container { max-width: 1200px; margin: 0 auto; }
.dashboard-kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }
.dashboard-kpi { background: #fff; border: 1px solid #E4E7EB; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); transition: transform 0.2s, box-shadow 0.2s; }
.dashboard-kpi:hover { transform: translateY(-2px); box-shadow: 0 4px 24px rgba(0,0,0,0.08); }
.dashboard-kpi-label { font-size: 12px; color: #9CA3AF; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.8px; }
.dashboard-kpi-value { font-size: 28px; font-weight: 700; }
.dashboard-section { display: flex; align-items: center; gap: 12px; margin: 28px 0 20px; }
.dashboard-section::before, .dashboard-section::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, transparent, #E4E7EB, transparent); }
.dashboard-section span { font-size: 13px; font-weight: 600; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1.5px; }
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.chart-cell { background: #fff; border: 1px solid #E4E7EB; border-radius: 12px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow: hidden; }
.chart-cell-full { grid-column: 1 / -1; }
@media (max-width: 768px) { .chart-grid { grid-template-columns: 1fr; } .dashboard-kpis { grid-template-columns: repeat(2, 1fr); } }
</style>
"""

def _fig_to_div(fig):
    if fig is None:
        return ""
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)

def _kpi_html(df):
    has_pred = "Prediction" in df.columns
    total = len(df)
    cards = []
    cards.append(f'<div class="dashboard-kpi" style="border-left:3px solid {COLORS["primary"]};"><div class="dashboard-kpi-label">Total Employees</div><div class="dashboard-kpi-value" style="color:{COLORS["primary"]};">{total:,}</div></div>')
    if has_pred:
        att_count = int(df["Prediction"].sum())
        stay_count = total - att_count
        att_rate = (att_count / total * 100)
        cards.append(f'<div class="dashboard-kpi" style="border-left:3px solid {COLORS["danger"]};"><div class="dashboard-kpi-label">Attrition Rate</div><div class="dashboard-kpi-value" style="color:{COLORS["danger"]};">{att_rate:.1f}%</div></div>')
        cards.append(f'<div class="dashboard-kpi" style="border-left:3px solid {COLORS["danger"]};"><div class="dashboard-kpi-label">Likely to Leave</div><div class="dashboard-kpi-value" style="color:{COLORS["danger"]};">{att_count:,}</div></div>')
        cards.append(f'<div class="dashboard-kpi" style="border-left:3px solid {COLORS["success"]};"><div class="dashboard-kpi-label">Likely to Stay</div><div class="dashboard-kpi-value" style="color:{COLORS["success"]};">{stay_count:,}</div></div>')
    else:
        cards.append(f'<div class="dashboard-kpi" style="border-left:3px solid {COLORS["primary"]};"><div class="dashboard-kpi-label">Data Fields</div><div class="dashboard-kpi-value" style="color:{COLORS["primary"]};">{len(df.columns)}</div></div>')
        cards.append('<div class="dashboard-kpi"></div><div class="dashboard-kpi"></div>')
    return f'<div class="dashboard-kpis">{"".join(cards)}</div>'


def render_dashboard_to_html(df):
    groups = dict(detect_groups(df))
    col_map = groups
    has_pred = "Prediction" in df.columns

    parts = [PLOTLY_CDN, STACKED_DASHBOARD_CSS, '<div class="dashboard-container">']

    parts.append(_kpi_html(df))

    parts.append('<div class="dashboard-section"><span>Analytics Dashboard</span></div>')

    charts = []

    if has_pred:
        if "Department" in col_map:
            charts.append(categorical_attrition_bar(df, col_map["Department"], "Department-wise Attrition"))
        if "JobRole" in col_map:
            charts.append(categorical_attrition_bar(df, col_map["JobRole"], "Job Role-wise Attrition"))

    if "MonthlyIncome" in col_map:
        if has_pred:
            dfc = df.copy()
            dfc["_status"] = dfc["Prediction"].map({1: "Leave", 0: "Stay"})
            charts.append(render_box(dfc, "_status", col_map["MonthlyIncome"], "Monthly Income vs Attrition"))
        else:
            charts.append(render_histogram(df, col_map["MonthlyIncome"], "Monthly Income Distribution"))

    if "YearsAtCompany" in col_map:
        if has_pred:
            charts.append(render_line(df, col_map["YearsAtCompany"], "Prediction", "Tenure vs Attrition"))
        else:
            charts.append(render_histogram(df, col_map["YearsAtCompany"], "Years at Company Distribution"))

    if has_pred:
        if "WorkLifeBalance" in col_map:
            charts.append(categorical_attrition_bar(df, col_map["WorkLifeBalance"], "Work-Life Balance vs Attrition"))
        if "OverTime" in col_map:
            charts.append(categorical_attrition_bar(df, col_map["OverTime"], "Overtime vs Attrition"))
        if "JobSatisfaction" in col_map:
            charts.append(categorical_attrition_bar(df, col_map["JobSatisfaction"], "Job Satisfaction vs Attrition"))

    if has_pred:
        rc_col = "Risk_Category" if "Risk_Category" in df.columns else "Risk Category"
        if rc_col in df.columns:
            charts.append(render_grouped_bar_chart(
                df.groupby(rc_col).size().reset_index(name="Count"),
                rc_col, "Count", "Risk Distribution", agg="sum", ascending=False
            ))

    if "Age" in col_map:
        bins = [0, 20, 30, 40, 50, 60, 200]
        labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60+"]
        dfc = df.copy()
        dc = col_map["Age"]
        dfc["_age_grp"] = pd.cut(dfc[dc], bins=bins, labels=labels, right=False)
        ag = dfc["_age_grp"].value_counts().sort_index().reset_index()
        ag.columns = ["Age Group", "Count"]
        fig = px.bar(ag, x="Age Group", y="Count", color="Count",
                     color_continuous_scale="Purples", text="Count", title="Age Distribution")
        charts.append(make_chart(fig, 340))

    if "Gender" in col_map:
        charts.append(render_pie(df, col_map["Gender"], "Gender Distribution"))
    if "Education" in col_map:
        charts.append(render_pie(df, col_map["Education"], "Education Distribution"))
    if "MaritalStatus" in col_map:
        charts.append(render_pie(df, col_map["MaritalStatus"], "Marital Status Distribution"))

    if not has_pred:
        for key in ["MonthlyIncome", "Age", "YearsAtCompany", "DailyRate", "HourlyRate", "TotalWorkingYears"]:
            if key in col_map:
                charts.append(render_histogram(df, col_map[key], f"{key} Distribution"))
                break
        for key in ["Department", "JobRole", "EducationField", "BusinessTravel"]:
            if key in col_map:
                counts = df[col_map[key]].value_counts().reset_index()
                counts.columns = [col_map[key], "Count"]
                fig = px.bar(counts, x=col_map[key], y="Count", color="Count",
                             color_continuous_scale="Viridis", title=f"{key} Distribution")
                charts.append(make_chart(fig, 340))
                break

    charts = [c for c in charts if c is not None]

    parts.append('<div class="chart-grid">')
    for i, fig in enumerate(charts):
        full = i == len(charts) - 1 and len(charts) % 2 == 1
        cls = "chart-cell chart-cell-full" if full else "chart-cell"
        parts.append(f'<div class="{cls}">{_fig_to_div(fig)}</div>')
    parts.append('</div>')

    parts.append('</div>')
    return '\n'.join(parts)
