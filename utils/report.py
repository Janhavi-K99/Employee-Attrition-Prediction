import io
import os
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from fpdf import FPDF
from utils.preprocessing import detect_groups, find_column, MODEL_REQUIRED_COLS

COLORS = {
    "primary": "#2563EB", "danger": "#DC2626", "success": "#059669",
    "warning": "#D97706", "info": "#0284C7", "text": "#1F2937", "text_muted": "#9CA3AF"
}

PALETTE = ["#2563EB", "#059669", "#DC2626", "#D97706", "#0284C7", "#8B5CF6", "#14B8A6", "#F59E0B"]


def _fig_to_bytes(fig, width=800, height=400):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=10),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    return pio.to_image(fig, format="png", width=width, height=height)


def _chart_card(fig):
    if fig is None:
        return None
    return _fig_to_bytes(fig)


def _kpi_chart(df):
    has_pred = "Prediction" in df.columns
    total = len(df)
    if has_pred:
        att = int(df["Prediction"].sum())
        stay = total - att
        rate = att / total * 100
    fig = px.bar(
        x=["Total Employees", "Likely to Leave", "Likely to Stay"],
        y=[total, att, stay] if has_pred else [total, 0, 0],
        color=["Total", "Leave", "Stay"],
        color_discrete_map={"Total": COLORS["primary"], "Leave": COLORS["danger"], "Stay": COLORS["success"]},
        text=[str(total), str(att), str(stay)] if has_pred else [str(total), "N/A", "N/A"],
        title="Workforce Overview"
    )
    fig.update_traces(textposition="outside", showlegend=False)
    fig.update_yaxes(visible=False, showticklabels=False)
    return _fig_to_bytes(fig, height=350)


def _attrition_by_col(df, col, title):
    if col not in df.columns or "Prediction" not in df.columns:
        return None
    grp = df.groupby(col)["Prediction"].mean().reset_index()
    grp["Attrition Rate"] = (grp["Prediction"] * 100).round(1)
    grp = grp.sort_values("Attrition Rate", ascending=True)
    fig = px.bar(grp, x="Attrition Rate", y=col, orientation="h",
                 color="Attrition Rate", color_continuous_scale="Reds_r",
                 text=grp["Attrition Rate"].astype(str) + "%",
                 title=title)
    fig.update_traces(textposition="outside")
    h = max(300, len(grp) * 35 + 80)
    return _fig_to_bytes(fig, height=int(h))


def _age_distribution(df, age_col):
    if age_col not in df.columns:
        return None
    bins = [0, 20, 30, 40, 50, 60, 200]
    labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60+"]
    dfc = df.copy()
    dfc["_age_grp"] = pd.cut(dfc[age_col], bins=bins, labels=labels, right=False)
    ag = dfc["_age_grp"].value_counts().sort_index().reset_index()
    ag.columns = ["Age Group", "Count"]
    fig = px.bar(ag, x="Age Group", y="Count", color="Count",
                 color_continuous_scale="Blues", text="Count", title="Age Distribution")
    fig.update_traces(textposition="outside")
    return _fig_to_bytes(fig)


def _income_box(df, status_col, income_col):
    if status_col not in df.columns or income_col not in df.columns:
        return None
    fig = px.box(df, x=status_col, y=income_col, color=status_col, title="Monthly Income Distribution")
    return _fig_to_bytes(fig)


def _tenure_line(df, tenure_col):
    if tenure_col not in df.columns or "Prediction" not in df.columns:
        return None
    grp = df.groupby(tenure_col)["Prediction"].mean().reset_index()
    grp["Attrition Rate"] = grp["Prediction"] * 100
    fig = px.line(grp, x=tenure_col, y="Attrition Rate", markers=True, title="Tenure vs Attrition Rate")
    fig.update_traces(line_color=COLORS["primary"], marker=dict(color=COLORS["primary"], size=5))
    fig.update_yaxes(ticksuffix="%")
    return _fig_to_bytes(fig)


def _risk_chart(df):
    if "Risk_Category" not in df.columns:
        return None
    rc = df["Risk_Category"].value_counts().reset_index()
    rc.columns = ["Risk", "Count"]
    fig = px.bar(rc, x="Risk", y="Count", color="Risk",
                 color_discrete_map={"High Risk": COLORS["danger"], "Medium Risk": COLORS["warning"], "Low Risk": COLORS["success"]},
                 text="Count", title="Risk Distribution")
    fig.update_traces(textposition="outside", showlegend=False)
    return _fig_to_bytes(fig)


def _pie_chart(df, col, title):
    if col not in df.columns:
        return None
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "Count"]
    fig = px.pie(counts, values="Count", names=col, title=title,
                 color_discrete_sequence=PALETTE, hole=0.4)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return _fig_to_bytes(fig, height=350)


def _histogram(df, col, title):
    if col not in df.columns:
        return None
    fig = px.histogram(df, x=col, nbins=20, title=title,
                       color_discrete_sequence=[COLORS["primary"]])
    return _fig_to_bytes(fig)


class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*self._hex_to_rgb(COLORS["text_muted"]))
            self.cell(0, 8, "AttritionIQ  |  Employee Attrition Intelligence Report", align="L")
            self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="R", new_x="LMARGIN", new_y="NEXT")
            self.line(10, 14, 200, 14)
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*self._hex_to_rgb(COLORS["text_muted"]))
        self.cell(0, 10, f"Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", align="C")

    @staticmethod
    def _hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*self._hex_to_rgb(COLORS["primary"]))
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self._hex_to_rgb(COLORS["primary"]))
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def add_image_centered(self, img_bytes, max_w=180):
        img = io.BytesIO(img_bytes)
        from PIL import Image
        pil_img = Image.open(img)
        iw, ih = pil_img.size
        if iw > max_w:
            ratio = max_w / iw
            iw = max_w
            ih = int(ih * ratio)
        x = (210 - iw) / 2
        self.image(img, x=x, w=iw, h=ih)
        self.ln(4)

    def kpi_table(self, data):
        self.set_font("Helvetica", "", 10)
        cols_w = [45, 45, 45, 45]
        headers = list(data.keys())
        self.set_fill_color(*self._hex_to_rgb("#EBF5FF"))
        self.set_draw_color(*self._hex_to_rgb("#E4E7EB"))
        for i, h in enumerate(headers):
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*self._hex_to_rgb(COLORS["text"]))
            self.cell(cols_w[i], 8, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 14)
        vals = list(data.values())
        for i, v in enumerate(vals):
            if "leave" in str(v).lower() or "leave" in str(headers[i]).lower():
                self.set_text_color(*self._hex_to_rgb(COLORS["danger"]))
            elif "stay" in str(headers[i]).lower() or "stay" in str(v).lower():
                self.set_text_color(*self._hex_to_rgb(COLORS["success"]))
            else:
                self.set_text_color(*self._hex_to_rgb(COLORS["text"]))
            self.cell(cols_w[i], 10, str(v), border=1, align="C")
        self.ln(8)

    def data_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(*self._hex_to_rgb("#F3F4F6"))
        self.set_draw_color(*self._hex_to_rgb("#D1D5DB"))
        self.set_text_color(*self._hex_to_rgb(COLORS["text"]))
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 7)
        for row in rows:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell)[:30], border=1, align="C")
            self.ln()
            if self.get_y() > 270:
                self.add_page()


def generate_report(df):
    has_pred = "Prediction" in df.columns
    groups = dict(detect_groups(df))
    col_map = groups

    pdf = PDFReport()
    pdf.alias_nb_pages()

    # ---- COVER PAGE ----
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 32)
    pdf.set_text_color(*pdf._hex_to_rgb(COLORS["primary"]))
    pdf.cell(0, 14, "AttritionIQ", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(*pdf._hex_to_rgb(COLORS["text"]))
    pdf.cell(0, 10, "Employee Attrition Intelligence Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_draw_color(*pdf._hex_to_rgb(COLORS["primary"]))
    pdf.set_line_width(0.8)
    mid = 105
    pdf.line(mid - 30, pdf.get_y(), mid + 30, pdf.get_y())
    pdf.ln(14)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*pdf._hex_to_rgb(COLORS["text_muted"]))
    date_str = datetime.now().strftime("%B %d, %Y")
    pdf.cell(0, 8, f"Report Date: {date_str}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Total Records: {len(df):,}", align="C", new_x="LMARGIN", new_y="NEXT")
    if has_pred:
        att = int(df["Prediction"].sum())
        pdf.cell(0, 8, f"Attrition Rate: {att/len(df)*100:.1f}%", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*PDFReport._hex_to_rgb(COLORS["text_muted"]))
    pdf.cell(0, 8, "This report is automatically generated by the AttritionIQ platform.", align="C", new_x="LMARGIN", new_y="NEXT")

    # ---- EXECUTIVE SUMMARY ----
    pdf.add_page()
    pdf.section_title("Executive Summary")
    if has_pred:
        total = len(df)
        att_count = int(df["Prediction"].sum())
        stay_count = total - att_count
        att_rate = att_count / total * 100
        pdf.kpi_table({
            "Total Employees": f"{total:,}",
            "Attrition Rate": f"{att_rate:.1f}%",
            "Likely to Leave": f"{att_count:,}",
            "Likely to Stay": f"{stay_count:,}"
        })
        pdf.ln(4)
        img = _kpi_chart(df)
        if img:
            pdf.add_image_centered(img)
        pdf.ln(4)
        if "Risk_Category" in df.columns:
            rc = df["Risk_Category"].value_counts()
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*pdf._hex_to_rgb(COLORS["text"]))
            for cat in ["High Risk", "Medium Risk", "Low Risk"]:
                cnt = rc.get(cat, 0)
                pct = cnt / total * 100
                pdf.cell(0, 7, f"  {cat}: {cnt:,} employees ({pct:.1f}%)",
                         new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*pdf._hex_to_rgb(COLORS["text_muted"]))
        pdf.cell(0, 7, "Basic analytics available. Upload additional data for predictions.", new_x="LMARGIN", new_y="NEXT")
        total = len(df)
        pdf.kpi_table({"Total Employees": f"{total:,}", "Data Fields": f"{len(df.columns)}", "": "", "": ""})

    # ---- ATTRITION ANALYSIS ----
    if has_pred:
        pdf.add_page()
        pdf.section_title("Attrition by Department")
        dept_col = col_map.get("Department")
        if dept_col:
            img = _attrition_by_col(df, dept_col, "Attrition Rate by Department")
            if img:
                pdf.add_image_centered(img)

        pdf.section_title("Attrition by Job Role")
        role_col = col_map.get("JobRole")
        if role_col:
            img = _attrition_by_col(df, role_col, "Attrition Rate by Job Role")
            if img:
                pdf.add_image_centered(img)

    # ---- DEMOGRAPHICS ----
    pdf.add_page()
    pdf.section_title("Demographics")

    age_col = col_map.get("Age")
    if age_col:
        img = _age_distribution(df, age_col)
        if img:
            pdf.add_image_centered(img)

    gender_col = col_map.get("Gender")
    if gender_col:
        img = _pie_chart(df, gender_col, "Gender Distribution")
        if img:
            pdf.add_image_centered(img, max_w=120)

    marital_col = col_map.get("MaritalStatus")
    if marital_col:
        img = _pie_chart(df, marital_col, "Marital Status")
        if img:
            pdf.add_image_centered(img, max_w=120)

    edu_col = col_map.get("Education")
    if edu_col:
        img = _pie_chart(df, edu_col, "Education Distribution")
        if img:
            pdf.add_image_centered(img, max_w=120)

    # ---- COMPENSATION & TENURE ----
    pdf.add_page()
    pdf.section_title("Compensation & Tenure")

    if has_pred:
        income_col = col_map.get("MonthlyIncome")
        if income_col:
            dfc = df.copy()
            dfc["_status"] = dfc["Prediction"].map({1: "Leave", 0: "Stay"})
            img = _income_box(dfc, "_status", income_col)
            if img:
                pdf.add_image_centered(img)

        tenure_col = col_map.get("YearsAtCompany")
        if tenure_col:
            img = _tenure_line(df, tenure_col)
            if img:
                pdf.add_image_centered(img)
    else:
        income_col = col_map.get("MonthlyIncome")
        if income_col:
            img = _histogram(df, income_col, "Monthly Income Distribution")
            if img:
                pdf.add_image_centered(img)

    # ---- WORK-LIFE & SATISFACTION ----
    if has_pred:
        pdf.add_page()
        pdf.section_title("Work-Life & Satisfaction")

        wlb_col = col_map.get("WorkLifeBalance")
        if wlb_col:
            img = _attrition_by_col(df, wlb_col, "Work-Life Balance vs Attrition")
            if img:
                pdf.add_image_centered(img)

        ot_col = col_map.get("OverTime")
        if ot_col:
            img = _attrition_by_col(df, ot_col, "Overtime vs Attrition")
            if img:
                pdf.add_image_centered(img)

        sat_col = col_map.get("JobSatisfaction")
        if sat_col:
            img = _attrition_by_col(df, sat_col, "Job Satisfaction vs Attrition")
            if img:
                pdf.add_image_centered(img)

        img = _risk_chart(df)
        if img:
            pdf.add_image_centered(img)

    # ---- RISK BREAKDOWN TABLE ----
    if has_pred and "Risk_Category" in df.columns:
        pdf.add_page()
        pdf.section_title("Risk Breakdown")
        rc = df["Risk_Category"].value_counts()
        total = len(df)
        pdf.data_table(
            ["Risk Category", "Count", "Percentage"],
            [[cat, f"{rc.get(cat, 0):,}", f"{rc.get(cat, 0)/total*100:.1f}%"]
             for cat in ["High Risk", "Medium Risk", "Low Risk"]],
            col_widths=[63, 63, 64]
        )

        if "Department" in df.columns and "Prediction" in df.columns:
            pdf.ln(6)
            pdf.section_title("Attrition by Department (Detail)")
            dept_stats = df.groupby("Department").agg(
                Total=("Prediction", "count"),
                Leaving=("Prediction", "sum")
            ).reset_index()
            dept_stats["Rate"] = (dept_stats["Leaving"] / dept_stats["Total"] * 100).round(1).astype(str) + "%"
            pdf.data_table(
                ["Department", "Total", "Leaving", "Rate"],
                dept_stats.values.tolist(),
                col_widths=[60, 43, 43, 44]
            )

    # ---- HIGH RISK EMPLOYEES ----
    if has_pred and "Risk_Category" in df.columns:
        pdf.add_page()
        pdf.section_title("High-Risk Employees")
        high_risk = df[df["Risk_Category"] == "High Risk"].copy()
        if len(high_risk) > 0:
            id_col = None
            for c in ["EmployeeNumber", "EmployeeId", "EmployeeID", "emp_id"]:
                if c in high_risk.columns:
                    id_col = c
                    break
            cols = [id_col] if id_col else []
            for c in ["Age", "Department", "JobRole", "MonthlyIncome", "YearsAtCompany"]:
                fc = find_column(df, [c])
                if fc:
                    cols.append(fc)
            cols = cols[:5]
            if len(cols) > 0:
                display_cols = [c for c in cols if c in high_risk.columns]
                if len(display_cols) > 0:
                    data_rows = high_risk[display_cols].head(50).values.tolist()
                    cw = [190 / len(display_cols)] * len(display_cols)
                    pdf.data_table(display_cols, data_rows, col_widths=cw)
                    if len(high_risk) > 50:
                        pdf.set_font("Helvetica", "I", 8)
                        pdf.set_text_color(*pdf._hex_to_rgb(COLORS["text_muted"]))
                        pdf.cell(0, 6, f"... and {len(high_risk) - 50} more high-risk employees",
                                 new_x="LMARGIN", new_y="NEXT")

    # ---- OUTPUT ----
    return bytes(pdf.output())
