import io
import os
import warnings
import tempfile
from datetime import datetime
from fpdf import FPDF
import pandas as pd
import plotly.graph_objects as go

# ==============================================================================
# BRANDING CONSTANTS
# ==============================================================================
BRAND_ORANGE = (242, 112, 36)   # #F27024
BRAND_BLUE   = (59, 130, 246)   # #3B82F6
TEXT_MAIN    = (40, 40, 40)
TEXT_MUTED   = (120, 120, 120)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_FAMILY  = "Arial"
FONT_PATH    = os.path.join(BASE_DIR, "assets", "fonts", "arial.ttf")
FONT_PATH_B  = os.path.join(BASE_DIR, "assets", "fonts", "arialbd.ttf")
FONT_PATH_I  = os.path.join(BASE_DIR, "assets", "fonts", "ariali.ttf")

class AuditReport(FPDF):
    """
    Custom FPDF class for generating the "The Transformers" Data Integrity Audit Report.
    """
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
        # Add Unicode fonts to support Vietnamese and symbols
        if os.path.exists(FONT_PATH):
            self.add_font(FONT_FAMILY, "", FONT_PATH, uni=True)
            self.add_font(FONT_FAMILY, "B", FONT_PATH_B, uni=True)
            self.add_font(FONT_FAMILY, "I", FONT_PATH_I, uni=True)
        else:
            warnings.warn(
                f"report_engine: custom font not found at '{FONT_PATH}'. "
                "PDF report will use FPDF default fonts (Vietnamese diacritics may not render correctly).",
                stacklevel=2,
            )

    def header(self):
        # Document Title (Center)
        self.set_font(FONT_FAMILY, 'B', 16)
        self.set_text_color(*BRAND_BLUE)
        self.cell(0, 10, "DATA INTEGRITY AUDIT REPORT", border=0, ln=True, align="C")
        
        # Divider Line
        self.set_draw_color(200, 200, 200)
        self.line(10, 22, 200, 22)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font(FONT_FAMILY, 'I', 8)
        self.set_text_color(*TEXT_MUTED)
        
        # Timestamp
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 10, f"Confidential - Generated on {date_str}", align="L")
        
        # Page Number
        self.set_x(0)
        self.cell(0, 10, f"Page {self.page_no()}", align="R")

    # --- HELPER FUNCTIONS ---
    def chapter_title(self, title):
        self.set_font(FONT_FAMILY, 'B', 14)
        self.set_text_color(*TEXT_MAIN)
        self.cell(0, 10, title.upper(), ln=True, align="L")
        self.ln(2)

# ==============================================================================
# REPORT LOGIC
# ==============================================================================

def draw_executive_summary(pdf, audit_data, dataset_name):
    """Draws the main statistics block."""
    pdf.add_page()
    pdf.chapter_title("1. Executive Summary")
    
    pdf.set_font(FONT_FAMILY, '', 11)
    pdf.set_text_color(*TEXT_MAIN)
    pdf.multi_cell(0, 6, f"Dataset Analyzed: {dataset_name}\n"
                         f"This report outlines the structural health, missing values, and anomalies detected "
                         f"during the integrity scan phase of the preprocessing pipeline.")
    pdf.ln(5)

    # Health Score
    health_score = audit_data.get('health_score', 0)
    pdf.set_font(FONT_FAMILY, 'B', 12)
    pdf.set_text_color(*BRAND_ORANGE)
    pdf.cell(0, 8, f"Global Health Score: {health_score:.1f}%", ln=True)
    pdf.ln(3)

    # Table
    pdf.set_fill_color(245, 245, 245)
    pdf.set_text_color(*TEXT_MAIN)
    pdf.set_font(FONT_FAMILY, 'B', 10)
    
    col_width = 45
    headers = ["Total Rows", "Total Columns", "Duplicate Rows", "Overall Missing %"]
    for h in headers:
        pdf.cell(col_width, 8, h, border=1, fill=True, align='C')
    pdf.ln()

    pdf.set_font(FONT_FAMILY, '', 10)
    stats = [
        f"{audit_data.get('total_rows', 0):,}",
        f"{audit_data.get('total_columns', 0):,}",
        f"{audit_data.get('duplicate_rows', 0):,}",
        f"{audit_data.get('missing_percentage', 0):.2f}%"
    ]
    for s in stats:
        pdf.cell(col_width, 8, s, border=1, align='C')
    pdf.ln(10)

def draw_missing_profile(pdf, df_missing):
    """Draws a zebra-striped table of missing configurations."""
    pdf.chapter_title("2. Missing Value Profile")
    if df_missing is None or df_missing.empty:
        pdf.set_font(FONT_FAMILY, 'I', 10)
        pdf.cell(0, 8, "No missing values detected. Dataset is perfectly complete.", ln=True)
        pdf.ln(5)
        return

    # Table Header
    pdf.set_fill_color(*BRAND_BLUE)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(FONT_FAMILY, 'B', 10)
    
    col_widths = [80, 50, 50]
    headers = ["Attribute Name", "Missing Count", "Missing Percentage"]
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, fill=True, align='C')
    pdf.ln()

    # Table Rows
    pdf.set_text_color(*TEXT_MAIN)
    pdf.set_font(FONT_FAMILY, '', 10)
    
    fill = False
    for _, row in df_missing.iterrows():
        # Zebra Striping colors
        if fill:
            pdf.set_fill_color(240, 245, 255)
        else:
            pdf.set_fill_color(255, 255, 255)
            
        pdf.cell(col_widths[0], 8, str(row.get('Column', '')), border=1, fill=True, align='L')
        pdf.cell(col_widths[1], 8, f"{row.get('Missing Count', 0):,}", border=1, fill=True, align='C')
        pct = row.get('Percentage', 0)
        pdf.cell(col_widths[2], 8, f"{pct:.2f}%", border=1, fill=True, align='C')
        pdf.ln()
        fill = not fill
    
    pdf.ln(10)

def draw_outlier_detection(pdf, outlier_dict, image_paths):
    """Documents the IQR analysis formulas and embeddings."""
    pdf.add_page()
    pdf.chapter_title("3. Outlier Detection Analysis")
    
    # Mathematical Formulas
    pdf.set_font(FONT_FAMILY, '', 10)
    pdf.multi_cell(0, 6, "Outliers are flagged using the standard Interquartile Range (IQR) method:")
    
    pdf.set_font(FONT_FAMILY, 'I', 10)
    pdf.set_x(15)
    pdf.cell(0, 6, "IQR = Q3 - Q1", ln=True, align='C')
    pdf.set_x(15)
    pdf.cell(0, 6, "Normal Range = [ Q1 - 1.5 * IQR ,  Q3 + 1.5 * IQR ]", ln=True, align='C')
    pdf.ln(5)

    if not outlier_dict:
        pdf.set_font(FONT_FAMILY, '', 10)
        pdf.cell(0, 6, "No significant outliers detected in numeric fields.", ln=True)
        return

    # List fields with anomalies
    pdf.set_font(FONT_FAMILY, 'B', 10)
    pdf.cell(0, 6, "Numeric Fields with Detected Anomalies (IQR Bounds):", ln=True)
    pdf.set_font(FONT_FAMILY, '', 10)
    
    for col, count in outlier_dict.items():
        pdf.cell(5)
        pdf.cell(0, 6, f"- {col}: {count:,} outliers", ln=True)
    pdf.ln(5)

    # Attach Temporary Plotly Charts if passed
    for img_path in image_paths:
        if os.path.exists(img_path):
            # Constrain image width to fit A4 (roughly 190mm max printable)
            pdf.image(img_path, w=180)
            pdf.ln(5)

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def generate_audit_report(audit_data: dict, dataset_name: str = "Employee Dataset") -> io.BytesIO:
    """
    Main pipeline to construct the PDF and return it as a Bytes stream.
    Requires audit_data dict containing:
      - health_score
      - total_rows, total_columns, duplicate_rows, missing_percentage
      - missing_df: pd.DataFrame
      - outlier_dict: dict mapped like { 'age': 456, 'income': 120 }
      - outlier_figs: list of plotly go.Figure (optional)
    """
    pdf = AuditReport()
    
    # 1. Executive Summary
    draw_executive_summary(pdf, audit_data, dataset_name)
    
    # 2. Missing Profile
    missing_df = audit_data.get('missing_df')
    draw_missing_profile(pdf, missing_df)
    
    # 3. Outlier Images processing via tempfile
    outlier_dict = audit_data.get('outlier_dict', {})
    outlier_figs = audit_data.get('outlier_figs', [])
    
    temp_files = []
    try:
        # Save plots to temporary PNGs
        for fig in outlier_figs:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.write_image(tmp.name, width=800, height=400, scale=2) # High Res
            temp_files.append(tmp.name)
            tmp.close()

        draw_outlier_detection(pdf, outlier_dict, temp_files)
        
    finally:
        # Cleanup critical to avoid storage bloat on server
        for tmp_file in temp_files:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception as e:
                    print(f"Failed to remove temp chart {tmp_file}: {e}")

    # Export to BytesIO
    pdf_buffer = io.BytesIO()
    # Output to buffer instead of file
    pdf_content = pdf.output(dest='S') # 'S' returns the document as a bytearray natively in fpdf2

    pdf_buffer.write(pdf_content)
    pdf_buffer.seek(0)
    return pdf_buffer
