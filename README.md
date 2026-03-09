# 💎 Employee Analytic System

A comprehensive, end-to-end **HR data analytics platform** built with Python and Streamlit. The system guides users through the full analytics pipeline — from raw data upload all the way to AI-powered conclusions and recommendations.

---

## ✨ Features

| Module | Description |
|---|---|
| 🔐 **Authentication** | Role-based login system (Admin / User) with secure password hashing and session management |
| 📋 **Overview** | Workspace dashboard — active file status, file metadata, data preview |
| 🔍 **Data Audit** | Automated quality check: missing values, duplicates, outliers, type inconsistencies |
| ⚙️ **Preprocessing** | Smart cleaning pipeline — garbage removal, imputation, outlier treatment with safe zones |
| 📊 **EDA** | Exploratory Data Analysis with interactive Plotly charts and data insights |
| 🧩 **Feature Preparation** | Feature engineering and data transformation for downstream modeling |
| 📝 **Conclusion & Recommendation** | AI-generated (Gemini) summary report, insights, and PDF export |
| 👥 **User Management** | Admin panel to create, edit, and deactivate user accounts *(Admin only)* |
| ⚙️ **Analytic Rule Settings** | Configure audit thresholds, safe zones, and outlier rules *(Admin only)* |

---

## 🏗️ Project Structure

```
Employee_Analytic_System/
├── app.py                      # Entry point — navigation & auth guard
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── pages/                      # Streamlit page modules
│   ├── login.py
│   ├── overview.py
│   ├── data_audit.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_preparation.py
│   ├── conclusion.py
│   └── management/
│       ├── user_management.py
│       └── analytic_rule_settings.py
│
├── modules/
│   ├── core/                   # Business logic engines
│   │   ├── auth_engine.py      # Authentication & user management
│   │   ├── data_engine.py      # Data loading & standardization
│   │   ├── audit_engine.py     # Data quality audit logic
│   │   ├── preprocessing_engine.py  # Cleaning pipeline
│   │   ├── llm_engine.py       # Google Gemini AI integration
│   │   └── report_engine.py    # PDF report generation (fpdf2)
│   ├── ui/                     # UI components & styling
│   │   ├── components.py
│   │   ├── styles.py
│   │   ├── icons.py
│   │   ├── dialogs.py
│   │   └── visualizer.py
│   └── utils/                  # Utilities
│       ├── db_config_manager.py
│       ├── helpers.py
│       ├── localization.py     # EN / VI language support
│       └── theme_manager.py
│
├── assets/                     # Fonts, images, video background
└── data/                       # Runtime data (not committed)
    ├── uploads/
    └── temp/
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/tricl0409/employee-analytic-system.git
cd employee-analytic-system

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.streamlit/secrets.toml` file (not committed) with your API key:

```toml
[gemini]
api_key = "YOUR_GOOGLE_GEMINI_API_KEY"
```

### Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔐 Default Credentials

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `admin123` |

> ⚠️ Change the default admin password after first login.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Framework | [Streamlit](https://streamlit.io/) 1.54 |
| Data Processing | [Pandas](https://pandas.pydata.org/) 3.0, [NumPy](https://numpy.org/) 2.3 |
| Visualization | [Plotly](https://plotly.com/python/) 6.5 |
| PDF Reports | [fpdf2](https://py-pdf.github.io/fpdf2/) 2.8 |
| AI / LLM | [Google Gemini](https://ai.google.dev/) via `google-genai` 1.65 |
| Database | SQLite (via `sqlite3`) |

---

## 🌐 Localization

The system supports **English** and **Vietnamese** — switchable at runtime from the sidebar.

---

## 📄 License

This project is for internal use at **Cranes FPT**. All rights reserved.
