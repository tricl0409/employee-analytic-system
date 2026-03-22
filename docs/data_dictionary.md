# Data Dictionary — Employee Census Income Dataset

## Overview

| Property | Value |
|----------|-------|
| **File** | `data/uploads/employee_data.csv` |
| **Origin** | U.S. Census Bureau (Adult Income dataset) |
| **Rows** | 32,561 |
| **Columns** | 15 |
| **Target Variable** | `Income` (binary classification) |
| **Total Missing Cells** | 10,094 (2.07% of all cells) |

> [!NOTE]
> Dataset này được sử dụng để phân tích và dự đoán mức thu nhập của nhân viên (≤50K hoặc >50K) dựa trên các đặc điểm nhân khẩu học, giáo dục, và nghề nghiệp.

---

## Column Specifications

### Numeric Columns (6)

#### `Age`
| Property | Value |
|----------|-------|
| **Dtype** | `float64` |
| **Description** | Tuổi của nhân viên |
| **Range** | 17 – 90 |
| **Mean / Median** | 38.57 / 37.0 |
| **Std Dev** | 13.65 |
| **Missing** | 1,628 (5.0%) |
| **Unique Values** | 73 |
| **Distribution** | Right-skewed (majority 25–50) |

#### `Fnlwgt`
| Property | Value |
|----------|-------|
| **Dtype** | `int64` |
| **Description** | Final weight — số người mà bản ghi đại diện trong tổng thể dân số. Giá trị do Census Bureau tính toán dựa trên các yếu tố nhân khẩu học |
| **Range** | 12,285 – 1,484,705 |
| **Mean / Median** | 189,778.37 / 178,356.0 |
| **Std Dev** | 105,549.98 |
| **Missing** | 0 |
| **Unique Values** | 21,648 |
| **Distribution** | Right-skewed with heavy tail |

#### `Education_Num`
| Property | Value |
|----------|-------|
| **Dtype** | `int64` |
| **Description** | Mã số thứ tự trình độ học vấn (numeric encoding của cột `Education`) |
| **Range** | 1 – 16 |
| **Mean / Median** | 10.08 / 10.0 |
| **Std Dev** | 2.57 |
| **Missing** | 0 |
| **Unique Values** | 16 |
| **Mapping** | 1 = Preschool → 16 = Doctorate |

#### `Capital_Gain`
| Property | Value |
|----------|-------|
| **Dtype** | `int64` |
| **Description** | Thu nhập từ lãi vốn (capital gains) trong năm, tính bằng USD |
| **Range** | 0 – 99,999 |
| **Mean / Median** | 1,077.65 / 0.0 |
| **Std Dev** | 7,385.29 |
| **Missing** | 0 |
| **Unique Values** | 119 |
| **Distribution** | Extremely right-skewed — ~91.7% giá trị = 0 |

#### `Capital_Loss`
| Property | Value |
|----------|-------|
| **Dtype** | `int64` |
| **Description** | Thua lỗ từ vốn (capital losses) trong năm, tính bằng USD |
| **Range** | 0 – 4,356 |
| **Mean / Median** | 87.30 / 0.0 |
| **Std Dev** | 402.96 |
| **Missing** | 0 |
| **Unique Values** | 92 |
| **Distribution** | Extremely right-skewed — ~95.2% giá trị = 0 |

#### `Hours_per_Week`
| Property | Value |
|----------|-------|
| **Dtype** | `float64` |
| **Description** | Số giờ làm việc trung bình mỗi tuần |
| **Range** | 1 – 99 |
| **Mean / Median** | 40.44 / 40.0 |
| **Std Dev** | 12.33 |
| **Missing** | 1,628 (5.0%) |
| **Unique Values** | 94 |
| **Distribution** | Near-normal, centered at 40 (standard work week) |

---

### Categorical Columns (9)

#### `Workclass`
| Property | Value |
|----------|-------|
| **Dtype** | `object` |
| **Description** | Loại hình tổ chức / khu vực việc làm |
| **Missing** | 2,279 (7.0%) |
| **Unique Values** | 9 |

| Value | Count | Percentage |
|-------|-------|------------|
| Private | 21,127 | 64.9% |
| Self-emp-not-inc | 2,354 | 7.2% |
| Local-gov | 1,936 | 5.9% |
| ? | 1,716 | 5.3% |
| State-gov | 1,201 | 3.7% |
| Self-emp-inc | 1,116 | 3.4% |
| Federal-gov | 960 | 2.9% |
| Without-pay | 14 | 0.04% |
| Never-worked | 7 | 0.02% |

> [!IMPORTANT]
> Giá trị `?` (1,716 records) là noise token — sẽ được chuyển thành NaN tại Step 2 (Noise Cleaning) của preprocessing pipeline.

#### `Education`
| Property | Value |
|----------|-------|
| **Dtype** | `object` |
| **Description** | Trình độ học vấn cao nhất đạt được |
| **Missing** | 0 |
| **Unique Values** | 16 |

| Value | Count | Percentage |
|-------|-------|------------|
| HS-grad | 10,501 | 32.3% |
| Some-college | 7,291 | 22.4% |
| Bachelors | 5,355 | 16.4% |
| Masters | 1,723 | 5.3% |
| Assoc-voc | 1,382 | 4.2% |
| 11th | 1,175 | 3.6% |
| Assoc-acdm | 1,067 | 3.3% |
| 10th | 933 | 2.9% |
| 7th-8th | 646 | 2.0% |
| Prof-school | 576 | 1.8% |
| 9th | 514 | 1.6% |
| 12th | 433 | 1.3% |
| Doctorate | 413 | 1.3% |
| 5th-6th | 333 | 1.0% |
| 1st-4th | 168 | 0.5% |
| Preschool | 51 | 0.2% |

#### `Marital_Status`
| Property | Value |
|----------|-------|
| **Dtype** | `object` |
| **Description** | Tình trạng hôn nhân |
| **Missing** | 0 |
| **Unique Values** | 7 |

| Value | Count | Percentage |
|-------|-------|------------|
| Married-civ-spouse | 14,976 | 46.0% |
| Never-married | 10,683 | 32.8% |
| Divorced | 4,443 | 13.6% |
| Separated | 1,025 | 3.1% |
| Widowed | 993 | 3.0% |
| Married-spouse-absent | 418 | 1.3% |
| Married-AF-spouse | 23 | 0.07% |

#### `Occupation`
| Property | Value |
|----------|-------|
| **Dtype** | `object` |
| **Description** | Nghề nghiệp / vị trí công việc |
| **Missing** | 2,605 (8.0%) |
| **Unique Values** | 15 |

| Value | Count | Percentage |
|-------|-------|------------|
| Prof-specialty | 3,813 | 11.7% |
| Craft-repair | 3,766 | 11.6% |
| Exec-managerial | 3,761 | 11.6% |
| Adm-clerical | 3,456 | 10.6% |
| Sales | 3,367 | 10.3% |
| Other-service | 3,295 | 10.1% |
| Machine-op-inspct | 1,966 | 6.0% |
| ? | 1,717 | 5.3% |
| Transport-moving | 1,572 | 4.8% |
| Handlers-cleaners | 1,370 | 4.2% |
| Farming-fishing | 989 | 3.0% |
| Tech-support | 914 | 2.8% |
| Protective-serv | 649 | 2.0% |
| Priv-house-serv | 149 | 0.5% |
| Armed-Forces | 9 | 0.03% |

#### `Relationship`
| Property | Value |
|----------|-------|
| **Dtype** | `object` |
| **Description** | Vai trò quan hệ trong gia đình |
| **Missing** | 0 |
| **Unique Values** | 6 |

| Value | Count | Percentage |
|-------|-------|------------|
| Husband | 13,193 | 40.5% |
| Not-in-family | 8,305 | 25.5% |
| Own-child | 5,068 | 15.6% |
| Unmarried | 3,446 | 10.6% |
| Wife | 1,568 | 4.8% |
| Other-relative | 981 | 3.0% |

#### `Race`
| Property | Value |
|----------|-------|
| **Dtype** | `object` |
| **Description** | Chủng tộc |
| **Missing** | 0 |
| **Unique Values** | 5 |

| Value | Count | Percentage |
|-------|-------|------------|
| White | 27,816 | 85.4% |
| Black | 3,124 | 9.6% |
| Asian-Pac-Islander | 1,039 | 3.2% |
| Amer-Indian-Eskimo | 311 | 1.0% |
| Other | 271 | 0.8% |

#### `Sex`
| Property | Value |
|----------|-------|
| **Dtype** | `object` |
| **Description** | Giới tính |
| **Missing** | 0 |
| **Unique Values** | 2 |

| Value | Count | Percentage |
|-------|-------|------------|
| Male | 21,790 | 66.9% |
| Female | 10,771 | 33.1% |

#### `Native_Country`
| Property | Value |
|----------|-------|
| **Dtype** | `object` |
| **Description** | Quốc gia xuất thân |
| **Missing** | 1,954 (6.0%) |
| **Unique Values** | 41 |

| Top 5 Values | Count | Percentage |
|---------------|-------|------------|
| United-States | 27,424 | 84.2% |
| Mexico | 602 | 1.8% |
| ? | 542 | 1.7% |
| Philippines | 187 | 0.6% |
| Germany | 130 | 0.4% |

> [!NOTE]
> 41 quốc gia trong dataset. Phần lớn (84.2%) là United-States. Các quốc gia còn lại mỗi nước chiếm < 2%.

#### `Income` *(Target Variable)*
| Property | Value |
|----------|-------|
| **Dtype** | `object` |
| **Description** | Mức thu nhập hàng năm — biến mục tiêu cho classification |
| **Missing** | 0 |
| **Unique Values** | 2 |

| Value | Count | Percentage |
|-------|-------|------------|
| ≤50K | 24,720 | 75.9% |
| >50K | 7,841 | 24.1% |

> [!WARNING]
> Dataset bị **class imbalance**: tỷ lệ ≤50K : >50K ≈ 3:1. Cần lưu ý khi xây dựng classification model (xem xét SMOTE, class weights, hoặc stratified sampling).

---

## Data Quality Summary

### Missing Values

| Column | Missing Count | Missing % | Nature |
|--------|--------------|-----------|--------|
| `Occupation` | 2,605 | 8.0% | Structural (noise `?`) |
| `Workclass` | 2,279 | 7.0% | Mixed (NaN + noise `?`) |
| `Native_Country` | 1,954 | 6.0% | Mixed (NaN + noise `?`) |
| `Age` | 1,628 | 5.0% | True missing |
| `Hours_per_Week` | 1,628 | 5.0% | True missing |
| Other columns | 0 | 0% | Complete |

### Noise Tokens

| Token | Columns Affected | Est. Count |
|-------|-----------------|------------|
| `?` | Workclass, Occupation, Native_Country | ~3,975 |

> [!IMPORTANT]
> Các giá trị `?` sẽ được pipeline tự động chuyển thành NaN tại **Step 2 — Noise Cleaning**, sau đó impute tại **Step 4 — Missing Value Imputation**.

### Potential Issues

| Issue | Detail |
|-------|--------|
| **High cardinality** | `Fnlwgt` (21,648 unique), `Native_Country` (41 unique) |
| **Redundant pair** | `Education` ↔ `Education_Num` (encoding tương đương) |
| **Zero-inflated** | `Capital_Gain` (~91.7% = 0), `Capital_Loss` (~95.2% = 0) |
| **Heavy tail outliers** | `Fnlwgt` max = 1,484,705, `Capital_Gain` max = 99,999 |
| **Class imbalance** | Target `Income`: 75.9% ≤50K vs 24.1% >50K |

---

## Column Relationships

### Redundancy Map

```mermaid
graph LR
    Education -->|"numeric encoding"| Education_Num
    Marital_Status -->|"subset relationship"| Relationship
```

- **Education ↔ Education_Num**: 1:1 mapping. Pipeline sẽ drop `Education` tại Step 8 (Feature Encoding) vì `Education_Num` đã tồn tại.
- **Marital_Status ↔ Relationship**: Có tương quan cao nhưng không hoàn toàn redundant (e.g., "Husband" và "Wife" đều map sang "Married-civ-spouse").

### Variable Nature Classification

| Nature | Columns | Count |
|--------|---------|-------|
| **Continuous** | Age, Fnlwgt, Capital_Gain, Capital_Loss, Hours_per_Week | 5 |
| **Discrete Ordinal** | Education_Num | 1 |
| **Nominal** | Workclass, Education, Marital_Status, Occupation, Relationship, Race, Native_Country | 7 |
| **Binary** | Sex, Income | 2 |

---

## Pipeline Transformation Preview

Đây là tóm tắt các biến đổi mà mỗi column sẽ trải qua trong preprocessing pipeline:

| Column | Step 2 (Noise) | Step 4 (Impute) | Step 5 (Outlier) | Step 6 (Log) | Step 7 (Bin/Map) | Step 8 (Encode) |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| Age | — | Mean/Median | Z-Score/IQR | — | ✅ Binning | Label |
| Workclass | ✅ `?`→NaN | Mode | — | — | ✅ Mapping | One-Hot |
| Fnlwgt | — | — | IQR/MZS | ✅ log1p | — | — |
| Education | — | — | — | — | ✅ Mapping | **Drop** |
| Education_Num | — | — | — | — | — | — |
| Marital_Status | — | — | — | — | ✅ Mapping | One-Hot |
| Occupation | ✅ `?`→NaN | Mode | — | — | ✅ Mapping | One-Hot |
| Relationship | — | — | — | — | — | One-Hot |
| Race | — | — | — | — | — | One-Hot |
| Sex | — | — | — | — | — | Label |
| Capital_Gain | — | — | MZS | ✅ log1p | — | — |
| Capital_Loss | — | — | MZS | ✅ log1p | — | — |
| Hours_per_Week | — | Mean/Median | Z-Score/IQR | — | ✅ Binning | Label |
| Native_Country | ✅ `?`→NaN | Mode | — | — | — | One-Hot |
| Income | — | — | — | — | — | Label |

> [!NOTE]
> Bảng trên là **dự kiến** dựa trên đặc điểm phân phối hiện tại. Method thực tế (Z-Score vs IQR vs Modified Z-Score) sẽ được auto-select tại runtime dựa trên skewness của dữ liệu sau các bước xử lý trước đó.
