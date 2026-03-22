# Cramér's V — Lý thuyết chi tiết

## 1. Mục tiêu

Đo **mức độ liên kết (association)** giữa 2 biến **categorical** (định danh/phân loại). Kết quả nằm trong khoảng:

| Giá trị V | Ý nghĩa |
|-----------|---------|
| **0** | Hai biến hoàn toàn **độc lập** (không có mối liên hệ) |
| **1** | Hai biến có mối liên hệ **hoàn hảo** |
| 0.1 – 0.3 | Liên kết **yếu** |
| 0.3 – 0.5 | Liên kết **trung bình** |
| > 0.5 | Liên kết **mạnh** |

---

## 2. Các bước thực hiện

### Bước 1 — Xây dựng Contingency Table (Bảng chéo)

Đếm tần suất xuất hiện đồng thời của từng cặp giá trị giữa 2 biến.

**Ví dụ:** Biến A = `education` (3 giá trị), Biến B = `income` (2 giá trị):

| | ≤50K | >50K | **Tổng hàng** |
|---|---|---|---|
| **HS-grad** | 800 | 100 | 900 |
| **Bachelors** | 400 | 300 | 700 |
| **Advanced** | 100 | 300 | 400 |
| **Tổng cột** | 1300 | 700 | **n = 2000** |

Trong code:

```python
contingency = pd.crosstab(col_a, col_b)
# → Ma trận r × c (ở đây: 3 × 2)
```

---

### Bước 2 — Kiểm định Chi-Square (χ²)

Ý tưởng: So sánh **giá trị thực tế** (Observed = O) với **giá trị kỳ vọng** (Expected = E) nếu 2 biến là hoàn toàn độc lập.

#### 2a. Tính Expected frequency cho mỗi ô

$$E_{ij} = \frac{\text{Tổng hàng}_i \times \text{Tổng cột}_j}{n}$$

**Ví dụ:** Ô (HS-grad, ≤50K):

$$E = \frac{900 \times 1300}{2000} = 585$$

Nếu 2 biến thật sự độc lập thì ta **kỳ vọng** 585 người HS-grad thuộc nhóm ≤50K, nhưng **thực tế** có 800 → sai lệch lớn → có liên kết.

#### 2b. Tính Chi-Square statistic

$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

- Duyệt qua **tất cả** các ô trong bảng chéo
- Mỗi ô: lấy (Thực tế − Kỳ vọng)², chia cho Kỳ vọng
- Cộng tất cả lại → χ²

> **Ý nghĩa**: χ² càng **lớn** → sự khác biệt giữa thực tế và kỳ vọng càng **nhiều** → 2 biến càng **không độc lập**.

Trong code:

```python
chi2_stat = chi2_contingency(contingency)[0]
# scipy tự tính Expected rồi trả về χ²
```

---

### Bước 3 — Chuẩn hóa thành Cramér's V

**Vấn đề:** χ² phụ thuộc vào:
- **n** (sample size) — càng nhiều dữ liệu thì χ² càng lớn
- **Kích thước bảng** (r × c) — bảng lớn hơn cho χ² lớn hơn

→ Không so sánh trực tiếp được. Cần **chuẩn hóa** về thang [0, 1].

**Công thức:**

$$V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1, \; c-1)}}$$

Trong đó:

| Ký hiệu | Ý nghĩa |
|----------|---------|
| χ² | Chi-Square statistic (bước 2) |
| n | Tổng số quan sát |
| r | Số hàng trong contingency table (số giá trị biến A) |
| c | Số cột (số giá trị biến B) |
| min(r-1, c-1) | Giới hạn trên lý thuyết — đảm bảo V ≤ 1 |

Trong code:

```python
n_obs = len(col_a)                      # n
min_dim = min(contingency.shape) - 1    # min(r-1, c-1)
V = np.sqrt(chi2_stat / (n_obs * min_dim))
```

**Ví dụ tiếp:** Giả sử χ² = 450, n = 2000, bảng 3×2:
- min(3-1, 2-1) = min(2, 1) = 1
- V = √(450 / (2000 × 1)) = √0.225 ≈ **0.474**
- → Liên kết **trung bình – mạnh** giữa education và income

---

## 3. Tóm tắt dạng pipeline

```
Biến A (categorical) + Biến B (categorical)
        │
        ▼
   ┌─────────────────────┐
   │ 1. Contingency Table │  ← Đếm tần suất đồng thời
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │ 2. Chi-Square (χ²)  │  ← So sánh Observed vs Expected
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │ 3. Chuẩn hóa → V   │  ← Chia cho n × min(r-1, c-1), rồi sqrt
   └──────────┬──────────┘
              ▼
        V ∈ [0, 1]
```

---

## 4. Trong context project (Employee Analytics System)

Ở `_compute_association_scores()` trong `pages/eda.py`:

- Khi gặp cột **categorical** (ví dụ: `marital_status`, `occupation`) → dùng **Cramér's V**
- Khi gặp cột **numeric** (ví dụ: `age`, `hours_per_week`) → dùng **Point-Biserial**

Logic chuyển biến `income` thành binary `"0"`/`"1"` rồi tính Cramér's V giữa cặp (categorical × binary_income). Đây là cách chuẩn vì income binary vẫn là biến categorical (2 giá trị: ≤50K / >50K).

### Implementation trong code

```python
def _cramers_v(col_a: pd.Series, col_b: pd.Series) -> float:
    contingency = pd.crosstab(col_a, col_b)       # Bước 1
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0                                 # Guard: không đủ variation
    chi2_stat = chi2_contingency(contingency)[0]   # Bước 2
    n_obs = len(col_a)
    min_dim = min(contingency.shape) - 1           # min(r-1, c-1)
    if min_dim == 0 or n_obs == 0:
        return 0.0                                 # Guard: tránh chia cho 0
    return float(np.sqrt(chi2_stat / (n_obs * min_dim)))  # Bước 3
```

---

## 5. Lưu ý nâng cao

### Bias Correction (Cramér's V hiệu chỉnh)

Phiên bản standard Cramér's V có thể **overestimate** ở sample size nhỏ. Phiên bản corrected (Bergsma, 2013) sử dụng:

$$\tilde{\phi}^2 = \max\left(0, \; \frac{\chi^2}{n} - \frac{(r-1)(c-1)}{n-1}\right)$$

$$\tilde{r} = r - \frac{(r-1)^2}{n-1}, \quad \tilde{c} = c - \frac{(c-1)^2}{n-1}$$

$$\tilde{V} = \sqrt{\frac{\tilde{\phi}^2}{\min(\tilde{r}-1, \; \tilde{c}-1)}}$$

Tuy nhiên, **với dataset lớn** (hàng chục ngàn rows như Adult dataset), standard Cramér's V là hoàn toàn phù hợp.

---

## Tham khảo

- Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.
- Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T. *Journal of the Korean Statistical Society*, 42(3), 323-328.
- `scipy.stats.chi2_contingency`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
