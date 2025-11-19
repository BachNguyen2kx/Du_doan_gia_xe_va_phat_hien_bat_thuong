# 🏍️ DỰ ĐOÁN GIÁ & PHÁT HIỆN BẤT THƯỜNG XE MÁY CŨ – CHỢ TỐT

## 👥 Thành viên thực hiện
- **Võ Thị Hoàng Anh** – Xây dựng mô hình dự đoán giá, soạn thuyết trình.  
- **Nguyễn Xuân Bách** – Khám phá & xử lý dữ liệu, phát hiện bất thường, kiểm tra code.  
- **Cả hai** – Viết báo cáo, so sánh kết quả và chuẩn bị slide.

---
## 🗂️ 0. CẤU TRÚC FILE & THƯ MỤC (theo Google Drive)

    📦 DL07_K308_VoThiHoangAnh_XuanBach
    ├── 📁 Data
    │   ├── data_motobikes.xlsx                  ← Dữ liệu thô (xe máy)
    │   └── du_lieu_xe_may_da_tien_xu_ly_1.xlsx  ← Bản chuẩn hoá & thêm feature
    │
    ├── 📄 Mô tả bộ dữ liệu Chợ Tốt.pdf           ← File mô tả dữ liệu nguồn
    │
    ├── 📁 models_final_project_1_bai2
    │   ├── 🧠 Mô hình bài 1 + 2 (anomaly)
    │   └── ⚙️ spark-3.5.1-bin-hadoop3            ← Spark runtime (local)
    ├── 📘price_pipeline_module.py   
    ├── 📁 RandomForest_model_pyspark_b1_1
    ├── 📘 Project1_HoangAnh_XuanBach.ipynb
    └── 📊 Project01_HoangAnh_XuanBach.pptx



> **Lưu ý:** Khi chạy Colab → mount Drive, đổi đường dẫn `/content/drive/MyDrive/...`
---
## 📑 MỤC LỤC

1. [🧭 I. BUSINESS UNDERSTANDING](#i-business-understanding)
2. [🧮 II. DATA UNDERSTANDING](#ii-data-understanding)
3. [⚙️ III. DATA PREPARATION](#iii-data-preparation)
4. [🤖 IV. MODELING (REGRESSION)](#iv-modeling-regression)
5. [🚨 V. ANOMALY DETECTION](#v-anomaly-detection)
6. [📈 VI. KẾT QUẢ & BIỂU ĐỒ](#vi-ket-qua-bieu-do)
7. [🧪 VII. CHẠY NOTEBOOK](#vii-chay-notebook)
8. [⚙️ VIII. TRIỂN KHAI PIPELINE DỰ ĐOÁN GIÁ & PHÁT HIỆN BẤT THƯỜNG](#viii-trien-khai-pipeline)
9. [🪄 IX. KẾT LUẬN](#ix-ket-luan)


---
<a id="i-business-understanding"></a>
## 🧭 I. BUSINESS UNDERSTANDING

- **Bối cảnh:** Chợ Tốt – nền tảng mua bán trực tuyến lớn tại Việt Nam.  
- **Mục tiêu:**
  - Dự đoán giá hợp lý cho xe máy cũ khi đăng bán.  
  - Phát hiện bài đăng có giá bất thường (quá cao/thấp).  
- **Lợi ích:**
  - Người bán → Gợi ý giá phù hợp.  
  - Nền tảng → Tự động kiểm duyệt bài đăng bất hợp lý.

---
<a id="ii-data-understanding"></a>
## 🧮 II. DATA UNDERSTANDING

### 📦 Nguồn dữ liệu
Dữ liệu được thu thập từ **các tin rao xe máy cũ trên Chợ Tốt**, gồm nhiều thương hiệu, loại xe, khu vực khác nhau.

### 🧱 Các trường dữ liệu chính
| Tên cột | Kiểu | Mô tả |
|----------|------|-------|
| **id** | int64 | Mã định danh duy nhất của tin đăng |
| **Tiêu_đề** | object | Tiêu đề bài đăng |
| **Giá** | object | Giá rao bán (VNĐ, đôi khi có đơn vị hoặc text kèm) |
| **Khoảng_giá_min** | object | Giá thấp nhất (nếu có dạng khoảng giá) |
| **Khoảng_giá_max** | object | Giá cao nhất (nếu có dạng khoảng giá) |
| **Địa_chỉ** | object | Địa điểm đăng bán (quận/huyện, thành phố) |
| **Mô_tả_chi_tiết** | object | Thông tin chi tiết người bán nhập |
| **Thương_hiệu** | object | Thương hiệu xe (Honda, Yamaha, BMW, v.v.) |
| **Dòng_xe** | object | Model cụ thể của xe |
| **Năm_đăng_ký** | object | Năm đăng ký hoặc sản xuất xe |
| **Số_Km_đã_đi** | int64 | Số km đã đi (mileage) |
| **Tình_trạng** | object | Tình trạng xe (mới, cũ, đã sửa chữa,...) |
| **Loại_xe** | object | Dòng xe: tay ga, số, côn tay, mô tô,... |
| **Dung_tích_xe** | object | Dung tích xi-lanh (cc) |
| **Xuất_xứ** | object | Nơi sản xuất (VN, Nhật, Thái,...) |
| **Chính_sách_bảo_hành** | object | Có hoặc không có bảo hành |
| **Trọng_lượng** | object | Trọng lượng xe (nếu có) |
| **Href** | object | Đường link gốc đến tin đăng |

### 🧰 Thư viện & công cụ
`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`,  
`xgboost`, `lightgbm`, `catboost`, `pyspark`, `ydata-profiling`, ...

---
<a id="iii-data-preparation"></a>
## ⚙️ III. DATA PREPARATION

- Làm sạch:
  - Loại bỏ giá trị null, trùng, hoặc giá không hợp lệ (“Liên hệ”).  
  - Chuẩn hoá đơn vị, chuyển “Giá” sang numeric.  
- Xử lý văn bản:
  - Tách dấu, lowercase, loại stopwords, rút trích keyword.  
- Mã hoá biến phân loại:
  - LabelEncoder cho Spark, OneHotEncoder cho sklearn.  
- Tạo đặc trưng mới:
  - `tuoi_xe = 2025 - Năm_đăng_ký`  
  - `gia/km = Giá / Số_Km_đã_đi`  
  - `brand_avg_price`, `district_avg_price`  
- Chia tập train/test: 80%–20%  
- Lưu dataset clean lại trong `du_lieu_xe_may_da_tien_xu_ly_1.xlsx`

---
<a id="iv-modeling-regression"></a>
## 🤖 IV. MODELING (Regression)

### 🔹 PySpark Models

| Mô hình | Ưu điểm | Giải thích & Ứng dụng |
|---|---|---|
| 🌲 **Random Forest** | Ổn định, dễ song song hóa, chống overfitting | Phù hợp dữ liệu lớn, chạy tốt trên Spark cluster. |
| 🚀 **Gradient Boosting (GBTRegressor)** | Độ chính xác cao, tối ưu mạnh trong Spark MLlib | Dự đoán giá xe liên tục, giảm sai số hiệu quả. |
| 🧮 **Ridge Regression** | Nhanh, nhẹ, dễ giải thích hệ số | Baseline tuyến tính, kiểm tra ảnh hưởng từng biến. |

**Kết quả:**  
- R² ~ **0.80**  
- Random Forest và GBT cho hiệu năng tốt, Ridge làm baseline.  
- Model được lưu tại `GradientBoosting_model_pyspark_b1_1/`.

---

### 🔹 ML Truyền thống (sklearn / boosting)

| Mô hình | Ưu điểm nổi bật | Giải thích & Ứng dụng |
|---|---|---|
| 🌲 **Random Forest** | Ổn định, kháng nhiễu, ít tuning | Baseline mạnh, phù hợp bài toán tabular. |
| 🚀 **Gradient Boosting** | Chính xác cao, xử lý tốt phi tuyến | Boosting tuần tự, giảm lỗi còn lại. |
| 🐱 **CatBoost** | Xử lý category tốt, không cần mã hoá | Thường outperform GBT khi có nhiều biến phân loại. |
| ⚡ **XGBoost** | Nhanh, hiệu quả, tinh chỉnh linh hoạt | Rất phổ biến trong bài toán giá. |
| 💡 **LightGBM** | Tiết kiệm RAM, huấn luyện nhanh | Phù hợp tập lớn, nhiều chiều. |
| 🧮 **Ridge Regression** | Đơn giản, dễ hiểu | Giải thích tác động từng biến đầu vào. |

**Kết quả:**  
- R² ~ **0.86–0.89**, MAE thấp.  
- Random Forest và XGBoost là mô hình tốt nhất.  
- Model lưu tại `models_final_project_1_bai2/`.

---
<a id="v-anomaly-detection"></a>
## 🚨 V. ANOMALY DETECTION

- Kết hợp **rule-based** + **ML-based**:
  - A = |Z-score| của phần dư (Giá – Giá_dự_đoán)
  - B = điểm bất thường (Isolation Forest / LOF)
  - Tổng hợp:  
    ```
    abnormal_score = 0.3*A + 0.7*B
    if abnormal_score ≥ 60 → flag bất thường
    ```
- Output gồm:
  | Cột | Ý nghĩa |
  |------|----------|
  | price_pred | Giá dự đoán |
  | abnormal_score | Điểm bất thường (0–100) |
  | flag | 1 = bất thường, 0 = bình thường |

---
<a id="vi-ket-qua-bieu-do"></a>
## 📈 VI. KẾT QUẢ & BIỂU ĐỒ

### 🔹 EDA
- **Histogram & Boxplot (Giá)** → phát hiện lệch phải.  
- **Heatmap tương quan** → `Dung_tích_xe`, `Thương_hiệu`, `Tuổi_xe` ảnh hưởng mạnh.  
- **Chi-square test** → `Loại_xe`, `Tình_trạng`, `Khu_vực` có mối quan hệ với giá.

### 🔹 Regression
- **Predicted vs Actual** → tuyến tính ổn định, không overfit.  
- **Residual plot** → phần dư quanh 0, mô hình tốt.  
- **Feature importance** → top đặc trưng: thương hiệu, dung tích, tuổi xe.

### 🔹 Anomaly
| Nhóm | Số lượng | Tỷ lệ | Nhận xét |
|------|-----------|--------|----------|
| Bình thường | 3,835 | 58% | Giá hợp lý |
| Vi phạm min/max | 2,327 | 35% | Giá lệch khung |
| Bất thường mạnh | 382 | 6% | Cần kiểm duyệt thủ công |

---
<a id="vii-chay-notebook"></a>
## 🧪 VII. CHẠY NOTEBOOK

### ⚙️ Cài đặt
pip install pyspark==3.5.1 scikit-learn xgboost lightgbm catboost pandas numpy seaborn matplotlib

---

### 🧱 **Bước 1: Chuẩn bị môi trường & nạp dữ liệu**

- Mở file **`Project1_HoangAnh_XuanBach.ipynb`**.  
- Chạy cell đầu tiên để import các thư viện (`pandas`, `numpy`, `seaborn`, `matplotlib`, `pyspark`, v.v.) và khởi tạo **SparkSession** nếu dùng PySpark.  
- Xác định đường dẫn dữ liệu:
  - `Data/data_motobikes.xlsx` → Dữ liệu gốc.  
  - `Data/du_lieu_xe_may_da_tien_xu_ly.xlsx` → Dữ liệu đã làm sạch.  
  - `Data/du_lieu_xe_may_da_tien_xu_ly_1.xlsx` → Dữ liệu hoàn thiện, có thêm đặc trưng (nên dùng file này để train).

---

### 🧹 **Bước 2: Tiền xử lý dữ liệu**

- Chạy cell làm sạch dữ liệu:
  - Loại bỏ các giá trị null, trùng lặp hoặc giá trị “Liên hệ”.  
  - Chuyển các cột `Giá`, `Số_Km_đã_đi`, `Năm_đăng_ký` về kiểu số (`float/int`).  
  - Tạo thêm đặc trưng:
    - `Tuổi_xe = 2025 - Năm_đăng_ký`  
    - `Giá_trên_km = Giá / Số_Km_đã_đi`
  - Chuẩn hóa đơn vị, xử lý văn bản mô tả (nếu có).

👉 Kết thúc bước này, kiểm tra bằng `df.info()` và `df.describe()` để đảm bảo dữ liệu đã sạch.

---

### 📊 **Bước 3: Phân tích dữ liệu (EDA)**

- Chạy cell hiển thị biểu đồ:
  - **Histogram** phân phối giá (phát hiện lệch phải).  
  - **Boxplot** để tìm ngoại lệ.  
  - **Heatmap** thể hiện tương quan giữa các đặc trưng.  
  - **Barplot** theo thương hiệu, loại xe, khu vực.  
- Nhận định:
  - `Dung_tích_xe`, `Tuổi_xe`, `Thương_hiệu` có ảnh hưởng mạnh đến giá.  
  - Các khu vực trung tâm thường có giá cao hơn vùng ven.

---

### ⚙️ **Bước 4: Huấn luyện mô hình Regression**

#### 🔹 **ML truyền thống (scikit-learn)**
- Chạy cell chia dữ liệu train/test (tỉ lệ 80/20).  
- Thử nghiệm các mô hình:
  - `RandomForestRegressor`
  - `GradientBoostingRegressor`
  - `XGBoost`
  - `LightGBM`
  - `Ridge`
- So sánh kết quả qua các chỉ số:
  - `R²`, `MAE`, `RMSE`  
- Lưu mô hình tốt nhất vào thư mục **`models_final_project_1_bai2/`**.

#### 🔹 **PySpark MLlib**
- Chạy cell tạo `Pipeline` với `StringIndexer`, `VectorAssembler` và `GBTRegressor`.  
- Đánh giá bằng `RegressionEvaluator` (metric `r2`).  
- Lưu model Spark vào **`GradientBoosting_model_pyspark_b1_1/`**.

---

### 📈 **Bước 5: Đánh giá mô hình**

- Quan sát biểu đồ **Predicted vs Actual** → đường xu hướng gần chéo, sai số nhỏ.  
- Kiểm tra **Residual Plot** để xác định xem mô hình có bias không.  
- Xem **Feature Importance**:
  - Các biến có sức ảnh hưởng lớn: `Thương_hiệu`, `Dung_tích_xe`, `Tuổi_xe`.

🧮 **Kết quả mẫu:**
| Mô hình | R² | Ghi chú |
|----------|----|---------|
| RandomForest (sklearn) | 0.87 | Chính xác, ổn định |
| GradientBoosting (sklearn) | 0.86 | Hiệu suất cao, dễ tuning |
| GBTRegressor (PySpark) | 0.80 | Tốt trên dữ liệu lớn |

---

### 🚨 **Bước 6: Phát hiện bất thường (Anomaly Detection)**

- Chạy cell tính **phần dư** = Giá thật – Giá dự đoán.  
- Tính **Z-score** để đo mức lệch (điểm A).  
- Áp dụng mô hình bất thường:
  - `IsolationForest` hoặc `LocalOutlierFactor` → tính điểm B.  
- Kết hợp thành:
abnormal_score = 0.3 * A + 0.7 * B
flag = 1 nếu abnormal_score ≥ 60

- Đánh dấu bài đăng có giá quá cao/thấp bất thường.  
- Kiểm tra tỉ lệ flag ≈ 5–6%.

---

### 💾 **Bước 7: Lưu kết quả**

- Chạy cell cuối để xuất kết quả:
- → chứa `price_pred`, `abnormal_score`, `flag`.  
- Nếu có batch mới (`new_batch.xlsx`) → mô hình sẽ dự đoán thêm và lưu vào `Data/scored_batch.xlsx`.

---

### ✅ **Tổng kết mục VII**
| Cell | Nội dung | Kết quả |
|------|-----------|----------|
| Import + setup | Khởi tạo môi trường, SparkSession | Hoàn tất môi trường |
| Load data | Đọc file Excel/Parquet | DataFrame sẵn sàng |
| Tiền xử lý | Làm sạch, thêm feature | Dữ liệu chuẩn |
| EDA | Vẽ biểu đồ, kiểm tra tương quan | Insight rõ ràng |
| Regression (sklearn) | Train RF, XGB, LGBM, Ridge | Lưu model tốt nhất |
| Regression (Spark) | Train GBT | Model Spark ổn định |
| Anomaly Detection | Tính abnormal_score & flag | Tin rao bất thường được đánh dấu |
| Lưu output | Xuất Excel + model | Kết quả hoàn chỉnh |

---
<a id="viii-trien-khai-pipeline"></a>
## ⚙️ VIII. TRIỂN KHAI PIPELINE DỰ ĐOÁN GIÁ & PHÁT HIỆN BẤT THƯỜNG

### 🧠 Mục tiêu
Xây dựng một **pipeline tái sử dụng** cho phép **dự đoán giá xe máy cũ** và **phát hiện tin đăng bất thường** trên dữ liệu mới, mà không cần huấn luyện lại mô hình.

---

### 🧩 Thành phần chính
Pipeline được đóng gói trong hai file:

| File | Vai trò |
|------|----------|
| **`price_pipeline_module.py`** | Chứa class `PricePipeline`, bao gồm toàn bộ bước tiền xử lý – trích đặc trưng – dự đoán – phát hiện bất thường. |

---

### 🔍 Quy trình xử lý

1. **Tiền xử lý dữ liệu**  
   - Chuẩn hoá văn bản (`clean_text_vi`).  
   - Rút trích **quận/huyện** từ địa chỉ (`extract_quan`).  
   - Mã hoá biến phân loại bằng `LabelEncoder`.  

2. **Tạo đặc trưng mới**  
   - `Tuổi_xe = 2025 - Năm_đăng_ký`  
   - `Km_trên_năm = Số_Km_đã_đi / Tuổi_xe`  
   - `log_Km = log(1 + Số_Km_đã_đi)`  
   - `Phân_khúc` giá tham chiếu (`Giá_rẻ`, `Trung_bình`, `Cao_cấp`, `Sang`).  

3. **Dự đoán giá xe**  
   - Sử dụng mô hình hồi quy (`model_A_price_predictor.pkl`) để sinh cột **`Giá_dự_đoán`**.  

4. **Phát hiện bất thường (Anomaly Detection)**  
   - Tính phần dư (`Residual`, `Z_resid`) so với giá thật.  
   - Tính điểm bất thường **B_score** bằng mô hình **LOF (`model_B_lof.pkl`)**.  
   - Kết hợp thành **`abnormal_score = 0.3 * A + 0.7 * B`**.  
   - Đánh nhãn: `Giá cao bất thường`, `Giá thấp bất thường`, `Vi phạm min/max`, `Bình thường`.  

5. **Xuất kết quả**  
   - `out_full`: bản chi tiết (toàn bộ đặc trưng, điểm bất thường, lý do).  
   - `out_view`: bản rút gọn (hiển thị chính).  

---

### 🚀 Cách chạy pipeline

#### 🔹 Trong Notebook
```python
from price_pipeline_module import load_pipeline
import pandas as pd

MODEL_DIR = r"models_final_project_1_bai2"
pp = load_pipeline(MODEL_DIR)

df_new = pd.read_csv(r"new_data.csv", encoding="utf-8-sig")
out_full, out_view = pp.run(df_new, return_view_cols=True)
display(out_view) 
```

<a id="ix-ket-luan"></a>
## 🪄 IX. KẾT LUẬN

### 📊 Kết quả & Insight

#### 💡 Hiệu quả mô hình
- **Mô hình tốt nhất:** `Random Forest` → cho **độ chính xác cao nhất** trong dự đoán giá xe.  
- **Mô hình phát hiện bất thường:** `LOF (Local Outlier Factor)` → hiệu quả nhất trong việc nhận diện cả **giá lệch ngữ cảnh** và **điểm dữ liệu lạ**.

| Hạng mục | Mô tả | Nhận xét |
|-----------|--------|----------|
| 🟩 **Bình thường** | 3,835 bản ghi (≈58%) | Phần lớn dữ liệu có giá hợp lý → mô hình ổn định. |
| 🟨 **Vi phạm min/max** | 2,327 bản ghi (≈35%) | Giá vượt khung tham chiếu (cao hoặc thấp hơn mức trung bình). <br>Thường gặp ở xe cũ, xe độ, xe bán gấp. <br>Không hẳn lỗi nhưng là vùng **rủi ro cao**, cần kiểm duyệt kỹ. |
| 🟥 **Giá bất thường mạnh** | 382 bản ghi (≈6%) | Rao giá quá cao (phiên bản hiếm, nâng giá) hoặc quá thấp (nhập sai, xe hỏng, mồi giá rẻ). <br>Cần kiểm tra thủ công hoặc gắn cờ cảnh báo. |

---

### 🧩 Ứng dụng thực tế
- 💬 **Gợi ý giá tự động:** khi người bán đăng tin, hệ thống tự tính **giá hợp lý** dựa trên mô hình.  
- 🚨 **Cảnh báo kiểm duyệt:** tự động flag tin có giá bất hợp lý, hỗ trợ nhân viên duyệt tin nhanh hơn.  
- 📈 **Phân tích thị trường:** theo dõi **xu hướng giá theo thương hiệu, dòng xe, khu vực**.  
- 🧠 **Mở rộng:** có thể kết hợp thêm dữ liệu ảnh, mô tả văn bản (NLP, CV) để tăng độ chính xác.

---

### 🏁 Tổng kết
- Bộ dữ liệu xe máy Chợ Tốt được xử lý và huấn luyện qua quy trình chuẩn **Data Science Pipeline**.  
- Kết quả mô hình:
  - **R² ≈ 0.87** với `Random Forest`  
  - **Sai số MAE thấp**, dự đoán giá ổn định, ít overfit.  
- Hệ thống phát hiện bất thường hoạt động hiệu quả (**≈6% tin rao được flag**), giúp cải thiện chất lượng dữ liệu đầu vào và hỗ trợ kiểm duyệt tự động.

> ✅ **Tổng thể:** Dự án đạt mục tiêu đề ra — vừa **dự đoán chính xác giá xe máy cũ**, vừa **phát hiện được các tin rao bất thường**, sẵn sàng mở rộng sang sản phẩm thật trong tương lai.
