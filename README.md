# 🛵 DỰ ÁN DỰ ĐOÁN GIÁ & PHÁT HIỆN BẤT THƯỜNG GIÁ XE MÁY
### 🎓 Đồ án tốt nghiệp — Data Science

### **Tác giả:** Võ Thị Hoàng Anh & Nguyễn Mai Xuân Bách  
### **Giảng viên hướng dẫn:** Cô Khuất Thùy Phương  

---


# ⭐ 0. Cấu trúc thư mục
    📦 DL07_K308_VoThiHoangAnh_XuanBach
    ├── 📁 Data
    │   ├── data_motobikes.xlsx                  ← Dữ liệu thô (xe máy)
    │   └── du_lieu_xe_may_da_tien_xu_ly_1.xlsx  ← Bản chuẩn hoá & thêm feature
    │
    ├── 📄 Mô tả bộ dữ liệu Chợ Tốt.pdf           ← File mô tả dữ liệu nguồn
    │
    ├── 📁 models_final_project_1_bai2          ← File lưu mô hình của dự đoán giá và phát hiện bất thường
    ├── 📁 spark-3.5.1-bin-hadoop3 
    ├── 📁 catboost_info
    ├── 📁 images                               ← Chứa hình ảnh, biểu đồ để làm ứng dụng người dùng (streamlit)
    │           
    ├── 📘price_pipeline_module.py   
    ├── 📁 RandomForest_model_pyspark_b1_1
    ├── 📘 Project1_HoangAnh_XuanBach.ipynb    ← File xử lý dữ liệu, EDA, huấn luyện và chọn mô hình giải quyết vến đề
    ├── 📊 Project01_HoangAnh_XuanBach.pptx   ← File báo cáo, thuyết trình đồ án
    ├── 📄 app.py  (hoặc app - Copy.py)       ← File chạy ứng dụng người dùng cuối (streamlit)
    ├── 📄 requirements.txt
    ├── 📄 Procfile
    └── 📄 setup.sh

# 📚 MỤC LỤC
1. [Giới thiệu dự án](#gioi-thieu)  
2. [Kiến trúc hệ thống](#kien-truc)  
3. [Business Understanding](#business)  
4. [Data Understanding](#data)  
5. [Modeling – Dự đoán giá](#modeling)  
6. [Phát hiện bất thường](#anomaly)  
7. [Quy luật phát hiện giá bất thường ](#quyluat)  
8. [Chức năng trong Streamlit](#streamlit)  
9. [Cài đặt & chạy ứng dụng](#install)  
10. [Hệ thống đăng nhập Admin](#quanli)  
11. [Hướng dẫn sử dụng cho Quản trị viên (Admin)](#admin)  
12. [Hướng dẫn sử dụng cho người dùng](#user-guide)



<a id="gioi-thieu"></a>

# ⭐ 1. Giới thiệu dự án

Thị trường xe máy cũ tại Việt Nam biến động mạnh theo thương hiệu, dòng xe, năm đăng ký, số km đã đi và vị trí rao bán. Tuy nhiên, giá mà người bán đưa ra thường dựa trên cảm tính, dẫn đến nhiều tin đăng:

- Giá quá cao  
- Giá quá thấp  
- Không phù hợp với thị trường  

Dự án được xây dựng nhằm giải quyết bằng:

- **Dự đoán mức giá hợp lý**  
- **Phát hiện tin đăng bất thường** (giá cao/thấp, LOF bất thường, vi phạm min–max)  
- **Hỗ trợ kiểm duyệt tự động**  
- **Ứng dụng trực quan bằng Streamlit**


<a id="kien-truc"></a>

# ⭐ 2. Kiến trúc hệ thống

Người dùng → Streamlit UI → Pipeline xử lý dữ liệu → RandomForest (Dự đoán giá) + LOF (Bất thường) → Kết luận + Lý do → Gửi yêu cầu Admin → Google Sheets


### Thành phần chính
- RandomForest (dự đoán giá)  
- LOF (phát hiện khác biệt nội dung)  
- TF-IDF  
- Pipeline chuẩn hóa  
- Google Sheets API  
- Dashboard Admin  



<a id="business"></a>

# ⭐ 3. Business Understanding

### 🎯 Bối cảnh
- Giá đăng thường theo cảm tính  
- Người mua không có chuẩn tham chiếu  
- Tin giá rẻ bất thường → nguy cơ lừa đảo  

### 🎯 Vấn đề cần giải quyết
1. Làm sao **ước tính giá hợp lý**?  
2. Làm sao **phát hiện bất thường** tự động?  
3. Có thể **giải thích rõ ràng** vì sao bất thường?



<a id="data"></a>

# ⭐ 4. Data Understanding

Dữ liệu hơn 6.000 tin đăng xe máy:
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

### DATA INSIGHT:

Dữ liệu sau làm sạch phản ánh chính xác hơn thị trường xe cũ, giảm nhiễu từ các giá trị cực đoan.
Giá xe bị chi phối mạnh bởi yếu tố thương hiệu, dung tích và tình trạng xe, trong khi các yếu tố kỹ thuật khác (năm đăng ký, số km) tác động yếu.
Phân khúc cao cấp (Harley, BMW, Triumph, Touring…) có mức giá vượt trội, thể hiện rõ ranh giới thị trường giữa xe phổ thông và xe sang.
Xe tay ga và xe phân khối lớn là nhóm có biên độ giá rộng, phù hợp cho các mô hình dự đoán và phát hiện bất thường.
Vị trí địa lý ảnh hưởng gián tiếp đến giá – khu vực trung tâm (Tân Bình, Phú Nhuận, Quận 5) có giá cao hơn vùng ven (Thủ Đức, Bình Tân).
Phân tích từ khóa gợi ý khả năng phát hiện hành vi người bán:
Tin giá cao → mô tả kỹ, nhiều chi tiết tin cậy.
Tin giá thấp → mô tả ngắn, nhấn mạnh thanh lý, bán nhanh.
Các yếu tố này có thể được dùng để huấn luyện mô hình dự đoán giá hợp lý và phát hiện tin bất thường (giá ảo, nhập sai hoặc gian lận).



<a id="modeling"></a>

# ⭐ 5. Modeling – Dự đoán giá 

## PySpark Models

| Mô hình | Ưu điểm | Giải thích & Ứng dụng |
|---|---|---|
| 🌲 **Random Forest** | Ổn định, dễ song song hóa, chống overfitting | Phù hợp dữ liệu lớn, chạy tốt trên Spark cluster. |
| 🚀 **Gradient Boosting (GBTRegressor)** | Độ chính xác cao, tối ưu mạnh trong Spark MLlib | Dự đoán giá xe liên tục, giảm sai số hiệu quả. |
| 🧮 **Ridge Regression** | Nhanh, nhẹ, dễ giải thích hệ số | Baseline tuyến tính, kiểm tra ảnh hưởng từng biến. |

### Kết quả mô hình:


| **Mô hình**              | **R² (Test)** | **MAE (Test)** | **RMSE (Test)** | **Thời gian (s)** |
| ------------------------ | ------------- | -------------- | --------------- | ----------------- |
| 🌲 **Random Forest**     | 0.799     | 6,525,991  | 20,885,617  | 107.35        |
| 🧮 **Ridge Regression**  | 0.533         | 12,584,408     | 31,862,088      | 14.07             |
| 🚀 **Gradient Boosting** | 0.412         | 8,112,962      | 35,745,903      | 114.18            |

**Kết quả:**  
- R² ~ **0.80**  
- Random Forest cho kết quả tốt nhất với R² cao nhất (0.799) và sai số thấp nhất (MAE, RMSE).  
- Gradient Boosting có thời gian chạy lâu nhưng hiệu quả không tốt (R² thấp, RMSE cao)
- Model được lưu tại `GradientBoosting_model_pyspark_b1_1/`.

## ML Truyền thống (sklearn)
### Các mô hình thử nghiệm:
- Linear Regression  
- Ridge  
- LightGBM  
- XGBoost  
- CatBoost  
- RandomForest  

| Mô hình | Ưu điểm nổi bật | Giải thích & Ứng dụng |
|---|---|---|
| 🌲 **Random Forest** | Ổn định, kháng nhiễu, ít tuning | Baseline mạnh, phù hợp bài toán tabular. |
| 🚀 **Gradient Boosting** | Chính xác cao, xử lý tốt phi tuyến | Boosting tuần tự, giảm lỗi còn lại. |
| 🐱 **CatBoost** | Xử lý category tốt, không cần mã hoá | Thường outperform GBT khi có nhiều biến phân loại. |
| ⚡ **XGBoost** | Nhanh, hiệu quả, tinh chỉnh linh hoạt | Rất phổ biến trong bài toán giá. |
| 💡 **LightGBM** | Tiết kiệm RAM, huấn luyện nhanh | Phù hợp tập lớn, nhiều chiều. |
| 🧮 **Ridge Regression** | Đơn giản, dễ hiểu | Giải thích tác động từng biến đầu vào. |

### Kết quả mô hình:


| Mô hình              | R² (Test) | MAE (Test) | RMSE (Test) | Thời gian (s) |
|----------------------|-----------|------------|--------------|----------------|
| 🌲 **Random Forest**  | **0.890** | 3,687,591  | 23,934,808   | 293.2          |
| 🚀 **Gradient Boosting** | 0.875     | 3,682,696  | 27,208,377   | 243.2          |
| 🐱 **CatBoost**       | 0.863     | 4,002,870  | 29,976,868   | 396.5          |
| ⚡ **XGBoost**         | 0.858     | 3,832,443  | 30,924,787   | 95.8           |
| 💡 **LightGBM**        | 0.285     | 4,939,162  | 156,238,045  | 45.1           |
| 🧵 **Ridge Regression** | 0.260     | 7,806,546  | 161,607,611  | 2.5            |


### ✔ Vì sao chọn RandomForest?
- Có R² cao nhất, sai số (MAE, RMSE) thấp nhất
- Tốt cho dữ liệu tabular  
- Hiệu năng ổn định ở cả train và test → không overfit  
- Kết hợp tốt với TF-IDF  

<a id="anomaly"></a>

# ⭐ 6. Phát hiện bất thường (Anomaly Detection)

## Các mô hình thử nghiệm:

- LOF (Local Outlier Factor)

- Isolation Forest

- One-Class SVM

## Chỉ số đánh giá:

- AUC (weak label)

- Average Precision (weak)

- Thời gian huấn luyện

| Mô hình                   | AUC (weak) | AP (weak) | Thời gian (s) |
|---------------------------|------------|-----------|----------------|
| **LOF (Local Outlier Factor)** | **0.742**    | **0.746**   | 0.62           |
| **Isolation Forest**      | 0.713      | 0.726     | 1.45           |
| **One-Class SVM**         | 0.543      | 0.583     | 0.25           |


### ✔ Vì sao chọn LOF?

- AUC & AP cao nhất trong 3 mô hình

- Nhận diện tốt điểm bất thường cục bộ

- Phù hợp dữ liệu tabular + text (TF-IDF)

- Thời gian huấn luyện nhanh


<a id="quyluat"></a>

# ⭐ 7. Quy luật phát hiện giá bất thường 

Hệ thống sử dụng RandomForest để dự đoán giá hợp lý, sau đó so sánh với giá thực và kết hợp thêm tín hiệu từ LOF để xác định mức độ bất thường:

### 1️⃣ Dự đoán giá hợp lý

RandomForest dự đoán Giá_dự_đoán từ đặc điểm xe (hãng – dòng – loại – dung tích – năm – km – mô tả).

Đây là chuẩn tham chiếu cho bước đánh giá bất thường.

### 2️⃣ So sánh Giá thực ↔ Giá dự đoán
```bash
diff_pct = (Giá thực – Giá dự đoán) / Giá dự đoán
Z_resid  = Residual / MAD
```

Lệch mạnh hoặc Z-score vượt ngưỡng → giá cao/thấp bất thường.

### 3️⃣ Kiểm tra khoảng min/max (±15%)
Nếu giá nằm ngoài khoảng min–max → vi phạm min/max.

### 4️⃣ LOF – Kiểm tra nội dung khác biệt
LOF đánh dấu những tin có nội dung bất thường → hỗ trợ giải thích.

### 5️⃣ Kết luận cuối 
- Giá cao bất thường

- Giá thấp bất thường

- Vi phạm min/max

- Bình thường


<a id="streamlit"></a>

# ⭐ 8. Chức năng Streamlit

### **1️⃣ 📌 Dự đoán giá + Phát hiện bất thường**
- Nhập dữ liệu **từng tin** hoặc **upload file CSV/XLSX**.
- Hệ thống tự chuẩn hóa dữ liệu.
- RandomForest → dự đoán **Giá_dự_đoán**.
- LOF + Z-score + Min/Max → phát hiện **Giá cao / Giá thấp / Vi phạm min–max**.
- Hiển thị:
  - Giá dự đoán  
  - Kết luận  
  - Lý do bất thường  
  - Bảng chi tiết từng dòng  
- Cho phép **gửi yêu cầu kiểm duyệt cho Admin**.

### **2️⃣  Giới thiệu & Quy trình**
- Giải thích bài toán, dữ liệu và mục tiêu.
- Minh hoạ EDA (numeric, categorical, wordcloud).
- So sánh mô hình dự đoán giá.
- Giải thích đầy đủ **quy trình phát hiện bất thường**:
  - Dự đoán giá  
  - Tính Z-score  
  - Kiểm tra min/max  
  - Điểm LOF  
  - Ra nhãn cuối cùng  

### **3️⃣ Quản trị viên (Admin)**
- Chỉ hiện khi admin đăng nhập.
- Xem & xử lý yêu cầu người dùng:
  - pending / approved / rejected
- Xem chi tiết:
  - Dữ liệu người dùng  
  - Giá dự đoán  
  - Kết luận bất thường  
  - Lý do chi tiết  
- Duyệt hoặc từ chối từng yêu cầu (có ghi chú).
- Biểu đồ thống kê:
  - Trạng thái yêu cầu  
  - Xu hướng theo ngày  
  - Phân loại bất thường  

<a id="install"></a>

# ⭐ 9. Cài đặt & chạy ứng dụng

```bash
pip install -r requirements.txt
streamlit run app.py
```


<a id="quanli"></a>

# ⭐ 10. Hệ thống đăng nhập Admin

Tài khoản mặc định
```bash
ADMIN_USER = "admin"
ADMIN_PWD = "123456"
```

Cơ chế đăng nhập
```bash
if st.button("Đăng nhập"):
    if user == ADMIN_USER and pwd == ADMIN_PWD:
        st.session_state.admin_logged_in = True
        st.success("Đăng nhập thành công!")
        st.rerun()
```

Chức năng admin

- Xem danh sách yêu cầu

- Thống kê trạng thái

- Xem chi tiết

- Duyệt / từ chối

- Lưu Google Sheets


<a id="admin"></a>

# ⭐ 11.  Hướng dẫn sử dụng cho Quản trị viên (Admin)

Trang Quản trị viên được dùng để xem – duyệt – từ chối các yêu cầu mà người dùng gửi lên. Chỉ admin đã đăng nhập mới truy cập được.

### 1️⃣ Đăng nhập Admin

Nhập username & password ở Sidebar.

Tài khoản mặc định
```bash
ADMIN_USER = "admin"
ADMIN_PWD = "123456"
```


Khi đăng nhập thành công, hệ thống sẽ:

- Mở tab 🛠 Quản trị viên

- Lưu trạng thái vào st.session_state.admin_logged_in

### 2️⃣ Xem danh sách yêu cầu

Admin có thể xem toàn bộ yêu cầu từ Google Sheets, gồm:

- Thời gian gửi

- Dữ liệu người dùng nhập

- Giá dự đoán

- Kết luận bất thường

- Lý do chi tiết

- Trạng thái: pending / approved / rejected

- Ghi chú của admin


### 3️⃣ Xem chi tiết một yêu cầu

Nhấn vào từng dòng để xem đầy đủ:

- Thông tin người dùng

- Giá thực + Giá dự đoán

- Độ lệch %

- Tín hiệu min/max

- Nhãn kết luận cuối

- Nội dung giải thích

-> Mục này giúp admin hiểu rõ vì sao hệ thống nhận định bất thường.

### 4️⃣ Duyệt hoặc từ chối yêu cầu

Admin có thể chọn:

✔ Approve — Tin đăng hợp lệ
✔ Reject — Tin đăng không hợp lệ / bất thường rõ ràng
✔ Thêm ghi chú để phản hồi cho người dùng

Hệ thống sẽ cập nhật vào Google Sheets:
```bash
status = "approved" hoặc "rejected"
note   = ghi chú admin
time   = thời gian xử lý
```

### 5️⃣ Chức năng thống kê tổng quan

Trang admin tự động hiển thị:

- Tổng số yêu cầu

- Số pending

- Số approved

- Số rejected

- Giúp kiểm soát tiến độ xử lý.

### 6️⃣ Bảo mật

- Người không đăng nhập không nhìn thấy tab Admin.

- Tất cả thao tác duyệt đều ghi lại log vào Google Sheets.

- Mỗi admin có phiên đăng nhập riêng qua st.session_state.


<a id="user-guide"></a>

# ⭐ 12. Hướng dẫn sử dụng cho người dùng

## 🔰 Chuẩn bị
- Mở ứng dụng Streamlit  
- Chuẩn bị thông tin xe hoặc file Excel  


## 🧭 Quy trình sử dụng

### 1️⃣ — Nhập dữ liệu
- Nhập từng trường  
- Hoặc upload file Excel  

### 2️⃣ — Dự đoán giá
Hệ thống trả về:
- Giá dự đoán  
- Khoảng giá min–max  
- Z-score  
- Lệch %  

### 3️⃣ — Phát hiện bất thường
Gồm:
- Nhãn bình thường / bất thường  
- Lý do  
- LOF  
- Vi phạm min–max  

### 4️⃣ — Gửi yêu cầu kiểm duyệt
Dữ liệu được lưu lên Google Sheets.

### Bước 5️⃣ — Theo dõi trạng thái
Admin sẽ duyệt / từ chối.


## 🧪 Ví dụ minh họa


### ✅ Ví dụ 1 — Giá cao bất thường (Z-score > +3)
**Thông tin tin đăng:**
- Thương hiệu: SH Mode  
- Năm đăng ký: 2020  
- Km đã đi: 20.000  
- **Giá người đăng: 45.000.000**  
- **Giá dự đoán: 33.000.000**

**Tính toán:**
- Chênh lệch: +12.000.000  
- Tỷ lệ lệch giá: **+36%**  
- **Z_resid = +4.2 > 3**

👉 **Kết luận:** **Giá cao bất thường**  
📌 *Lý do:* Giá thực cao hơn giá dự đoán rất nhiều (Z_resid = 4.2), vượt ngưỡng an toàn.


### ✅ Ví dụ 2 — Vi phạm min/max
**Thông tin tin đăng:**
- Dòng xe: Vision 2018  
- Khoảng giá tham chiếu của nhóm:  
  - Min: 14.000.000  
  - Max: 18.000.000  
- **Giá người đăng: 12.000.000**

**Kiểm tra:**
- 12 triệu < 14 triệu × 85%  
→ Giá thấp hơn mức tối thiểu cho phép

👉 **Kết luận:** **Vi phạm min/max**  
📌 *Lý do:* Giá nằm **ngoài khoảng giá tối thiểu** của nhóm dòng xe.



### ✅ Ví dụ 3 — LOF cao (tin đăng có nội dung khác biệt)
**Thông tin tin đăng:**
- Dòng xe: Wave Alpha  
- Năm: 2017  
- Km đã đi: 5.000 (thấp bất thường)  
- Mô tả: “Xe zin, biển ngũ quý 9, hàng sưu tầm, chạy vài trăm mét”  
- Giá người đăng: 32.000.000

**Phân tích:**
- Nội dung chứa từ khóa hiếm: *“sưu tầm”, “ngũ quý 9”*  
- Pattern nội dung khác hoàn toàn các tin Wave còn lại  
- LOF đánh giá điểm bất thường **cao** → `B_flag = 1`

👉 **Kết luận:** **Khác biệt nội dung (LOF cao)**  
📌 *Lý do:* Tin đăng có **pattern bất thường**, không giống nhóm Wave thông thường, nên bị đánh dấu là outlier.


## 🛡 Bảo mật
- Người dùng không truy cập được trang Admin  
- Session riêng cho từng người  
- Chỉ admin mới có quyền duyệt yêu cầu  

