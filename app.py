import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
from scipy.stats import median_abs_deviation

# LOAD MODEL / TFIDF / ENCODER
MODEL_DIR = Path("models_final_project_1_bai2")  
model_A = joblib.load(MODEL_DIR / "model_A_price_predictor.pkl")
model_B = joblib.load(MODEL_DIR / "model_B_lof.pkl")
tfidf   = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
le_map  = joblib.load(MODEL_DIR / "label_encoders.pkl")   


# HELPER HÀM TEXT / LABEL / PHÂN KHÚC
def clean_text_vi(s: str) -> str:
    if pd.isna(s): 
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"https?://\S+|www\.\S+|\b(0|\+84)\d{8,11}\b|[\w\.-]+@[\w\.-]+\.\w+", " ", s)
    s = re.sub(r"[₫$€£¥₹#@]", " ", s)
    s = re.sub(r"(\d+)\s*(?:tr|triệu)\b", lambda m: str(int(m.group(1))*1_000_000), s)
    s = re.sub(r"(\d+)\s*cc\b", lambda m: m.group(1), s)
    s = re.sub(r"(\d+)[.,](\d+)", r"\1\2", s)
    s = "".join(ch if (ch.isalnum() or ch in " _-/") and unicodedata.category(ch)[0] not in ("C","S") else " " for ch in s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def normalize_price(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().strip()
    s = s.replace(",", "").replace(".", ".")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(tr|triệu)", s)
    if m:
        return float(m.group(1)) * 1_000_000
    m = re.fullmatch(r"(\d+)\s*tr\s*(\d+)", s)
    if m:
        tr = int(m.group(1))
        decimal_part = int(m.group(2))
        power = len(m.group(2))
        return tr * 1_000_000 + decimal_part * (10 ** (6 - power))
    if s.replace(".", "").isdigit():
        return int(float(s))

    return np.nan


def fmt_num(x):
    if pd.isna(x):
        return ""
    try:
        return f"{int(x):,}"
    except:
        return x


def extract_quan(addr: str) -> str:
    if pd.isna(addr):
        return "Khác"

    addr = str(addr)

    # 1) Đổi hết về lowercase và xóa ký tự đặc biệt
    clean = (
        addr.lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace("-", " ")
    )
    clean = re.sub(r"\s+", " ", clean).strip()

    # 2) Quận đánh số 1–12
    m = re.search(r"\b(quận|q)\s*(\d{1,2})\b", clean)
    if m:
        num = int(m.group(2))
        if 1 <= num <= 12:
            return f"Quận {num}"

    # 3) Quận/tên huyện đặc biệt TP.HCM
    special = {
        "bình thạnh": "Bình Thạnh",
        "phú nhuận": "Phú Nhuận",
        "tân bình": "Tân Bình",
        "gò vấp": "Gò Vấp",
        "thủ đức": "Thủ Đức",
        "bình tân": "Bình Tân",
        "bình chánh": "Bình Chánh",
        "nhà bè": "Nhà Bè",
        "hóc môn": "Hóc Môn",
    }

    for key, val in special.items():
        if key in clean:
            return val

    return "Khác"




def safe_label_encode(series: pd.Series, le) -> pd.Series:
    mapping = {cls: i for i, cls in enumerate(le.classes_)}
    return series.astype(str).map(lambda v: mapping.get(v, 0)).astype(int)

def phan_khuc(g):
    if pd.isna(g): 
        return "N/A"
    g = float(g)
    if g < 20_000_000:  return "Giá rẻ"
    if g < 50_000_000:  return "Trung bình"
    if g < 100_000_000: return "Cao cấp"
    return "Xe phân khối lớn / Sang"

def compute_group_medians(df):
    df = df.copy()
    df["Thương_hiệu"] = df["Thương_hiệu"].astype(str).str.lower().str.strip()
    df["Dòng_xe"]     = df["Dòng_xe"].astype(str).str.lower().str.strip()
    df["Loại_xe"]     = df["Loại_xe"].astype(str).str.lower().str.strip()

    groups = {}

    groups["blt_min"] = df.groupby(["Thương_hiệu", "Dòng_xe", "Loại_xe"])["Khoảng_giá_min"].median()
    groups["blt_max"] = df.groupby(["Thương_hiệu", "Dòng_xe", "Loại_xe"])["Khoảng_giá_max"].median()

    return groups

def get_expected_min_max(row, mg):
    key = (row["Thương_hiệu"], row["Dòng_xe"], row["Loại_xe"])

    exp_min = mg["blt_min"].get(key, np.nan)
    exp_max = mg["blt_max"].get(key, np.nan)

    return exp_min, exp_max



def check_minmax_deviation(df_orig: pd.DataFrame, median_groups: Dict[str, Any], tol: float = 0.2):
    """
    Trả về các dòng mà Khoảng_giá_min/max lệch > tol (20%) so với median nhóm.
    """
    rows = []
    for idx, row in df_orig.iterrows():
        user_min = row.get("Khoảng_giá_min", np.nan)
        user_max = row.get("Khoảng_giá_max", np.nan)
        if pd.isna(user_min) and pd.isna(user_max):
            continue  # không nhập => không cảnh báo

        exp_min, exp_max = get_expected_min_max(row, median_groups)

        warn_min = False
        warn_max = False
        diff_min = diff_max = np.nan

        if not pd.isna(user_min) and not pd.isna(exp_min) and exp_min > 0:
            diff_min = (user_min - exp_min) / exp_min
            if abs(diff_min) > tol:
                warn_min = True

        if not pd.isna(user_max) and not pd.isna(exp_max) and exp_max > 0:
            diff_max = (user_max - exp_max) / exp_max
            if abs(diff_max) > tol:
                warn_max = True

        if warn_min or warn_max:
            rows.append({
                "index": idx,
                "Thương_hiệu": row.get("Thương_hiệu", ""),
                "Dòng_xe": row.get("Dòng_xe", ""),
                "Loại_xe": row.get("Loại_xe", ""),
                "Khoảng_giá_min_nhập": user_min,
                "Khoảng_giá_min_median": exp_min,
                "Lệch_min(%)": None if pd.isna(diff_min) else round(diff_min*100, 1),
                "Khoảng_giá_max_nhập": user_max,
                "Khoảng_giá_max_median": exp_max,
                "Lệch_max(%)": None if pd.isna(diff_max) else round(diff_max*100, 1),
            })
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=[
        "index","Thương_hiệu","Dòng_xe","Loại_xe",
        "Khoảng_giá_min_nhập","Khoảng_giá_min_median","Lệch_min(%)",
        "Khoảng_giá_max_nhập","Khoảng_giá_max_median","Lệch_max(%)"
    ])

@dataclass
class PricePipeline:
    model_A: Any
    model_B: Any
    tfidf: Any
    le_map: Dict[str, Any]
    median_groups: Dict[str, Any] = None  # thêm median
    TOL: float = 0.15   # NGƯỠNG ±15%


    # cấu hình tính năng & nhãn
    drop_cols: List[str] = field(default_factory=list)  # không drop max nữa
    year_ref: int = 2025
    top_outlier_ratio: float = 0.02
    EPS_MINMAX: float = 0.10
    Z_ABS_THR: float = 3.0
    wA: float = 0.3
    wB: float = 0.7
    SCORE_THR: float = 60.0

    # tiền xử lý
    def _prep_base(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        #  CHUẨN HÓA CÁC CỘT CATEGORY 
        for col in ["Thương_hiệu", "Dòng_xe", "Loại_xe"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
            else:
                df[col] = "khác"

        #  ĐẢM BẢO MIN/MAX TỒN TẠI 
        if "Khoảng_giá_min" not in df.columns:
            df["Khoảng_giá_min"] = np.nan
        if "Khoảng_giá_max" not in df.columns:
            df["Khoảng_giá_max"] = np.nan

        #   AUTO-FILL MIN/MAX THEO MEDIAN NHÓM 
        if self.median_groups is not None:

            TOL = 0.15  # ±15%

            def fill_min(row):
                if not pd.isna(row["Khoảng_giá_min"]):
                    return row["Khoảng_giá_min"]

                exp_min, _ = get_expected_min_max(row, self.median_groups)
                return exp_min * (1 - TOL) if pd.notna(exp_min) else np.nan


            def fill_max(row):
                if not pd.isna(row["Khoảng_giá_max"]):
                    return row["Khoảng_giá_max"]

                _, exp_max = get_expected_min_max(row, self.median_groups)
                return exp_max * (1 + TOL) if pd.notna(exp_max) else np.nan



            df["Khoảng_giá_min"] = df.apply(fill_min, axis=1)
            df["Khoảng_giá_max"] = df.apply(fill_max, axis=1)

        #  QUẬN 
        if "Quận" not in df.columns:
            df["Quận"] = df.get("Địa_chỉ", "").map(extract_quan)
        else:
            df["Quận"] = df["Quận"]
            
        df["Quận"] = df["Quận"].astype(str).str.lower().str.strip()


        #  CLEAN TEXT 
        df["tieu_de_clean"] = df.get("Tiêu_đề", "").map(clean_text_vi)
        df["mo_ta_chi_tiết_clean"] = df.get("Mô_tả_chi_tiết", "").map(clean_text_vi)
        df["text_all_clean"] = (
            df["tieu_de_clean"].fillna("") + " " + df["mo_ta_chi_tiết_clean"].fillna("")
        ).str.strip()

        #  FEATURE NUMERIC 
        df["Tuổi_xe"] = (
            self.year_ref - pd.to_numeric(df.get("Năm_đăng_ký"), errors="coerce")
        ).clip(0).fillna(0)

        df["Số_Km_đã_đi"] = pd.to_numeric(df.get("Số_Km_đã_đi"), errors="coerce").fillna(0)

        df["Km_trên_năm"] = (
            df["Số_Km_đã_đi"] / df["Tuổi_xe"].replace(0, 1)
        ).replace([np.inf, -np.inf], 0)

        df["log_Km"] = np.log1p(df["Số_Km_đã_đi"])

        #  DÒNG_XE_TOP 
        if "Dòng_xe_top" in self.le_map and "Dòng_xe" in df.columns:
            known = set(self.le_map["Dòng_xe_top"].classes_.tolist())
            df["Dòng_xe_top"] = df["Dòng_xe"].astype(str).where(df["Dòng_xe"].isin(known), "khác")

        def compute_segment(row):
            if self.median_groups is None:
                return "N/A"
            
            exp_min, exp_max = get_expected_min_max(row, self.median_groups)

            # Nếu median group thiếu → dùng global
            if pd.isna(exp_min): exp_min = self.median_groups["global_min"]
            if pd.isna(exp_max): exp_max = self.median_groups["global_max"]
            base_price = (exp_min + exp_max) / 2
            return phan_khuc(base_price)

        df["Phân_khúc"] = df.apply(compute_segment, axis=1)



        #  LABEL ENCODER 
        for c, le in self.le_map.items():
            if c in df.columns:
                df[c] = safe_label_encode(df[c], le)

        return df

    # build features
    def _build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_text = self.tfidf.transform(df["text_all_clean"])
        X_text_df = pd.DataFrame.sparse.from_spmatrix(X_text, columns=self.tfidf.get_feature_names_out())

        num_cols = [c for c in ["Khoảng_giá_min","Năm_đăng_ký","Số_Km_đã_đi","Tuổi_xe","log_Km","Km_trên_năm"] if c in df]
        cat_cols = [c for c in self.le_map.keys() if c in df.columns]

        X_A = pd.concat([df[num_cols].reset_index(drop=True),
                         df[cat_cols].reset_index(drop=True),
                         X_text_df.reset_index(drop=True)], axis=1)

        fit_cols_A = getattr(self.model_A, "feature_names_in_", None)
        if fit_cols_A is not None:
            X_A = X_A.reindex(columns=list(fit_cols_A), fill_value=0)

        fit_cols_B = getattr(self.model_B, "feature_names_in_", None)
        X_B = X_A if fit_cols_B is None else X_A.reindex(columns=list(fit_cols_B), fill_value=0)
        return X_A, X_B

    # dự đoán giá
    def _predict_price(self, X_A: pd.DataFrame) -> np.ndarray:
        yhat_log = self.model_A.predict(X_A)
        return np.expm1(yhat_log)

    # điểm bất thường unsupervised
    def _lof_score(self, X_B: pd.DataFrame) -> np.ndarray:
        try:
            score_raw = -self.model_B.decision_function(X_B)
        except Exception:
            score_raw = -self.model_B.score_samples(X_B)
        ptp = np.ptp(score_raw)
        if ptp == 0:
            return np.zeros_like(score_raw, dtype=float)
        return (score_raw - score_raw.min()) / ptp * 100.0  # 0..100

    # hậu xử lý: đồng bộ logic NHÃN
    def _postprocess(self, df_base: pd.DataFrame, y_pred: np.ndarray, bscore: np.ndarray) -> pd.DataFrame:
        out = df_base.copy()
        out["Giá_dự_đoán"] = np.round(y_pred)

        if "Giá" in out.columns and out["Giá"].notna().any():
            resid = out["Giá"].to_numpy() - y_pred
            mad_sigma = median_abs_deviation(resid, scale="normal")

            pred_price = out["Giá_dự_đoán"].iat[0]

            mad_min = max(0.05 * pred_price, 2_000_000)

            if (not np.isfinite(mad_sigma)) or mad_sigma < mad_min:
                mad_sigma = mad_min


            z_resid = resid / mad_sigma

            out["Residual"] = np.round(resid)
            out["Z_resid"] = np.round(z_resid, 2)
        else:
            out["Residual"] = np.nan
            out["Z_resid"] = np.nan

        price = pd.to_numeric(out.get("Giá"), errors="coerce")

        # Lấy min/max người dùng nhập
        minv_user = pd.to_numeric(out.get("Khoảng_giá_min"), errors="coerce")
        maxv_user = pd.to_numeric(out.get("Khoảng_giá_max"), errors="coerce")

        # Median theo nhóm
        exp_min, exp_max = get_expected_min_max(out.iloc[0], self.median_groups)

        TOL = self.TOL   # ±15%

        # Nếu có giá trị người dùng nhập → dùng người dùng
        minv = minv_user.copy()
        maxv = maxv_user.copy()

        # Ngược lại → dùng median ±15%
        if minv.isna().any() or (minv == 0).any():
            if pd.notna(exp_min):
                minv = pd.Series([exp_min * (1 - TOL)])

        if maxv.isna().any() or (maxv == 0).any():
            if pd.notna(exp_max):
                maxv = pd.Series([exp_max * (1 + TOL)])

        # Kiểm tra min/max trực tiếp
        TOL = self.TOL
        violate = (
            (pd.notna(price) & pd.notna(minv) & (price < minv * (1 - TOL))) |
            (pd.notna(price) & pd.notna(maxv) & (price > maxv * (1 + TOL)))
        ).astype(int)        
        out["vi_pham_minmax"] = violate.astype(int)

        
        out["B_score"] = np.round(bscore, 2)
        n = int(len(out))
        k = max(1, int(self.top_outlier_ratio * n))
        den = n if n > 0 else 1

        if n > 1:
            thr_B = float(np.percentile(out["B_score"].to_numpy(), 100 - 100 * k / den))
        else:
            bs = out["B_score"].to_numpy()
            thr_B = float(np.nanmax(bs)) if bs.size else 0.0

        out["B_flag"] = (out["B_score"] >= thr_B).astype(int)

        # abnormal_score
        Z = pd.to_numeric(out.get("Z_resid", 0), errors="coerce").fillna(0)
        scoreA_all = np.minimum(np.abs(Z) / 3 * 100, 100) 
        scoreA = scoreA_all       
        scoreB = out["B_score"].to_numpy()   
        out["abnormal_score"] = np.round(self.wA * scoreA + self.wB * scoreB, 2)

        # Kết luận cuối
        cond_gia_cao  = (Z >= self.Z_ABS_THR)
        cond_gia_thap = (Z <= -self.Z_ABS_THR)
        cond_violate = violate.astype(bool)


        out["Kết_luận_cuối"] = np.select(
            [cond_gia_cao,            cond_gia_thap,            cond_violate],
            ["Giá cao bất thường",    "Giá thấp bất thường",    "Vi phạm min/max"],
            default="Bình thường"
        )

        # HƯỚNG BẤT THƯỜNG (đồng bộ với nhãn)
        out["Hướng_bất_thường"] = out["Kết_luận_cuối"].where(
            ~out["Kết_luận_cuối"].eq("Bình thường"),
            other="Bình thường"
        )

        # Lý do
        reasons = []
        for i in range(len(out)):
            label = out["Kết_luận_cuối"].iat[i]
            price = pd.to_numeric(out["Giá"].iat[i], errors="coerce")
            pred  = pd.to_numeric(out["Giá_dự_đoán"].iat[i], errors="coerce")
            viol = out["vi_pham_minmax"].iat[i]
            lof_flag = out["B_flag"].iat[i]

            r = []

            # 1. Chênh lệch %
            price = pd.to_numeric(price, errors="coerce")
            pred  = pd.to_numeric(pred,  errors="coerce")
            if pd.notna(price) and pd.notna(pred) and pred > 0:
                diff = price - pred
                diff_pct = diff / pred * 100

                if diff_pct >= 20:
                    r.append(f"Giá thực **cao hơn** giá dự đoán khoảng **{abs(diff_pct):.1f}%**.")
                elif diff_pct <= -20:
                    r.append(f"Giá thực **thấp hơn** giá dự đoán khoảng **{abs(diff_pct):.1f}%**.")

            # 2. Min/max
            if viol == 1:
                r.append("Giá **nằm ngoài khoảng min/max** bạn cung cấp.")

            # 3. LOF (giải thích thân thiện)
            if lof_flag == 1 and abs(diff_pct) > 15:
                r.append("Tin đăng có đặc điểm **khác biệt so với các tin còn lại**, nên được đánh dấu là bất thường.")

            # Nếu không có gì bất thường → để trống
            reasons.append("<br>• " + "<br>• ".join(r) if r else "")

        final_reasons = []
        for i in range(len(out)):
            if out["Kết_luận_cuối"].iat[i] == "Bình thường":
                final_reasons.append("")
            else:
                final_reasons.append(reasons[i])

        out["Loại_bất_thường"] = final_reasons
        # --- Tạo lý do ngắn gọn để in bảng ---
        short_reasons = []
        for i in range(len(out)):
            if out["Kết_luận_cuối"].iat[i] == "Bình thường":
                short_reasons.append("")
            else:
                r = final_reasons[i]
                r_short = []

                if "cao hơn" in r:
                    r_short.append("cao hơn dự đoán")
                if "thấp hơn" in r:
                    r_short.append("thấp hơn dự đoán")
                if "min/max" in r:
                    r_short.append("ngoài min/max")
                if "khác biệt" in r:
                    r_short.append("đặc điểm khác biệt")

                short_reasons.append(", ".join(r_short))

        out["Lý_do_ngắn_gọn"] = short_reasons

        # Nếu thiếu Giá thật, phân khúc theo Giá dự đoán
        if "Giá" not in out.columns or out["Giá"].isna().all():
            out["Phân_khúc"] = out["Giá_dự_đoán"].apply(phan_khuc)

        return out

    # chạy pipeline
    def run(self, df_input: pd.DataFrame, return_view_cols: bool = True
            ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        display_cols = ["Thương_hiệu","Dòng_xe","Loại_xe","Dung_tích_xe","Quận"]
        df_display = df_input.copy()
        if "Quận" not in df_input.columns or df_input["Quận"].isna().all() or (df_input["Quận"] == "").all():
            df_display["Quận"] = df_input["Địa_chỉ"].map(extract_quan)
        else:
            df_display["Quận"] = df_input["Quận"]


        base = self._prep_base(df_input)
        X_A, X_B = self._build_features(base)
        y_pred = self._predict_price(X_A)
        bscore = self._lof_score(X_B)
        out_full = self._postprocess(base, y_pred, bscore)

        out_view = None
        if return_view_cols:
            cols_show = [
                "id","Thương_hiệu","Dòng_xe","Loại_xe","Dung_tích_xe","Quận",
                "Khoảng_giá_min","Khoảng_giá_max",
                "Năm_đăng_ký","Tuổi_xe","Số_Km_đã_đi",
                "Giá","Giá_dự_đoán","Kết_luận_cuối",
                "Loại_bất_thường","Lý_do_ngắn_gọn",
                "B_flag","vi_pham_minmax"
            ]


        out_view = out_full[[c for c in cols_show if c in out_full.columns]].copy()
        if "id" in df_input.columns:
            out_view["id"] = df_input["id"].values
        # GẮN LẠI TÊN THẬT
        for col in ["Thương_hiệu","Dòng_xe","Loại_xe","Dung_tích_xe"]:
            if col in df_display.columns:
                out_view[col] = df_display[col].values
            
        if "Quận" in df_display.columns:
            out_view["Quận"] = df_display["Quận"].values
    
        return out_full, out_view


# TÍNH MEDIAN TỪ DATA GỐC
df_full = pd.read_excel("Data/du_lieu_xe_may_da_tien_xu_ly_1.xlsx")
median_groups = compute_group_medians(df_full)
df_full["Thương_hiệu"] = df_full["Thương_hiệu"].astype(str).str.lower().str.strip()
df_full["Dòng_xe"]     = df_full["Dòng_xe"].astype(str).str.lower().str.strip()
df_full["Loại_xe"]     = df_full["Loại_xe"].astype(str).str.lower().str.strip()


pipeline = PricePipeline(
    model_A=model_A,
    model_B=model_B,
    tfidf=tfidf,
    le_map=le_map,
    median_groups=median_groups
)


st.set_page_config(
    page_title="Dự đoán giá xe máy",
    page_icon="🛵",
    layout="wide"
)



# SIDEBAR
with st.sidebar:
    st.markdown("""
    <style>
    /* === Sidebar Styling === */

    .sb-title {
        font-size: 22px;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.35;
        margin-bottom: 10px;
    }

    .sb-block {
        margin-bottom: 20px;
    }

    .sb-header {
        font-size: 17px;
        font-weight: 600;
        color: #9CDCFE;
        margin-bottom: 6px;
    }

    .sb-list {
        list-style-type: none;
        padding-left: 12px;
        margin: 0;
        line-height: 1.45;
        color: #E0E0E0;
    }

    .sb-list li {
        margin: 2px 0;
    }

    .sb-note {
        color: #BBBBBB;
        font-size: 14px;
        font-style: italic;
        margin-left: 10px;
        margin-top: -4px;
    }
    </style>

    <div class="sb-block">
        <div class="sb-title">🎓 Đồ án tốt nghiệp<br>Data Science</div>
    </div>

    <div class="sb-block">
        <div class="sb-header">👥 Người thực hiện</div>
        <ul class="sb-list">
            <li>• <b>Võ Thị Hoàng Anh</b></li>
            <li class="sb-email">✉ anhvo.bio@gmail.com</li>
            <li>• <b>Nguyễn Mai Xuân Bách</b></li>
            <li class="sb-email">✉ 	bachxdn@gmail.com</li>
        </ul>
    </div>

    <div class="sb-block">
        <div class="sb-header">👩‍🏫 Giảng viên hướng dẫn</div>
        <ul class="sb-list">
            <li>• <b>Cô Khuất Thùy Phương</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        .nav-title {
            font-size: 20px;
            font-weight: 700;
            color: #FFFFFF;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .nav-item {
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            margin-bottom: 6px;
            color: #E0E0E0;
        }
        .nav-item:hover {
            background-color: #333;
            color: #fff;
        }
    </style>
    """, unsafe_allow_html=True)


    st.markdown("<div class='nav-title'>📑 Điều hướng</div>", unsafe_allow_html=True)

    nav = st.selectbox(
        "Chọn mục điều hướng:",
        [
            "🔰 Dataset Input",
            "📘 Business + Data Understanding",
            "📊 EDA Numeric",
            "📊 EDA Categorical",
            "🌥 WordCloud",
            "🤖 Modeling",
            "🚨 Anomaly Detection",
            "🧪 Evaluation",
            "👥 Phân công công việc" 
        ]
    )


    anchors = {
        "🔰 Dataset Input": "dataset_input",
        "📘 Business + Data Understanding": "business_understanding",
        "📊 EDA Numeric": "eda_numeric",
        "📊 EDA Categorical": "eda_categorical",
        "🌥 WordCloud": "wordcloud",
        "🤖 Modeling": "modeling",
        "🚨 Anomaly Detection": "anom_detection",
        "🧪 Evaluation": "evaluation",
        "👥 Phân công công việc": "phancong"

    }

    # Scroll đến anchor
    components.html(
        f"""
        <script>
            const el = window.parent.document.getElementById("{anchors[nav]}");
            if (el) {{
                // Cuộn tới section
                el.scrollIntoView({{ behavior: "smooth", block: "start" }});

                // Sau đó đẩy màn hình xuống thêm 80px (không dùng scrollTo)
                setTimeout(() => {{
                    const sc = window.parent.document.documentElement;
                    sc.scrollTop = sc.scrollTop -200;   
                }}, 300);
            }}
        </script>
        """,
        height=0,
    )
    with st.expander("❓ Vì sao giá có thể bị xem là bất thường?", expanded=False):
        st.markdown("""
        **Giá có thể được xem là bất thường khi rơi vào một trong các trường hợp sau:**

        **1️⃣ Mức giá chênh lệch nhiều so với mặt bằng chung**  
        Giá bạn nhập cao hoặc thấp khác thường so với những xe cùng loại, cùng đời, cùng tình trạng trên thị trường.

        **2️⃣ Không phù hợp với khoảng giá bạn đã cung cấp**  
        Nếu giá thực nằm ngoài khoảng thấp nhất – cao nhất mà bạn nhập vào (hoặc hệ thống tự ước lượng), sẽ bị báo là không khớp.

        **3️⃣ Thông tin của tin đăng khác biệt so với phần lớn các tin khác**  
        Ví dụ: mô tả, đặc điểm xe hoặc thông tin đi kèm quá khác so với các tin đăng thông thường, khiến giá trở nên thiếu hợp lý.

        👉 Chỉ cần một trong những điều trên xảy ra, giá sẽ được cảnh báo là bất thường để bạn kiểm tra lại.
        """)



# TIÊU ĐỀ TRUNG TÂM
st.markdown(
    "<h1 style='text-align:center; color:white;'>🛵 Dự đoán giá & Phát hiện bất thường giá xe máy</h1>",
    unsafe_allow_html=True
)
st.image("images/xe_may_cu.jpg", use_container_width=True)

st.write("")

# MENU NGANG
tab1, tab2 = st.tabs(
    ["📌 Dự đoán giá + Phát hiện bất thường", "ℹ️ Giới thiệu & Quy trình"]
)
# 1️⃣ TRANG DỰ ĐOÁN GIÁ
with tab1:
    st.subheader("📌 Thực hiện dự đoán giá & kiểm tra bất thường")

    # Chọn cách nhập dữ liệu
    mode = st.radio(
        "Chọn cách nhập dữ liệu:",
        ["Nhập tay từng xe", "Tải file CSV/XLSX"],
        horizontal=True
    )
    # =
    # CASE 1: NHẬP TAY
    # =
    if mode == "Nhập tay từng xe":
        with st.form("form_manual"):
            col1, col2, col3 = st.columns(3)
            with col1:
                thuong_hieu = st.text_input("Thương hiệu", "Honda")
                dong_xe = st.text_input("Dòng xe", "SH")

                loai_xe = st.selectbox(
                    "Loại xe",
                    ["Tay côn/Moto", "Tay ga", "Xe số"],
                    index=1  # default là Tay ga
                )

                dung_tich = st.selectbox(
                    "Dung tích xe",
                    ["Dưới 50 cc", "50 - 100 cc", "100 - 175 cc", "Trên 175 cc", "Không có"],
                    index=2  # default "100 - 175 cc"
                )

            with col2:
                nam = st.number_input("Năm đăng ký", min_value=1990, max_value=2025, value=2020)
                so_km = st.number_input("Số km đã đi", min_value=0, value=20000, step=1000)
                gia = st.number_input("Giá thực (VNĐ) – dùng để đánh giá bất thường", min_value=0, step=1_000_000, value=50_000_000)
            with col3:
                gia_min = st.number_input("Khoảng_giá_min (VNĐ) – có thể bỏ trống", min_value=0, step=1_000_000, value=0)
                gia_max = st.number_input("Khoảng_giá_max (VNĐ) – có thể bỏ trống", min_value=0, step=1_000_000, value=0)

            tieude = st.text_input("Tiêu đề tin đăng", "Bán SH Mode 125 chính chủ")
            mota   = st.text_area("Mô tả chi tiết", "Xe đẹp, bao test, biển số TP, giá có thương lượng.")
            diachi = st.text_input("Địa chỉ", "Quận 1, TP. Hồ Chí Minh")

            colb1, colb2 = st.columns(2)
            with colb1:
                btn_predict = st.form_submit_button("🔵 Dự đoán giá")
            with colb2:
                btn_anom = st.form_submit_button("🔴 Phát hiện bất thường")

        if btn_predict or btn_anom:
            # Chuẩn hóa dữ liệu input (0 => NaN)
            min_val = np.nan if gia_min == 0 else gia_min
            max_val = np.nan if gia_max == 0 else gia_max
            gia_val = np.nan if gia == 0 else gia

            df_input = pd.DataFrame([{
                "Thương_hiệu": thuong_hieu,
                "Dòng_xe": dong_xe,
                "Loại_xe": loai_xe,
                "Dung_tích_xe": dung_tich,
                "Năm_đăng_ký": nam,
                "Số_Km_đã_đi": so_km,
                "Giá": gia_val,
                "Khoảng_giá_min": min_val,
                "Khoảng_giá_max": max_val,
                "Tiêu_đề": tieude,
                "Mô_tả_chi_tiết": mota,
                "Địa_chỉ": diachi
            }])
            
            # --- ĐỒNG BỘ GIÁ (normalize toàn bộ) ---
            df_input["Giá"] = df_input["Giá"].apply(normalize_price)
            df_input["Khoảng_giá_min"] = df_input["Khoảng_giá_min"].apply(normalize_price)
            df_input["Khoảng_giá_max"] = df_input["Khoảng_giá_max"].apply(normalize_price)

            # --- ĐỒNG BỘ QUẬN (extract từ địa chỉ) ---
            df_input["Địa_chỉ"] = (
                df_input["Địa_chỉ"]
                .astype(str)
                .str.lower()
                .str.replace(r"[,.;:()\-_/\\]+", " ", regex=True)   # <--- QUAN TRỌNG
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
        
            
            for col in ["Thương_hiệu", "Dòng_xe", "Loại_xe", "Dung_tích_xe"]:
                if col in df_input.columns:
                    df_input[col] = df_input[col].astype(str).str.lower().str.strip()

            # Chạy pipeline
            out_full, out_view = pipeline.run(df_input)
            cols_reason = []

            if "id" in out_view.columns:
                cols_reason.append("id")
            else:
                out_view["id_temp"] = out_view.index
                cols_reason.append("id_temp")

            cols_reason += ["Kết_luận_cuối", "Loại_bất_thường"]

            df_reason = out_view[cols_reason]



            # Hiển thị kết quả dự đoán giá
            gia_du_doan_fmt = f"{int(out_view['Giá_dự_đoán'].iloc[0]):,}"
            st.markdown(f"### 🔍 Kết quả dự đoán\n**Giá dự đoán:** <span style='font-size:24px;color:#00FFAA;'>{gia_du_doan_fmt} VNĐ</span>", unsafe_allow_html=True)

            if btn_predict and not btn_anom:
                st.markdown("### 📋 Thông tin chi tiết (không chạy phát hiện bất thường)")

                show_df = out_view.copy()
                for col in ["Số_Km_đã_đi", "Giá", "Giá_dự_đoán"]:
                    if col in show_df.columns:
                        show_df[col] = show_df[col].apply(
                            lambda x: f"{int(x):,}" if pd.notna(x) else ""
                        )

                st.dataframe(show_df[[
                    c for c in ["Thương_hiệu", "Dòng_xe", "Loại_xe", "Năm_đăng_ký",
                                "Số_Km_đã_đi", "Giá", "Giá_dự_đoán"]
                    if c in show_df.columns
                ]])


            if btn_anom:
                # Cảnh báo bất thường
                st.markdown("### 🚨 Đánh giá bất thường về giá")

                row = out_view.iloc[0]
                ket_luan = row.get("Kết_luận_cuối", "Bình thường")
                reason   = row.get("Loại_bất_thường", "")
                bflag    = row.get("B_flag", 0)
                violate  = row.get("vi_pham_minmax", 0)

                if pd.isna(gia_val):
                    st.warning("⚠️ Bạn chưa nhập Giá thực, nên không thể đánh giá 'Giá cao/ thấp bất thường'.")
                else:
                    if ket_luan == "Bình thường":
                        st.success("✅ Giá này được hệ thống đánh giá là **BÌNH THƯỜNG** ")
                    else:
                        st.error(f"🚨 Kết luận: **{ket_luan}**")
                        if reason:
                            st.markdown("**Lý do:**<br>" + reason, unsafe_allow_html=True)
                st.markdown("### 📋 Bảng chi tiết")

                # format chỉ vài cột số
                show_df = out_view.copy()
                for col in ["Khoảng_giá_min", "Khoảng_giá_max", "Giá", "Giá_dự_đoán"]:
                    if col in show_df.columns:
                        show_df[col] = show_df[col].apply(
                            lambda x: f"{int(x):,}" if pd.notna(x) else ""
                        )

                # CHỈ HIỆN CÁC CỘT TRONG cols_show
                cols_show = [
                    "Thương_hiệu","Dòng_xe","Loại_xe","Dung_tích_xe","Quận",
                    "Khoảng_giá_min","Khoảng_giá_max",
                    "Năm_đăng_ký","Tuổi_xe","Số_Km_đã_đi",
                    "Giá","Giá_dự_đoán","Kết_luận_cuối"
                ]

                st.dataframe(show_df[[c for c in cols_show if c in show_df.columns]])

    # CASE 2: UPLOAD FILE
    else:
        col1, col2 = st.columns([1.5, 1])

        # Upload file
        with col1:
            file = st.file_uploader(
                "Chọn file dữ liệu:",
                type=["csv", "xlsx"],
                help="Dung lượng tối đa 200MB"
            )

            # 👉 Đặt nút NGAY DƯỚI uploader
            colb1, colb2 = st.columns(2)
            with colb1:
                btn_predict_file = st.button("🔵 Dự đoán giá cho file", use_container_width=True)
            with colb2:
                btn_anom_file = st.button("🔴 Phát hiện bất thường cho file", use_container_width=True)

        # Cột phải: danh sách cột yêu cầu
        with col2:
            st.write("### 📌 File cần có các cột:")
            st.markdown("""
            - Thương_hiệu  
            - Dòng_xe  
            - Loại_xe  
            - Dung_tích_xe  
            - Năm_đăng_ký  
            - Số_Km_đã_đi  
            - Giá *(tùy chọn)*  
            - Khoảng_giá_min  
            - Khoảng_giá_max  
            - Tiêu_đề  
            - Mô_tả_chi_tiết  
            - Địa_chỉ  
            """)
            st.warning("⚠ Thiếu cột → hệ thống sẽ báo lỗi.", icon="⚠️")

        if (btn_predict_file or btn_anom_file) and file is not None:
            # Đọc file
            if file.name.endswith(".csv"):
                df_input = pd.read_csv(file)
            else:
                df_input = pd.read_excel(file)
                
            # Gắn ID nếu chưa có
            if "id" not in df_input.columns:
                df_input["id"] = df_input.index
            else:
                df_input["id"] = df_input["id"].astype(int)

            
            for col in ["Khoảng_giá_min", "Khoảng_giá_max", "Giá"]:
                if col in df_input.columns:
                    df_input[col] = df_input[col].apply(normalize_price)

            # Km vẫn parse bình thường
            if "Số_Km_đã_đi" in df_input.columns:
                df_input["Số_Km_đã_đi"] = pd.to_numeric(
                    df_input["Số_Km_đã_đi"].astype(str).str.replace(",", "", regex=False), 
                    errors="coerce"
                )

            # 1) Chuyển về lowercase
            for col in ["Thương_hiệu", "Dòng_xe", "Loại_xe", "Dung_tích_xe"]:
                if col in df_input.columns:
                    df_input[col] = df_input[col].astype(str).str.lower().str.strip()

            # 2) Giá trị 0 coi như không nhập
            for col in ["Khoảng_giá_min", "Khoảng_giá_max", "Giá"]:
                if col in df_input.columns:
                    df_input[col] = df_input[col].replace(0, np.nan)

            # 3) Chuẩn hóa quận từ Địa chỉ (nếu có)
            if "Địa_chỉ" in df_input.columns:
                df_input["Quận"] = df_input["Địa_chỉ"].map(extract_quan)

            # Chạy pipeline
            out_full, out_view = pipeline.run(df_input)
            
            # BẢNG LÝ DO THEO ID
            df_reason = out_view.copy()

            fmt_cols = ["Giá", "Giá_dự_đoán"]
            for col in fmt_cols:
                if col in df_reason.columns:
                    df_reason[col] = df_reason[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "")

            reason_cols = [
                "id",
                "Giá",
                "Giá_dự_đoán",
                "Kết_luận_cuối",
                "Loại_bất_thường",
            ]

            df_reason_show = df_reason[reason_cols]


            # 1️⃣ HIỂN THỊ KHI NHẤN "Dự đoán giá"
            if btn_predict_file:
                st.markdown("### 🔍 Giá dự đoán (toàn bộ file)")
                cols_predict = [
                    "Thương_hiệu","Dòng_xe","Loại_xe","Năm_đăng_ký",
                    "Số_Km_đã_đi","Giá","Giá_dự_đoán"
                ]

                df_predict_show = out_view[[c for c in cols_predict if c in out_view.columns]].copy()

                for col in ["Giá", "Giá_dự_đoán"]:
                    if col in df_predict_show.columns:
                        df_predict_show[col] = df_predict_show[col].apply(
                            lambda x: f"{int(x):,}" if pd.notna(x) else ""
                        )
                st.dataframe(df_predict_show)

            # HIỂN THỊ KHI NHẤN "Phát hiện bất thường"
            if btn_anom_file:
                st.markdown("### 🚨 Phát hiện bất thường (toàn bộ file)")

                cols_anom = [
                    "Thương_hiệu","Dòng_xe","Loại_xe","Dung_tích_xe","Quận",
                    "Khoảng_giá_min","Khoảng_giá_max",
                    "Năm_đăng_ký","Tuổi_xe","Số_Km_đã_đi",
                    "Giá","Giá_dự_đoán","Kết_luận_cuối"
                ]

                df_show = out_view[[c for c in cols_anom if c in out_view.columns]]
                format_cols = ["Khoảng_giá_min", "Khoảng_giá_max", "Giá", "Giá_dự_đoán","Số_Km_đã_đi"]
                for col in format_cols:
                    if col in df_show.columns:
                        df_show[col] = df_show[col].apply(
                            lambda x: f"{int(x):,}" if pd.notna(x) else ""
                        )
                st.dataframe(df_show)

                # Các dòng bất thường
                df_abn  = out_view[out_view["Kết_luận_cuối"] != "Bình thường"].copy()
                df_norm = out_view[out_view["Kết_luận_cuối"] == "Bình thường"].copy()

                # Format số
                for col in ["Khoảng_giá_min", "Khoảng_giá_max", "Giá", "Giá_dự_đoán", "Số_Km_đã_đi"]:
                    if col in out_view.columns:
                        df_abn[col] = df_abn[col].apply(fmt_num)
                        df_norm[col] = df_norm[col].apply(fmt_num)

                # 1️⃣ BẢNG 1 — TIN BẤT THƯỜNG
                if df_abn.empty:
                    st.success("✅ Không có tin bất thường.")
                else:
                    st.error(f"🚨 Có {len(df_abn)} dòng bất thường.")
                    st.dataframe(
                        df_abn[
                            ["id","Thương_hiệu","Dòng_xe","Loại_xe","Dung_tích_xe","Quận",
                            "Giá","Giá_dự_đoán","Kết_luận_cuối","Lý_do_ngắn_gọn"]
                        ],
                        use_container_width=True
                    )

                # 2️⃣ BẢNG 2 — TIN BÌNH THƯỜNG
                st.success("✔ Các tin còn lại là BÌNH THƯỜNG")
                st.dataframe(
                    df_norm[
                        ["id","Thương_hiệu","Dòng_xe","Loại_xe","Dung_tích_xe","Quận",
                        "Giá","Giá_dự_đoán","Kết_luận_cuối"]
                    ],
                    use_container_width=True
                )

                    
# 2 TRANG GIỚI THIỆU
with tab2:

    st.markdown("""
    <style>
    .nav-item {
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        cursor: pointer;
        background-color: #222;
        color: white;
        font-size: 15px;
    }
    .nav-item:hover {
        background-color: #444;
    }
    </style>
    """, unsafe_allow_html=True)

    # ĐỌC DỮ LIỆU TỪ FILE NỘI BỘ

    st.markdown("<a id='dataset_input'></a>", unsafe_allow_html=True)
    st.markdown("## 📂 Đọc dữ liệu ban đầu (Dataset Input)")


    import pandas as pd
    try:
        df = pd.read_excel("Data/data_motobikes.xlsx")
        st.write("📌 **5 dòng đầu tiên của dữ liệu:**")
        st.dataframe(df.head(5))
    except:
        st.error("❌ Không tìm thấy file: Data/data_motobikes.xlsx")

    st.markdown("---")


    # HIỂU BÀI TOÁN (Business + Data Understanding)
    st.markdown("<a id='business_understanding'></a>", unsafe_allow_html=True)
    st.markdown("## 🧭 Hiểu bài toán (Business + Data Understanding)")


    st.markdown("""
### 🎯 Bối cảnh & vấn đề cần giải quyết
- Giá xe cũ trên thị trường (đặc biệt Chợ Tốt) biến động lớn.  
- Nhiều trường hợp giá rẻ bất thường, đắt bất thường hoặc nhập sai giá.  
- Người mua khó đánh giá mức giá hợp lý, và nền tảng cũng khó kiểm duyệt các tin đăng giá ảo hoặc giá bất thường.  

➡️ Mục tiêu:  
**Dự đoán giá hợp lý** + **phát hiện bất thường** để hỗ trợ người dùng và hệ thống kiểm duyệt.""")

    
    st.markdown("---")
    st.markdown("<a id='eda_numeric'></a>", unsafe_allow_html=True)
    st.markdown("""
### 📊 EDA: Giá ↔ Biến số (Song song 2 hình)
#### 4 biểu đồ tương quan GIÁ với biến số  
""")

    # 4 HÌNH SONG SONG (2x2)
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/num_plot_1.png", caption="Giá vs Biến số 1", width="stretch")
        st.image("images/num_plot_2.png", caption="Giá vs Biến số 2", width="stretch")
    with col2:
        st.image("images/num_plot_3.png", caption="Giá vs Biến số 3", width="stretch")
        st.image("images/num_plot_4.png", caption="Giá vs Biến số 4", width="stretch")

    st.markdown("---")

    st.markdown("<a id='eda_categorical'></a>", unsafe_allow_html=True)
    st.markdown("""
### 📊 EDA: Giá ↔ Biến phân loại (Song song 2 hình)
#### 4 biểu đồ categorical  
""")

    col3, col4 = st.columns(2)
    with col3:
        st.image("images/cat_plot_1.png", width="stretch")
        st.image("images/cat_plot_2.png" ,width="stretch")
    with col4:
        st.image("images/cat_plot_3.png", width="stretch")
        st.image("images/cat_plot_4.png", width="stretch")

    st.markdown("---")
    
    st.markdown("<a id='wordcloud'></a>", unsafe_allow_html=True)
    st.markdown("### ☁ WordCloud mô tả tin đăng")
    st.image("images/wordcloud.png", caption="WordCloud", width="stretch")

    st.markdown("""
### 🔍 Insight rút ra
- Thương hiệu & dung tích là yếu tố ảnh hưởng mạnh nhất đến giá.
- Số km & năm đăng ký ảnh hưởng yếu → không dùng trực tiếp.
- Các phân khúc cao cấp (BMW, Harley) có giá vượt trội.
- Vị trí quận/khu vực ảnh hưởng rõ rệt (trung tâm giá cao hơn).
""")

    st.markdown("---")

    st.markdown("<a id='modeling'></a>", unsafe_allow_html=True)
    # MODELING – DỰ ĐOÁN GIÁ
    st.markdown("## 🤖 Modeling – Xây dựng mô hình dự đoán giá")

    st.markdown("""
### 🔧 Giới thiệu mô hình
Thử nhiều mô hình:
- RandomForest  
- GradientBoosting  
- XGBoost  
- CatBoost  
- LightGBM  
- Ridge Regression  

### 📊 Hiệu năng mô hình
""")

    st.image("images/model_compare.png", caption="So sánh hiệu năng mô hình", width="stretch")

    st.markdown("""
### ✔ Tại sao chọn RandomForest?
- Hiệu năng cao nhất (R² ≈ 0.89).  
- Ổn định, kháng nhiễu, phù hợp dữ liệu tabular.  
- Bắt tốt quan hệ phi tuyến.  
- Không cần tune quá nhiều.  

**Nhược điểm:**
- Chậm hơn mô hình tuyến tính.
- Kích thước model lớn, khó giải thích.""")


    st.success("✔ Mô hình được chọn: **RandomForest** (R² cao nhất, ổn định nhất)")

    st.markdown("---")

    st.markdown("""
### 📈 Giá thực vs Giá dự đoán và Phân phối Residual
""")

    st.image("images/real_vs_pred.png", width="stretch")

    st.markdown("---")

    st.markdown("<a id='anom_detection'></a>", unsafe_allow_html=True)
    # PHÁT HIỆN BẤT THƯỜNG
    st.markdown("## 🚨 Phát hiện bất thường (Anomaly Detection)")

    st.markdown("""
### 🔧 Các mô hình thử nghiệm
- **LOF (Local Outlier Factor)**
- Isolation Forest
- One-Class SVM

Model được đánh giá dựa trên:
- AUC (weak label)
- Average Precision (weak)
- Thời gian huấn luyện  

👉 Kết quả so sánh 3 mô hình như bảng dưới đây:
""")

    # BẢNG SO SÁNH 3 MÔ HÌNH (bạn tự thay bằng bảng thật của bạn)
    import pandas as pd

    df_anom_model = pd.DataFrame({
        "Model": ["LOF", "IsolationForest", "OneClassSVM"],
        "AUC(weak)": [0.741525, 0.712916, 0.542578],
        "AP(weak)": [0.746060, 0.726143, 0.583293],
        "Time(s)": [0.616718, 1.449044, 0.247992]
    })

    st.dataframe(df_anom_model)
    st.success("✔ Mô hình được chọn: **LOF** (hiệu năng tốt nhất)")

    st.markdown("---")

    st.markdown("### 📄 Bảng kết quả bất thường")

    df_anom_example = pd.DataFrame({
        "id": [3640, 1456, 3549, 2522, 4304],
        "Giá": [49000000, 49000000, 46000000, 17000000, 19000000],
        "Khoảng_giá_min": [6020000, 6240000, 8920000, 5290000, 31230000],
        "Khoảng_giá_max": [7060000, 7320000, 10470000, 6210000, 36660000],
        "Giá_dự_đoán": [3.6883e7, 3.6702e7, 3.5532e7, 6.9374e6, 9.4079e6],
        "Residual": [1.211e7, 1.229e7, 1.046e7, 1.006e7, 9.592e6],
        "Hướng_bất_thường": ["Giá cao"]*5,
        "Kết_luận_cuối": ["Giá cao bất thường"]*5
    })

    st.dataframe(df_anom_example)

    st.markdown("---")
    
    st.markdown("<a id='evaluation'></a>", unsafe_allow_html=True)
    st.markdown("## 🧪 Đánh giá mô hình")

    col_left, col_right = st.columns([1, 1])   

    with col_right:
        st.markdown("### 📝 Nhận xét")
        st.markdown("""
        **Kết quả:**

        - **Bình thường: chiếm đa số** → Phần lớn dữ liệu có mức giá hợp lý, cho thấy hệ thống đánh giá hoạt động ổn định.  
        - **Vi phạm min/max: nhóm lớn thứ hai** → Giá rao nằm ngoài khoảng giá tham chiếu (cao hơn hoặc thấp hơn khung hợp lý).  
        Nhóm này không hẳn sai, nhưng là vùng rủi ro cần được xem xét kỹ khi kiểm duyệt (xe độ, xe hiếm, xe bán gấp…).  
        - **Giá bất thường: chiếm tỷ lệ nhỏ** → Những tin đăng có mức giá cao hoặc thấp khác thường, thường liên quan tới nâng giá, nhập sai, hoặc mô tả bất thường.

        **Ứng dụng:**

        - Gợi ý mức giá hợp lý cho người bán.  
        - Cảnh báo kiểm duyệt khi giá vượt ngưỡng bất hợp lý.  
        - Hỗ trợ phân tích xu hướng thị trường theo khu vực và dòng xe.
        """)

    with col_left:
        st.markdown("### 📊 Biểu đồ đánh giá")
        st.image("images/eval_chart.png", caption="Phân bố nhóm bất thường", width="stretch")
        
    
    # PHÂN CÔNG CÔNG VIỆC
    st.markdown("<a id='phancong'></a>", unsafe_allow_html=True)
    st.markdown("## 👥 Phân công công việc")

    st.markdown("""
    | Thành viên | Nhiệm vụ |
    |-----------|-----------|
    | 👩‍💼 Võ Thị Hoàng Anh | Xây dựng mô hình dự đoán giá<br>Soạn thuyết trình |
    | 👨‍💻 Nguyễn Mai Xuân Bách | Khám phá & xử lý dữ liệu<br>Phát hiện bất thường<br>Kiểm tra code |
    | 🤝 Cả hai | Viết báo cáo<br>So sánh kết quả<br>Chuẩn bị slide |
    """, unsafe_allow_html=True)
