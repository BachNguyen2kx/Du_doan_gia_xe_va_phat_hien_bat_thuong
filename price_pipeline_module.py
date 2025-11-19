# price_pipeline_module.py
# -*- coding: utf-8 -*-
"""
Reusable pipeline for applying trained models to new datasets.
- Provides text cleaning, feature building, price prediction, anomaly scoring, and post-processing.
- Includes helper functions to load models and run on CSV files.
- Can also be used as a CLI: see bottom "if __name__ == '__main__':" block or use run_price_pipeline.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import re
import unicodedata   # ✅ Đúng
import joblib
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation


# ---------- Helpers ----------
def clean_text_vi(s: str) -> str:
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    # remove urls, phones, emails
    s = re.sub(r"https?://\S+|www\.\S+|\b(0|\+84)\d{8,11}\b|[\w\.-]+@[\w\.-]+\.\w+", " ", s)
    # remove currency/special
    s = re.sub(r"[₫$€£¥₹#@]", " ", s)
    # normalize 'triệu' -> number of VND
    s = re.sub(r"(\d+)\s*(?:tr|triệu)\b", lambda m: str(int(m.group(1)) * 1_000_000), s)
    # 'cc' capacity
    s = re.sub(r"(\d+)\s*cc\b", lambda m: m.group(1), s)
    # 1.2 -> 12; 1,2 -> 12
    s = re.sub(r"(\d+)[.,](\d+)", r"\1\2", s)
    # keep alnum and a few symbols; drop control/symbol categories
    s = "".join(
        ch if (ch.isalnum() or ch in " _-/") and unicodedata.category(ch)[0] not in ("C", "S") else " "
        for ch in s
    )
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def extract_quan(addr: str) -> str:
    if pd.isna(addr):
        return "Khác"
    addr = str(addr)
    m = re.search(
        r"(Quận\s*\d+|Q\d+|Bình Thạnh|Phú Nhuận|Tân Bình|Gò Vấp|Thủ Đức|Bình Tân|Bình Chánh|Nhà Bè|Hóc Môn)",
        addr,
        flags=re.I,
    )
    if not m:
        return "Khác"
    q = m.group(1).title().replace("Q", "Quận ")
    q = re.sub(r"Quận\s+(\d+)\s*", r"Quận \1", q)
    return q


def safe_label_encode(series: pd.Series, le) -> pd.Series:
    mapping = {cls: i for i, cls in enumerate(le.classes_)}
    return series.astype(str).map(lambda v: mapping.get(v, 0)).astype(int)


def phan_khuc(g):
    if pd.isna(g):
        return "N/A"
    g = float(g)
    if g < 20_000_000:
        return "Giá rẻ"
    if g < 50_000_000:
        return "Trung bình"
    if g < 100_000_000:
        return "Cao cấp"
    return "Xe phân khối lớn / Sang"


# ---------- Core Pipeline ----------
@dataclass
class PricePipeline:
    model_A: Any
    model_B: Any
    tfidf: Any
    le_map: Dict[str, Any]

    # cấu hình tính năng & nhãn
    drop_cols: List[str] = field(default_factory=lambda: ["Khoảng_giá_max"])
    year_ref: int = 2025
    top_outlier_ratio: float = 0.02
    EPS_MINMAX: float = 0.10
    Z_ABS_THR: float = 2.0
    wA: float = 0.3
    wB: float = 0.7
    SCORE_THR: float = 60.0

    # tiền xử lý
    def _prep_base(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1) drop cột không dùng khi build features (không ảnh hưởng min/max)
        df = df.drop(columns=self.drop_cols, errors="ignore")

        # 2) Quận
        if "Quận" not in df.columns:
            df["Quận"] = df.get("Địa_chỉ", pd.Series([""] * len(df))).map(extract_quan)

        # 3) clean text
        df["tieu_de_clean"] = df.get("Tiêu_đề", "").map(clean_text_vi)
        df["mo_ta_chi_tiết_clean"] = df.get("Mô_tả_chi_tiết", "").map(clean_text_vi)
        df["text_all_clean"] = (df["tieu_de_clean"].fillna("") + " " + df["mo_ta_chi_tiết_clean"].fillna("")).str.strip()

        # 4) numeric features
        df["Tuổi_xe"] = (self.year_ref - pd.to_numeric(df.get("Năm_đăng_ký"), errors="coerce")).clip(lower=0).fillna(0)
        df["Số_Km_đã_đi"] = pd.to_numeric(df.get("Số_Km_đã_đi"), errors="coerce").fillna(0)
        df["Km_trên_năm"] = (df["Số_Km_đã_đi"] / df["Tuổi_xe"].replace(0, 1)).replace([np.inf, -np.inf], 0)
        df["log_Km"] = np.log1p(df["Số_Km_đã_đi"])

        # 5) Dòng_xe_top nếu có le_map
        if "Dòng_xe_top" in self.le_map:
            known = set(self.le_map["Dòng_xe_top"].classes_.tolist())
            df["Dòng_xe_top"] = df["Dòng_xe"].astype(str).where(df["Dòng_xe"].astype(str).isin(known), "Khác")

        # 6) Phân_khúc tham chiếu
        base_for_segment = df["Giá"] if "Giá" in df.columns else df.get("Khoảng_giá_min", pd.Series([np.nan] * len(df)))
        df["Phân_khúc"] = base_for_segment.apply(phan_khuc)

        # 7) label encode
        for c, le in self.le_map.items():
            if c in df.columns:
                df[c] = safe_label_encode(df[c], le)

        return df

    # build features
    def _build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_text = self.tfidf.transform(df["text_all_clean"])
        X_text_df = pd.DataFrame.sparse.from_spmatrix(X_text, columns=self.tfidf.get_feature_names_out())

        num_cols = [c for c in ["Khoảng_giá_min", "Năm_đăng_ký", "Số_Km_đã_đi", "Tuổi_xe", "log_Km", "Km_trên_năm"] if c in df]
        cat_cols = [c for c in self.le_map.keys() if c in df.columns]

        X_A = pd.concat(
            [df[num_cols].reset_index(drop=True), df[cat_cols].reset_index(drop=True), X_text_df.reset_index(drop=True)],
            axis=1,
        )

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

        # Residual + Z_resid (robust theo MAD)
        if "Giá" in out.columns:
            resid = out["Giá"].to_numpy() - y_pred
            mad_sigma = median_abs_deviation(resid, scale="normal")
            mad_sigma = float(mad_sigma if np.isfinite(mad_sigma) and mad_sigma > 0 else 1e-6)
            z_resid = resid / mad_sigma
            out["Residual"] = np.round(resid)
            out["Z_resid"] = np.round(z_resid, 2)
        else:
            out["Residual"] = np.nan
            out["Z_resid"] = np.nan

        # vi_pham_minmax
        price = pd.to_numeric(out.get("Giá"), errors="coerce")
        minv = pd.to_numeric(out.get("Khoảng_giá_min"), errors="coerce")
        maxv = pd.to_numeric(out.get("Khoảng_giá_max"), errors="coerce")

        has_min = "Khoảng_giá_min" in out.columns
        has_max = "Khoảng_giá_max" in out.columns

        if has_min and has_max:
            min_rule = minv * (1 - self.EPS_MINMAX)
            max_rule = maxv * (1 + self.EPS_MINMAX)
            violate = (price < min_rule) | (price > max_rule)
        elif has_min:
            min_rule = minv * (1 - self.EPS_MINMAX)
            violate = price < min_rule
        elif has_max:
            max_rule = maxv * (1 + self.EPS_MINMAX)
            violate = price > max_rule
        else:
            violate = pd.Series(False, index=out.index)

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
        scoreA_all = np.minimum(np.abs(Z) / 3 * 100, 100)  # 0..100
        scoreA = np.where(violate, scoreA_all, 0.0)
        scoreB = np.where(violate, out["B_score"].to_numpy(), 0.0)
        out["abnormal_score"] = np.round(self.wA * scoreA + self.wB * scoreB, 2)

        # Kết luận cuối
        cond_gia_cao = violate & (Z >= self.Z_ABS_THR)
        cond_gia_thap = violate & (Z <= -self.Z_ABS_THR)
        cond_violate = violate  # phần còn lại (vi phạm nhưng |Z| < ngưỡng)

        out["Kết_luận_cuối"] = np.select(
            [cond_gia_cao, cond_gia_thap, cond_violate],
            ["Giá cao bất thường", "Giá thấp bất thường", "Vi phạm min/max"],
            default="Bình thường",
        )

        # HƯỚNG BẤT THƯỜNG (đồng bộ với nhãn)
        out["Hướng_bất_thường"] = out["Kết_luận_cuối"].where(~out["Kết_luận_cuối"].eq("Bình thường"), other="Bình thường")

        # Lý do
        reasons = []
        for i in range(len(out)):
            label = out["Kết_luận_cuối"].iat[i]
            if label == "Bình thường":
                reasons.append("")
                continue

            r = []
            if label == "Giá cao bất thường":
                r.append(f"Giá cao (Z≥{self.Z_ABS_THR:g})")
                r.append("Vượt ngoài khoảng min/max")
            elif label == "Giá thấp bất thường":
                r.append(f"Giá thấp (Z≤-{self.Z_ABS_THR:g})")
                r.append("Vượt ngoài khoảng min/max")
            elif label == "Vi phạm min/max":
                r.append("Vượt ngoài khoảng min/max")

            # Mức độ nghi ngờ (mô tả, không đổi nhãn)
            if out["abnormal_score"].iat[i] >= self.SCORE_THR:
                r.append(f"Điểm bất thường {out['abnormal_score'].iat[i]:.0f}≥{self.SCORE_THR:.0f}")

            if out["B_flag"].iat[i] == 1:
                r.append("B_flag=1")

            reasons.append("; ".join(r))

        out["Loại_bất_thường"] = reasons

        # Nếu thiếu Giá thật, phân khúc theo Giá dự đoán
        if "Giá" not in out.columns:
            out["Phân_khúc"] = out["Giá_dự_đoán"].apply(phan_khuc)

        return out

    # chạy pipeline
    def run(self, df_input: pd.DataFrame, return_view_cols: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Trả về (out_full, out_view) — out_view là bảng gọn để hiển thị.
        """
        base = self._prep_base(df_input)
        X_A, X_B = self._build_features(base)
        y_pred = self._predict_price(X_A)
        bscore = self._lof_score(X_B)
        out_full = self._postprocess(base, y_pred, bscore)

        out_view = None
        if return_view_cols:
            cols_show = [
                "Thương_hiệu",
                "Dòng_xe",
                "Dòng_xe_top",
                "Loại_xe",
                "Dung_tích_xe",
                "Quận",
                "Khoảng_giá_min",
                "Khoảng_giá_max",
                "Năm_đăng_ký",
                "Tuổi_xe",
                "Số_Km_đã_đi",
                "Km_trên_năm",
                "log_Km",
                "Giá",
                "Giá_dự_đoán",
                "Residual",
                "Z_resid",
                "vi_pham_minmax",
                "abnormal_score",
                "Hướng_bất_thường",
                "Loại_bất_thường",
                "B_score",
                "B_flag",
                "Kết_luận_cuối",
                "Phân_khúc",
            ]
            out_view = out_full[[c for c in cols_show if c in out_full.columns]]
        return out_full, out_view


# ---------- Convenience loaders / runners ----------
def load_pipeline(model_dir: str | Path) -> PricePipeline:
    """
    Load models/vectorizers/encoders from a directory.
    Expected filenames:
    - model_A_price_predictor.pkl
    - model_B_lof.pkl
    - tfidf_vectorizer.pkl
    - label_encoders.pkl  (dict: {col_name: LabelEncoder})
    """
    model_dir = Path(model_dir)
    model_A = joblib.load(model_dir / "model_A_price_predictor.pkl")
    model_B = joblib.load(model_dir / "model_B_lof.pkl")
    tfidf = joblib.load(model_dir / "tfidf_vectorizer.pkl")
    le_map = joblib.load(model_dir / "label_encoders.pkl")
    return PricePipeline(model_A=model_A, model_B=model_B, tfidf=tfidf, le_map=le_map)


def apply_pipeline_to_csv(
    model_dir: str | Path,
    input_csv: str | Path,
    out_full_csv: str | Path = "out_full.csv",
    out_view_csv: Optional[str | Path] = "out_view.csv",
    return_view_cols: bool = True,
    encoding: str = "utf-8",
) -> Tuple[str, Optional[str]]:
    """
    Load pipeline, read CSV, run, and save outputs to CSV.
    Returns tuple of paths (out_full_csv, out_view_csv or None).
    """
    pp = load_pipeline(model_dir)
    df = pd.read_csv(input_csv, encoding=encoding)
    out_full, out_view = pp.run(df, return_view_cols=return_view_cols)
    pd.DataFrame(out_full).to_csv(out_full_csv, index=False, encoding=encoding)
    if return_view_cols and out_view is not None:
        pd.DataFrame(out_view).to_csv(out_view_csv, index=False, encoding=encoding)
        return str(out_full_csv), str(out_view_csv)
    return str(out_full_csv), None


# ---------- Optional CLI ----------
def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(description="Apply PricePipeline to a CSV file.")
    p.add_argument("--models", required=True, help="Path to directory containing trained artifacts (.pkl).")
    p.add_argument("--input", required=True, help="Input CSV path.")
    p.add_argument("--out-full", default="out_full.csv", help="Output CSV path for full results.")
    p.add_argument("--out-view", default="out_view.csv", help="Output CSV path for compact view.")
    p.add_argument("--no-view", action="store_true", help="Do not produce the compact view CSV.")
    p.add_argument("--encoding", default="utf-8", help="CSV encoding (default: utf-8).")
    return p


def main_cli():
    args = _build_arg_parser().parse_args()
    out_full, out_view = apply_pipeline_to_csv(
        model_dir=args.models,
        input_csv=args.input,
        out_full_csv=args.out_full,
        out_view_csv=None if args.no_view else args.out_view,
        return_view_cols=not args.no_view,
        encoding=args.encoding,
    )
    print(f"Saved full results to: {out_full}")
    if out_view:
        print(f"Saved compact view to: {out_view}")


if __name__ == "__main__":
    main_cli()
