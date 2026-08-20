"""Inference and input preparation for the WP collectibility V2 model.

The trained model expects one row per taxpayer (WP). This module supports both
pre-aggregated WP features and raw kohir-level data. Target/outcome columns are
never used during inference feature preparation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

TRAINING_SKLEARN_VERSION = "1.9.0"

LABEL_NAMES = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
LABEL_DESCRIPTIONS = {
    0: "Indikasi ketertagihan rendah — verifikasi kapasitas, sengketa, dan aset.",
    1: "Indikasi ketertagihan sedang — perlukan tindakan lanjutan terarah.",
    2: "Indikasi ketertagihan tinggi — kandidat quick win/persuasif.",
}

NUMERIC_FEATURES = [
    "TOTAL_TUNGGAKAN_POKOK",
    "LOG_TOTAL_TUNGGAKAN",
    "RATA_TUNGGAKAN",
    "MEDIAN_TUNGGAKAN",
    "MAX_TUNGGAKAN",
    "LOG_MAX_TUNGGAKAN",
    "JML_KETETAPAN",
    "JML_JENIS_PAJAK",
    "JML_JENIS_KETETAPAN",
    "UMUR_UTANG_RATA",
    "UMUR_UTANG_TERTUA",
    "SISA_DALUWARSA_TERDEKAT",
    "RATA_SELISIH_TAHUN_TERBIT",
    "RASIO_LAPOR_SPT_3THN",
    "FLAG_LAPOR_SPT_TERAKHIR",
    "FLAG_TANPA_DATA_SPT",
    "PEREDARAN_BRUTO",
    "LOG_PEREDARAN_BRUTO",
    "FLAG_PEREDARAN_NOL_KOSONG",
    "RASIO_TUNGGAKAN_PEREDARAN",
    "JML_CUSTOMER",
    "DPP_CUSTOMER",
    "LOG_DPP_CUSTOMER",
    "FAKTUR_CUSTOMER",
    "JML_SUPPLIER",
    "DPP_SUPPLIER",
    "LOG_DPP_SUPPLIER",
    "FAKTUR_SUPPLIER",
    "TOTAL_MITRA",
]

CATEGORICAL_FEATURES = [
    "STS_WP",
    "JENIS_WP",
    "JENIS_KPP_BKM",
    "KD_KANWIL",
    "SEKTOR_KLU",
    "JENIS_PAJAK_DOMINAN",
    "KODE_KETETAPAN_DOMINAN",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
IDENTIFIER_COLUMNS = ["NPWP16", "NAMA_WP"]

RAW_REQUIRED_COLUMNS = [
    "NPWP16",
    "NO_STPSKP",
    "JENIS_PAJAK",
    "NILAI_STPSKP",
    "TGL_UTANG_DPT_DITAGIH",
]
RAW_RECOMMENDED_COLUMNS = [
    "NPWP16",
    "NAMA_WP",
    "STS_WP",
    "JENIS_WP",
    "JENIS_KPP_BKM",
    "KD_KANWIL",
    "KD_KLU",
    "NO_STPSKP",
    "JENIS_PAJAK",
    "NILAI_STPSKP",
    "TGL_PRODUK_HUKUM",
    "TGL_UTANG_DPT_DITAGIH",
    "TGL_DALUWARSA",
    "TH_PJK",
]
# Only these canonical predictor/source fields may determine whether two raw
# rows represent the same kohir observation. Outcome fields are excluded by
# construction, so changing LABEL/SETOR_* cannot change aggregate features.
RAW_CANONICAL_COLUMNS = RAW_RECOMMENDED_COLUMNS.copy()

# Explicit guardrail: these source columns form or closely proxy the target.
FORBIDDEN_OUTCOME_COLUMNS = {
    "LABEL",
    "NILAI_SISA",
    "NILAI_CAIR_HISTORIS",
    "SETOR_SEBELUM_COLL_DATE",
    "SETOR_SEBELUM_TEGURAN",
    "SETOR_TEGURAN",
    "SETOR_PAKSA",
    "SETOR_SITA",
    "SETOR_CEGAH",
    "SETOR_SPRINDRA",
    "JML_SURAT_TEGURAN",
    "JML_SURAT_PAKSA",
    "FLAG_PERNAH_DISITA",
    "FLAG_PERNAH_BLOKIR",
    "FLAG_RESPON_PENAGIHAN",
    "TGL_TEGURAN",
    "TGL_PENYAMPAIAN_SP",
    "TGL_BAPS",
}

KPP_MAP = {"P": "PRATAMA", "M": "MADYA", "B": "BESAR", "K": "KHUSUS"}
SECTOR_MAP = {
    "0": "PERTANIAN",
    "1": "PERTAMBANGAN",
    "2": "INDUSTRI",
    "3": "ENERGI/AIR",
    "4": "KONSTRUKSI",
    "5": "PERDAGANGAN",
    "6": "TRANSPORTASI",
    "7": "INFORMASI",
    "8": "JASA",
    "9": "JASA_LAIN",
}


class InputValidationError(ValueError):
    """Raised when uploaded inference data cannot be prepared safely."""


@dataclass
class PreparationReport:
    input_rows: int
    output_wp: int
    duplicate_rows_removed: int = 0
    snapshot_date: str | None = None
    missing_model_features: list[str] | None = None
    ignored_outcome_columns: list[str] | None = None
    warnings: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_model(path: str | Path):
    """Load a trusted local joblib model and validate its feature contract."""
    if sklearn.__version__ != TRAINING_SKLEARN_VERSION:
        raise RuntimeError(
            "Versi scikit-learn tidak kompatibel dengan artefak model: "
            f"runtime={sklearn.__version__}, training={TRAINING_SKLEARN_VERSION}. "
            "Instal dependensi dari requirements.txt."
        )
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    model = joblib.load(model_path)
    actual = list(getattr(model, "feature_names_in_", []))
    if actual and actual != FEATURE_COLUMNS:
        raise RuntimeError(
            "Kontrak fitur model berbeda dari aplikasi. "
            f"Model={len(actual)} fitur, aplikasi={len(FEATURE_COLUMNS)} fitur."
        )
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Model tidak menyediakan predict_proba().")
    return model


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().upper() for c in out.columns]
    if out.columns.duplicated().any():
        dupes = out.columns[out.columns.duplicated()].tolist()
        raise InputValidationError(f"Nama kolom duplikat setelah normalisasi: {dupes}")
    return out


def _nonempty_mask(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype("string").str.strip().ne("")


def normalize_npwp(series: pd.Series, *, strict: bool = False) -> pd.Series:
    """Normalize NPWP to nullable 16-digit strings.

    Only digits and common NPWP separators (dot, hyphen, whitespace) are
    accepted. Alphabetic/scientific notation is rejected rather than silently
    stripped into a different identifier.
    """
    raw = series.astype("string").str.strip()
    nonempty = raw.notna() & raw.ne("")
    decimal_suffix = raw.str.match(r"^\d+\.0$", na=False)
    allowed = raw.str.match(r"^[0-9.\-\s]+$", na=False) & ~raw.str.contains(r"[eE]", na=False)
    canonical = raw.where(~decimal_suffix, raw.str.replace(r"\.0$", "", regex=True))
    canonical = canonical.str.replace(r"[.\-\s]", "", regex=True)
    valid = nonempty & allowed & canonical.str.match(r"^\d{1,16}$", na=False)
    invalid = nonempty & ~valid
    if strict and invalid.any():
        raise InputValidationError(
            f"{int(invalid.sum())} NPWP memiliki format tidak valid; gunakan maksimal 16 digit "
            "dengan pemisah titik/hyphen opsional, bukan notasi ilmiah atau huruf."
        )
    canonical = canonical.where(valid)
    return canonical.str.zfill(16)


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("float64")

    raw = series.astype("string").str.strip()
    direct = pd.to_numeric(raw, errors="coerce")
    cleaned = raw.str.replace(r"(?i)^rp\s*", "", regex=True).str.replace(" ", "", regex=False)
    valid_syntax = cleaned.str.match(r"^-?\d[\d.,]*$", na=False)
    cleaned = cleaned.where(valid_syntax)

    # US/international thousands style: 1,000,000.50
    comma_thousands = cleaned.str.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$", na=False)
    cleaned = cleaned.where(~comma_thousands, cleaned.str.replace(",", "", regex=False))
    # Indonesian thousands style: 1.000.000,50
    dot_thousands = cleaned.str.match(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$", na=False)
    indo = cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    cleaned = cleaned.where(~dot_thousands, indo)
    # Decimal comma without thousands separator: 1234,50
    comma_decimal = cleaned.str.contains(",", na=False) & ~cleaned.str.contains(r"\.", na=False)
    cleaned = cleaned.where(~comma_decimal, cleaned.str.replace(",", ".", regex=False))

    fallback = pd.to_numeric(cleaned, errors="coerce")
    return direct.fillna(fallback).astype("float64")


def _coerce_numeric_strict(series: pd.Series, column: str) -> pd.Series:
    parsed = _to_numeric(series)
    invalid = _nonempty_mask(series) & (parsed.isna() | ~np.isfinite(parsed))
    if invalid.any():
        raise InputValidationError(
            f"Kolom {column} memiliki {int(invalid.sum())} nilai numerik tidak valid/non-finite. "
            "Gunakan angka mesin atau format ribuan konsisten."
        )
    return parsed


def _coerce_date_strict(series: pd.Series, column: str) -> pd.Series:
    raw = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    iso = raw.str.match(r"^\d{4}-\d{2}-\d{2}(?:[ T].*)?$", na=False)
    dmy = raw.str.match(r"^\d{2}/\d{2}/\d{4}(?:[ T].*)?$", na=False)
    parsed.loc[iso] = pd.to_datetime(raw.loc[iso], errors="coerce", format="mixed")
    parsed.loc[dmy] = pd.to_datetime(raw.loc[dmy], errors="coerce", dayfirst=True, format="mixed")
    invalid = _nonempty_mask(series) & parsed.isna()
    if invalid.any():
        raise InputValidationError(
            f"Kolom {column} memiliki {int(invalid.sum())} tanggal tidak valid. "
            "Gunakan YYYY-MM-DD atau DD/MM/YYYY."
        )
    return parsed


def _as_category(series: pd.Series, column: str) -> pd.Series:
    s = series.astype("string").str.strip().replace("", pd.NA)
    if column == "KD_KANWIL":
        s = s.str.replace(r"\.0$", "", regex=True).str.zfill(3)
    elif column == "JENIS_PAJAK_DOMINAN":
        s = s.str.replace(r"\.0$", "", regex=True).str.zfill(6)
    elif column == "KODE_KETETAPAN_DOMINAN":
        s = s.str.replace(r"\.0$", "", regex=True).str.zfill(3)
    return s.astype(object).where(s.notna(), np.nan)


def _mode_safe(series: pd.Series):
    mode = series.dropna().mode()
    return mode.iloc[0] if not mode.empty else np.nan


def _fill_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    def fill(column: str, values: pd.Series) -> None:
        out[column] = out[column].where(out[column].notna(), values)

    safe_count = out["JML_KETETAPAN"].replace(0, np.nan)
    fill("RATA_TUNGGAKAN", out["TOTAL_TUNGGAKAN_POKOK"] / safe_count)
    fill("LOG_TOTAL_TUNGGAKAN", np.log1p(out["TOTAL_TUNGGAKAN_POKOK"].clip(lower=0)))
    fill("LOG_MAX_TUNGGAKAN", np.log1p(out["MAX_TUNGGAKAN"].clip(lower=0)))
    fill("LOG_PEREDARAN_BRUTO", np.log1p(out["PEREDARAN_BRUTO"].clip(lower=0)))
    fill("LOG_DPP_CUSTOMER", np.log1p(out["DPP_CUSTOMER"].clip(lower=0)))
    fill("LOG_DPP_SUPPLIER", np.log1p(out["DPP_SUPPLIER"].clip(lower=0)))

    turnover_valid = out["PEREDARAN_BRUTO"] > 0
    ratio = (out["TOTAL_TUNGGAKAN_POKOK"] / out["PEREDARAN_BRUTO"]).where(turnover_valid)
    fill("RASIO_TUNGGAKAN_PEREDARAN", ratio)
    fill("FLAG_TANPA_DATA_SPT", out["RASIO_LAPOR_SPT_3THN"].isna().astype(float))
    fill(
        "FLAG_PEREDARAN_NOL_KOSONG",
        (out["PEREDARAN_BRUTO"].isna() | (out["PEREDARAN_BRUTO"] <= 0)).astype(float),
    )
    total_mitra = out[["JML_CUSTOMER", "JML_SUPPLIER"]].sum(axis=1, min_count=1)
    fill("TOTAL_MITRA", total_mitra)
    return out


def prepare_feature_input(frame: pd.DataFrame) -> tuple[pd.DataFrame, PreparationReport]:
    """Prepare a one-row-per-WP feature table for the model.

    Missing model columns are added as NaN and handled by the fitted model's
    imputers. Known derived fields are calculated whenever their base values are
    available. Outcome columns are ignored, never used as predictors.
    """
    if frame is None or frame.empty:
        raise InputValidationError("Data fitur kosong.")
    data = _standardize_columns(frame)
    ignored = sorted(set(data.columns) & FORBIDDEN_OUTCOME_COLUMNS)

    if "NPWP" in data.columns and "NPWP16" not in data.columns:
        data = data.rename(columns={"NPWP": "NPWP16"})
    if "NPWP16" in data.columns:
        data["NPWP16"] = normalize_npwp(data["NPWP16"], strict=True)
        invalid_npwp = data["NPWP16"].notna() & data["NPWP16"].str.len().ne(16)
        if invalid_npwp.any():
            raise InputValidationError(f"{int(invalid_npwp.sum())} NPWP tidak memiliki 16 digit.")
        duplicate_npwp = data["NPWP16"].notna() & data["NPWP16"].duplicated(keep=False)
        if duplicate_npwp.any():
            raise InputValidationError(
                "Data fitur harus satu baris per WP; ditemukan "
                f"{data.loc[duplicate_npwp, 'NPWP16'].nunique()} NPWP duplikat."
            )
    else:
        data["NPWP16"] = pd.Series([pd.NA] * len(data), dtype="string")
    if "NAMA_WP" not in data.columns:
        data["NAMA_WP"] = np.nan

    originally_missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
    for column in FEATURE_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan

    for column in NUMERIC_FEATURES:
        data[column] = _coerce_numeric_strict(data[column], column)
    for column in CATEGORICAL_FEATURES:
        data[column] = _as_category(data[column], column)

    data = _fill_derived_features(data)
    report = PreparationReport(
        input_rows=len(frame),
        output_wp=len(data),
        missing_model_features=originally_missing,
        ignored_outcome_columns=ignored,
        warnings=[],
    )
    return data[IDENTIFIER_COLUMNS + FEATURE_COLUMNS], report


def _latest_external_rows(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "TAHUN" in data.columns:
        data["TAHUN"] = _coerce_numeric_strict(data["TAHUN"], "TAHUN")
        data = data.sort_values("TAHUN", na_position="first")
    return data.drop_duplicates("NPWP16", keep="last")


def _prepare_metrics(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["NPWP16", "RASIO_LAPOR_SPT_3THN", "FLAG_LAPOR_SPT_TERAKHIR", "PEREDARAN_BRUTO"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    data = _standardize_columns(frame)
    if "NPWP16" not in data.columns and "NPWP" in data.columns:
        data = data.rename(columns={"NPWP": "NPWP16"})
    if "NPWP16" not in data.columns:
        raise InputValidationError("Data metrik SPT harus memiliki kolom NPWP atau NPWP16.")
    aliases = {
        "RASIO_LAPOR_SPT_3THN": "RASIO_LAPOR_SPT_3THN",
        "FLAG_LAPOR_SPT_TERAKHIR": "FLAG_LAPOR_SPT_TERAKHIR",
        "PEREDARAN_BRUTO": "PEREDARAN_BRUTO",
    }
    data["NPWP16"] = normalize_npwp(data["NPWP16"], strict=True)
    if data["NPWP16"].isna().any():
        raise InputValidationError("Data metrik SPT mengandung NPWP kosong.")
    for target, source in aliases.items():
        data[target] = _coerce_numeric_strict(data[source], source) if source in data.columns else np.nan
    return _latest_external_rows(data)[columns]


def _prepare_partner(frame: pd.DataFrame | None, kind: str) -> pd.DataFrame:
    if kind not in {"customer", "supplier"}:
        raise ValueError(kind)
    prefix = kind.upper()
    n_col = f"JML_{prefix}"
    dpp_col = f"DPP_{prefix}"
    invoice_col = f"FAKTUR_{prefix}"
    columns = ["NPWP16", n_col, dpp_col, invoice_col]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)

    data = _standardize_columns(frame)
    if "NPWP16" not in data.columns and "NPWP" in data.columns:
        data = data.rename(columns={"NPWP": "NPWP16"})
    if "NPWP16" not in data.columns:
        raise InputValidationError(f"Data {kind} harus memiliki kolom NPWP atau NPWP16.")

    aliases = {
        n_col: [n_col, f"JML_{prefix}"],
        dpp_col: [dpp_col, "TOTAL_JML_DPP"],
        invoice_col: [invoice_col, "TOTAL_JML_FAKTUR"],
    }
    data["NPWP16"] = normalize_npwp(data["NPWP16"], strict=True)
    if data["NPWP16"].isna().any():
        raise InputValidationError(f"Data {kind} mengandung NPWP kosong.")
    for target, candidates in aliases.items():
        source = next((c for c in candidates if c in data.columns), None)
        data[target] = _coerce_numeric_strict(data[source], source) if source else np.nan
    return _latest_external_rows(data)[columns]


def aggregate_raw_kohir(
    raw: pd.DataFrame,
    snapshot_date: str | pd.Timestamp | None = None,
    metrics: pd.DataFrame | None = None,
    customer: pd.DataFrame | None = None,
    supplier: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, PreparationReport]:
    """Aggregate raw kohir rows into the exact WP feature contract.

    The raw input does not need LABEL and any supplied outcome columns are
    ignored. The official extraction/prediction date should be supplied as
    ``snapshot_date``; otherwise the maximum debt date is used as a fallback.
    """
    if raw is None or raw.empty:
        raise InputValidationError("Data kohir kosong.")
    data = _standardize_columns(raw)
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise InputValidationError("Kolom wajib data kohir belum lengkap: " + ", ".join(missing))

    ignored = sorted(set(data.columns) & FORBIDDEN_OUTCOME_COLUMNS)
    input_rows = len(data)
    # Remove outcomes before any operation that can change row multiplicity.
    data = data.drop(columns=ignored, errors="ignore")

    optional = [
        "NAMA_WP", "STS_WP", "JENIS_WP", "JENIS_KPP_BKM", "KD_KANWIL", "KD_KLU",
        "TGL_PRODUK_HUKUM", "TGL_DALUWARSA", "TH_PJK",
    ]
    for column in optional:
        if column not in data.columns:
            data[column] = np.nan

    data["NPWP16"] = normalize_npwp(data["NPWP16"], strict=True)
    invalid_npwp = data["NPWP16"].isna() | data["NPWP16"].str.len().ne(16)
    if invalid_npwp.any():
        raise InputValidationError(
            f"{int(invalid_npwp.sum())} baris memiliki NPWP kosong/tidak valid (harus 16 digit)."
        )

    data["NILAI_STPSKP"] = _coerce_numeric_strict(data["NILAI_STPSKP"], "NILAI_STPSKP")
    data["TH_PJK"] = _coerce_numeric_strict(data["TH_PJK"], "TH_PJK")
    for column in ["TGL_PRODUK_HUKUM", "TGL_UTANG_DPT_DITAGIH", "TGL_DALUWARSA"]:
        data[column] = _coerce_date_strict(data[column], column)

    for column in ["NAMA_WP", "STS_WP", "JENIS_WP", "JENIS_KPP_BKM", "KD_KANWIL", "KD_KLU", "NO_STPSKP", "JENIS_PAJAK"]:
        data[column] = data[column].astype("string").str.strip().replace("", pd.NA)
    for required_value in ["NO_STPSKP", "JENIS_PAJAK", "NILAI_STPSKP", "TGL_UTANG_DPT_DITAGIH"]:
        missing_values = data[required_value].isna()
        if missing_values.any():
            raise InputValidationError(
                f"Kolom wajib {required_value} kosong pada {int(missing_values.sum())} baris."
            )

    data["JENIS_KPP_BKM"] = data["JENIS_KPP_BKM"].replace(KPP_MAP)
    data["KD_KANWIL"] = data["KD_KANWIL"].str.replace(r"\.0$", "", regex=True).str.zfill(3)
    klu = data["KD_KLU"].str.replace(r"\.0$", "", regex=True).str.zfill(5)
    data["KD_KLU"] = klu

    # Canonical deduplication is independent of LABEL/SETOR_* and happens only
    # after NPWP/numeric/date/category normalization.
    canonical_columns = [c for c in RAW_CANONICAL_COLUMNS if c in data.columns]
    data = data.drop_duplicates(subset=canonical_columns).copy()
    duplicate_rows = input_rows - len(data)
    duplicate_kohir = data.duplicated(["NPWP16", "NO_STPSKP"], keep=False)
    if duplicate_kohir.any():
        raise InputValidationError(
            "Satu NO_STPSKP memiliki nilai predictor yang saling bertentangan: "
            f"{data.loc[duplicate_kohir, ['NPWP16', 'NO_STPSKP']].drop_duplicates().shape[0]} kohir."
        )

    warnings: list[str] = []
    if snapshot_date is None:
        snapshot = data["TGL_UTANG_DPT_DITAGIH"].max()
        warnings.append(
            "Tanggal snapshot tidak diberikan; aplikasi memakai maksimum "
            "TGL_UTANG_DPT_DITAGIH sebagai fallback."
        )
    else:
        snapshot = pd.to_datetime(snapshot_date, errors="coerce")
        if pd.isna(snapshot):
            raise InputValidationError("Tanggal snapshot tidak valid.")

    future_debt = (data["TGL_UTANG_DPT_DITAGIH"] > snapshot).sum()
    if future_debt:
        warnings.append(f"{future_debt:,} baris memiliki tanggal utang setelah snapshot.")

    data["SEKTOR_KLU"] = klu.loc[data.index].str[0].map(SECTOR_MAP).fillna("LAINNYA")
    data["KODE_KETETAPAN"] = data["NO_STPSKP"].str.split("/").str[1].fillna("UNKNOWN")
    data["SELISIH_TAHUN_TERBIT"] = data["TGL_PRODUK_HUKUM"].dt.year - data["TH_PJK"]
    # The serialized V2 model was trained with age since the legal product date
    # (despite the historical feature name UMUR_UTANG_*); preserve that serving contract.
    data["UMUR_UTANG_HARI"] = (snapshot - data["TGL_PRODUK_HUKUM"]).dt.days
    data["SISA_DALUWARSA_HARI"] = (data["TGL_DALUWARSA"] - snapshot).dt.days

    wp = (
        data.groupby("NPWP16", dropna=False)
        .agg(
            NAMA_WP=("NAMA_WP", _mode_safe),
            STS_WP=("STS_WP", _mode_safe),
            JENIS_WP=("JENIS_WP", _mode_safe),
            JENIS_KPP_BKM=("JENIS_KPP_BKM", _mode_safe),
            KD_KANWIL=("KD_KANWIL", _mode_safe),
            SEKTOR_KLU=("SEKTOR_KLU", _mode_safe),
            JENIS_PAJAK_DOMINAN=("JENIS_PAJAK", _mode_safe),
            KODE_KETETAPAN_DOMINAN=("KODE_KETETAPAN", _mode_safe),
            TOTAL_TUNGGAKAN_POKOK=("NILAI_STPSKP", lambda s: s.sum(min_count=1)),
            RATA_TUNGGAKAN=("NILAI_STPSKP", "mean"),
            MEDIAN_TUNGGAKAN=("NILAI_STPSKP", "median"),
            MAX_TUNGGAKAN=("NILAI_STPSKP", "max"),
            JML_KETETAPAN=("NO_STPSKP", "count"),
            JML_JENIS_PAJAK=("JENIS_PAJAK", "nunique"),
            JML_JENIS_KETETAPAN=("KODE_KETETAPAN", "nunique"),
            UMUR_UTANG_RATA=("UMUR_UTANG_HARI", "mean"),
            UMUR_UTANG_TERTUA=("UMUR_UTANG_HARI", "max"),
            SISA_DALUWARSA_TERDEKAT=("SISA_DALUWARSA_HARI", "min"),
            RATA_SELISIH_TAHUN_TERBIT=("SELISIH_TAHUN_TERBIT", "mean"),
        )
        .reset_index()
    )

    metrics_ready = _prepare_metrics(metrics)
    customer_ready = _prepare_partner(customer, "customer")
    supplier_ready = _prepare_partner(supplier, "supplier")
    if metrics is None:
        warnings.append("Metrik SPT/omzet tidak diunggah; model akan memakai imputasi training.")
    if customer is None:
        warnings.append("Data customer tidak diunggah; model akan memakai imputasi training.")
    if supplier is None:
        warnings.append("Data supplier tidak diunggah; model akan memakai imputasi training.")

    wp = (
        wp.merge(metrics_ready, on="NPWP16", how="left", validate="one_to_one")
        .merge(customer_ready, on="NPWP16", how="left", validate="one_to_one")
        .merge(supplier_ready, on="NPWP16", how="left", validate="one_to_one")
    )

    prepared, feature_report = prepare_feature_input(wp)
    report = PreparationReport(
        input_rows=input_rows,
        output_wp=len(prepared),
        duplicate_rows_removed=duplicate_rows,
        snapshot_date=pd.Timestamp(snapshot).date().isoformat(),
        missing_model_features=feature_report.missing_model_features,
        ignored_outcome_columns=ignored,
        warnings=warnings,
    )
    return prepared, report


def predict_wp(model, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict class probabilities for one-row-per-WP input."""
    prepared, _ = prepare_feature_input(frame)
    model_features = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    x = prepared[model_features]
    pred = model.predict(x).astype(int)
    probabilities = model.predict_proba(x)
    classes = [int(c) for c in model.named_steps["clf"].classes_]

    result = prepared[IDENTIFIER_COLUMNS].reset_index(drop=True).copy()
    result["PREDIKSI"] = pred
    result["KELAS_PREDIKSI"] = [LABEL_NAMES[int(v)] for v in pred]
    for idx, cls in enumerate(classes):
        result[f"P_{LABEL_NAMES[cls].upper()}"] = probabilities[:, idx]
    for cls in LABEL_NAMES:
        column = f"P_{LABEL_NAMES[cls].upper()}"
        if column not in result:
            result[column] = 0.0
    result["CONFIDENCE_MODEL"] = probabilities.max(axis=1)
    result["REKOMENDASI"] = [LABEL_DESCRIPTIONS[int(v)] for v in pred]
    return result


def build_manual_feature(values: dict[str, Any]) -> pd.DataFrame:
    """Build a single-row feature frame, including known derived fields."""
    frame = pd.DataFrame([values])
    prepared, _ = prepare_feature_input(frame)
    return prepared


def feature_template() -> pd.DataFrame:
    """Return an empty template for pre-aggregated feature uploads."""
    return pd.DataFrame(columns=IDENTIFIER_COLUMNS + FEATURE_COLUMNS)


def raw_template() -> pd.DataFrame:
    """Return an empty template for raw kohir uploads."""
    return pd.DataFrame(columns=RAW_RECOMMENDED_COLUMNS)
