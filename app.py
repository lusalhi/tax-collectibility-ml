"""Streamlit website for WP collectibility V2 inference."""
from __future__ import annotations

import hashlib
import io
import json
import logging
from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st

from src.inference import (
    CATEGORICAL_FEATURES,
    LABEL_DESCRIPTIONS,
    NUMERIC_FEATURES,
    InputValidationError,
    aggregate_raw_kohir,
    build_manual_feature,
    feature_template,
    load_model,
    predict_wp,
    prepare_feature_input,
    raw_template,
)
from src.security import to_csv_bytes

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "ketertagihan_wp_v2.joblib"
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = 140 * 1024 * 1024
LOGGER = logging.getLogger("ketertagihan_app")

st.set_page_config(
    page_title="Prediksi Ketertagihan WP",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1450px;}
    .hero {
        padding: 1.35rem 1.55rem; border-radius: 16px;
        background: linear-gradient(120deg, #0f3d56 0%, #176b73 58%, #2e8b75 100%);
        color: white; margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(15, 61, 86, .14);
    }
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
    .hero p {margin: 0; opacity: .92; font-size: 1rem;}
    .result-card {
        border: 1px solid rgba(49, 51, 63, .15); border-radius: 14px;
        padding: 1.1rem 1.25rem; background: rgba(250, 250, 250, .55);
        margin: .5rem 0 1rem 0;
    }
    .badge-rendah, .badge-sedang, .badge-tinggi {
        display: inline-block; padding: .35rem .75rem; border-radius: 999px;
        color: white; font-weight: 700; letter-spacing: .02em;
    }
    .badge-rendah {background: #b23a48;}
    .badge-sedang {background: #d28b20;}
    .badge-tinggi {background: #23856d;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, .12); border-radius: 12px;
        padding: .75rem 1rem; background: rgba(255,255,255,.6);
    }
    .small-note {font-size: .86rem; color: #6b7280;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model(MODEL_PATH)


def read_uploaded_file(uploaded) -> pd.DataFrame:
    """Read CSV/TXT, XLSX, or JSON uploaded through Streamlit."""
    if uploaded is None:
        raise InputValidationError("File belum dipilih.")
    suffix = Path(uploaded.name).suffix.lower()
    payload = uploaded.getvalue()
    if len(payload) > MAX_FILE_BYTES:
        raise InputValidationError(f"{uploaded.name} melebihi batas 100 MB per file.")
    try:
        if suffix in {".csv", ".txt"}:
            last_error = None
            for encoding in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    return pd.read_csv(
                        io.BytesIO(payload), sep=None, engine="python", dtype=str,
                        encoding=encoding,
                    )
                except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                    last_error = exc
            raise last_error or ValueError("CSV tidak dapat dibaca")
        if suffix == ".xlsx":
            return pd.read_excel(io.BytesIO(payload), dtype=str)
        if suffix == ".json":
            obj = json.loads(payload.decode("utf-8-sig"))
            if isinstance(obj, dict):
                for key in ("data", "results", "items"):
                    if isinstance(obj.get(key), list):
                        obj = obj[key]
                        break
            return pd.DataFrame(obj)
    except Exception as exc:
        raise InputValidationError(f"Gagal membaca {uploaded.name}: {exc}") from exc
    raise InputValidationError("Format tidak didukung. Gunakan CSV, TXT, XLSX, atau JSON.")


def validate_total_upload_size(*uploads) -> None:
    total = sum(len(upload.getvalue()) for upload in uploads if upload is not None)
    if total > MAX_TOTAL_UPLOAD_BYTES:
        raise InputValidationError("Total file upload melebihi batas 140 MB.")


def input_fingerprint(mode: str, snapshot, *uploads) -> str:
    digest = hashlib.sha256()
    digest.update(mode.encode())
    digest.update(str(snapshot).encode())
    for upload in uploads:
        if upload is None:
            digest.update(b"<none>")
        else:
            # Streamlit assigns a new file_id when an upload changes. Use its
            # metadata rather than hashing potentially large confidential data.
            upload_size = getattr(upload, "size", None)
            if upload_size is None:
                upload_size = len(upload.getbuffer())
            identity = (
                getattr(upload, "file_id", ""),
                upload.name,
                upload_size,
                getattr(upload, "type", ""),
            )
            digest.update(repr(identity).encode("utf-8", errors="replace"))
    return digest.hexdigest()


def clear_batch_state() -> None:
    for key in ("batch_result", "batch_report", "batch_mode", "batch_fingerprint"):
        st.session_state.pop(key, None)


def show_unexpected_error(context: str) -> None:
    error_id = uuid4().hex[:10]
    LOGGER.exception("%s failed [error_id=%s]", context, error_id)
    st.error(f"Terjadi kesalahan internal. ID insiden: {error_id}")


def mask_npwp(value) -> str:
    if pd.isna(value):
        return "—"
    text = str(value).zfill(16)
    return f"{text[:4]}••••••••{text[-4:]}"


def show_report(report) -> None:
    with st.expander("Laporan penyiapan data", expanded=False):
        a, b, c = st.columns(3)
        a.metric("Baris input", f"{report.input_rows:,}")
        b.metric("WP siap prediksi", f"{report.output_wp:,}")
        c.metric("Duplikat dibuang", f"{report.duplicate_rows_removed:,}")
        if report.snapshot_date:
            st.write("**Tanggal snapshot:**", report.snapshot_date)
        if report.ignored_outcome_columns:
            st.info(
                "Kolom outcome ditemukan dan sengaja diabaikan: "
                + ", ".join(report.ignored_outcome_columns)
            )
        if report.missing_model_features:
            st.write(
                f"Kolom model yang tidak tersedia di input awal: {len(report.missing_model_features)}. "
                "Sebagian diturunkan otomatis; sisanya memakai imputer yang telah di-fit saat training."
            )
        for warning in report.warnings or []:
            st.warning(warning)


def show_batch_results(results: pd.DataFrame, report) -> None:
    st.success(f"Prediksi selesai untuk {len(results):,} WP.")
    counts = results["KELAS_PREDIKSI"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total WP", f"{len(results):,}")
    c2.metric("Rendah", f"{int(counts.get('Rendah', 0)):,}")
    c3.metric("Sedang", f"{int(counts.get('Sedang', 0)):,}")
    c4.metric("Tinggi", f"{int(counts.get('Tinggi', 0)):,}")

    left, right = st.columns([1, 2])
    with left:
        chart = counts.reindex(["Rendah", "Sedang", "Tinggi"], fill_value=0).rename("Jumlah WP")
        st.bar_chart(chart, color="#176b73")
    with right:
        sort_choice = st.selectbox(
            "Urutkan hasil",
            ["Peluang Rendah tertinggi", "Peluang Tinggi tertinggi", "Confidence tertinggi"],
            key="sort_batch_result",
        )
        sort_column = {
            "Peluang Rendah tertinggi": "P_RENDAH",
            "Peluang Tinggi tertinggi": "P_TINGGI",
            "Confidence tertinggi": "CONFIDENCE_MODEL",
        }[sort_choice]
        display = results.sort_values(sort_column, ascending=False).copy()
        if not st.session_state.get("show_full_npwp", False):
            display["NPWP16"] = display["NPWP16"].map(mask_npwp)
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                "P_RENDAH": st.column_config.ProgressColumn("P Rendah", min_value=0.0, max_value=1.0, format="percent"),
                "P_SEDANG": st.column_config.ProgressColumn("P Sedang", min_value=0.0, max_value=1.0, format="percent"),
                "P_TINGGI": st.column_config.ProgressColumn("P Tinggi", min_value=0.0, max_value=1.0, format="percent"),
                "CONFIDENCE_MODEL": st.column_config.NumberColumn("Confidence", format="%.3f"),
            },
        )

    st.caption("Probabilitas adalah skor relatif model dan belum dikalibrasi sebagai peluang absolut.")
    st.download_button(
        "⬇️ Unduh hasil lengkap (CSV)",
        data=to_csv_bytes(results),
        file_name=f"prediksi_ketertagihan_{date.today().isoformat()}.csv",
        mime="text/csv",
        type="primary",
    )
    show_report(report)


def render_manual_result(result: pd.DataFrame) -> None:
    row = result.iloc[0]
    label = str(row["KELAS_PREDIKSI"])
    css = {"Rendah": "badge-rendah", "Sedang": "badge-sedang", "Tinggi": "badge-tinggi"}[label]
    st.markdown(
        f"""
        <div class="result-card">
          <div class="small-note">HASIL PREDIKSI</div>
          <h2><span class="{css}">{label}</span></h2>
          <p>{LABEL_DESCRIPTIONS[int(row['PREDIKSI'])]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P Rendah", f"{row['P_RENDAH']:.1%}")
    c2.metric("P Sedang", f"{row['P_SEDANG']:.1%}")
    c3.metric("P Tinggi", f"{row['P_TINGGI']:.1%}")
    c4.metric("Confidence", f"{row['CONFIDENCE_MODEL']:.1%}")
    st.caption("Confidence/probabilitas belum dikalibrasi dan tidak menggantikan judgment petugas.")


try:
    model = get_model()
except Exception as exc:
    st.error(f"Aplikasi tidak dapat memuat model: {exc}")
    st.info("Pastikan `models/ketertagihan_wp_v2.joblib` tersedia dan jalankan aplikasi dari root proyek.")
    st.stop()

with st.sidebar:
    st.markdown("## 🏛️ Ketertagihan WP")
    st.success("Model V2 siap")
    st.caption("Logistic Regression · 36 fitur · 3 kelas")
    st.info("Mode lokal saja (127.0.0.1)")
    st.metric("Test F1-macro", "0,562")
    st.divider()
    st.checkbox("Tampilkan NPWP lengkap", value=False, key="show_full_npwp")
    st.caption("Secara default NPWP dimasking di layar. File unduhan tetap memuat identitas lengkap.")
    st.divider()
    st.warning("Baseline cross-sectional, belum validasi prospektif/temporal.")
    st.caption("Model tidak memakai SETOR_*, NILAI_SISA, atau tindakan penagihan sebagai fitur.")

st.markdown(
    """
    <div class="hero">
      <h1>Prediksi Ketertagihan Wajib Pajak</h1>
      <p>Skoring kapasitas & kemauan bayar pada level WP — Rendah, Sedang, atau Tinggi.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Model memberi rekomendasi triase berbasis pola historis. Hasil wajib dikombinasikan "
    "dengan informasi aset, sengketa, dan judgment petugas."
)

batch_tab, manual_tab, guide_tab = st.tabs(["📦 Prediksi Batch", "👤 Prediksi Manual", "📘 Panduan"])

with batch_tab:
    st.subheader("Prediksi data baru secara batch")
    input_mode = st.radio(
        "Bentuk data input",
        ["Kohir mentah (diagregasi otomatis)", "Fitur WP siap model"],
        horizontal=True,
    )

    if input_mode.startswith("Kohir"):
        st.write(
            "Unggah data satu baris per kohir. LABEL tidak diperlukan. "
            "Metrik SPT/customer/supplier bersifat opsional, tetapi sangat disarankan."
        )
        raw_file = st.file_uploader(
            "Data kohir utama (CSV/XLSX)", type=["csv", "txt", "xlsx"], key="raw_file"
        )
        snapshot = st.date_input(
            "Tanggal snapshot/prediksi",
            value=date.today(),
            help="Gunakan tanggal resmi tarikan data. Umur utang dan sisa daluwarsa dihitung relatif terhadap tanggal ini.",
        )
        e1, e2, e3 = st.columns(3)
        with e1:
            metrics_file = st.file_uploader(
                "Metrik SPT/omzet", type=["csv", "txt", "xlsx", "json"], key="metrics_file"
            )
        with e2:
            customer_file = st.file_uploader(
                "Data customer", type=["csv", "txt", "xlsx", "json"], key="customer_file"
            )
        with e3:
            supplier_file = st.file_uploader(
                "Data supplier", type=["csv", "txt", "xlsx", "json"], key="supplier_file"
            )

        t1, t2 = st.columns(2)
        t1.download_button(
            "Unduh template kohir",
            data=to_csv_bytes(raw_template()),
            file_name="template_kohir_v2.csv",
            mime="text/csv",
        )
        t2.caption("CSV dapat memakai pemisah koma atau titik koma.")
        current_fingerprint = input_fingerprint(
            "raw", snapshot, raw_file, metrics_file, customer_file, supplier_file
        )

        if st.button("🚀 Jalankan prediksi batch", type="primary", key="predict_raw"):
            clear_batch_state()
            if raw_file is None:
                st.warning("Pilih data kohir utama terlebih dahulu.")
            else:
                try:
                    validate_total_upload_size(raw_file, metrics_file, customer_file, supplier_file)
                    with st.spinner("Menyiapkan fitur dan menjalankan model..."):
                        raw_df = read_uploaded_file(raw_file)
                        metrics_df = read_uploaded_file(metrics_file) if metrics_file else None
                        customer_df = read_uploaded_file(customer_file) if customer_file else None
                        supplier_df = read_uploaded_file(supplier_file) if supplier_file else None
                        prepared, report = aggregate_raw_kohir(
                            raw_df,
                            snapshot_date=snapshot,
                            metrics=metrics_df,
                            customer=customer_df,
                            supplier=supplier_df,
                        )
                        result = predict_wp(model, prepared)
                        st.session_state["batch_result"] = result
                        st.session_state["batch_report"] = report
                        st.session_state["batch_mode"] = "raw"
                        st.session_state["batch_fingerprint"] = current_fingerprint
                except (InputValidationError, ValueError) as exc:
                    st.error(str(exc))
                except Exception:
                    show_unexpected_error("raw batch prediction")
    else:
        st.write(
            "Unggah satu baris per WP dengan 36 fitur model. Kolom yang belum tersedia akan "
            "diimputasi, tetapi input lengkap memberikan hasil paling konsisten."
        )
        feature_file = st.file_uploader(
            "Data fitur WP (CSV/XLSX)", type=["csv", "txt", "xlsx"], key="feature_file"
        )
        st.download_button(
            "Unduh template 36 fitur",
            data=to_csv_bytes(feature_template()),
            file_name="template_fitur_wp_v2.csv",
            mime="text/csv",
        )
        current_fingerprint = input_fingerprint("features", None, feature_file)
        if st.button("🚀 Jalankan prediksi batch", type="primary", key="predict_features"):
            clear_batch_state()
            if feature_file is None:
                st.warning("Pilih data fitur WP terlebih dahulu.")
            else:
                try:
                    validate_total_upload_size(feature_file)
                    with st.spinner("Memvalidasi fitur dan menjalankan model..."):
                        feature_df = read_uploaded_file(feature_file)
                        prepared, report = prepare_feature_input(feature_df)
                        result = predict_wp(model, prepared)
                        st.session_state["batch_result"] = result
                        st.session_state["batch_report"] = report
                        st.session_state["batch_mode"] = "features"
                        st.session_state["batch_fingerprint"] = current_fingerprint
                except (InputValidationError, ValueError) as exc:
                    st.error(str(exc))
                except Exception:
                    show_unexpected_error("feature batch prediction")

    expected_mode = "raw" if input_mode.startswith("Kohir") else "features"
    result_is_current = (
        st.session_state.get("batch_mode") == expected_mode
        and st.session_state.get("batch_fingerprint") == current_fingerprint
        and "batch_result" in st.session_state
    )
    if result_is_current:
        st.divider()
        show_batch_results(st.session_state["batch_result"], st.session_state["batch_report"])

with manual_tab:
    st.subheader("Prediksi satu WP")
    st.write("Isi profil agregat WP. Nilai log, rasio tunggakan/omzet, flag missing, dan total mitra dihitung otomatis.")

    with st.form("manual_form"):
        st.markdown("#### Identitas & profil")
        a, b = st.columns(2)
        npwp = a.text_input("NPWP16", placeholder="16 digit", max_chars=20)
        nama = b.text_input("Nama WP (opsional)")
        c1, c2, c3, c4 = st.columns(4)
        sts = c1.selectbox("Status WP", ["AKTIF", "NE", "DE", "UNKNOWN"])
        jenis_wp = c2.selectbox("Jenis WP", ["BADAN", "OP", "BENDAHARA"])
        jenis_kpp = c3.selectbox("Jenis KPP", ["PRATAMA", "MADYA", "KHUSUS", "BESAR"])
        kanwil = c4.selectbox(
            "Kode Kanwil",
            ["040", "050", "080", "090", "100", "110", "120", "130", "140", "150", "160", "200", "210", "220", "260", "270", "310", "320", "330", "340"],
            index=14,
        )
        c1, c2, c3 = st.columns(3)
        sektor = c1.selectbox(
            "Sektor KLU",
            ["KONSTRUKSI", "PERDAGANGAN", "INDUSTRI", "PERTANIAN", "PERTAMBANGAN", "ENERGI/AIR", "TRANSPORTASI", "INFORMASI", "JASA", "JASA_LAIN", "LAINNYA"],
        )
        pajak = c2.selectbox(
            "Jenis pajak dominan",
            ["411211", "411121", "411126", "411128", "411124", "411125", "411122", "411622", "411315", "411313", "411621", "411127", "411212", "411319", "411123"],
        )
        ketetapan = c3.selectbox(
            "Kode ketetapan dominan",
            ["107", "101", "106", "105", "103", "240", "140", "207", "203", "201", "110", "202", "174", "199", "102", "205"],
        )

        st.markdown("#### Portofolio tunggakan")
        c1, c2, c3, c4 = st.columns(4)
        total = c1.number_input("Total tunggakan (Rp)", min_value=0.0, value=100_000_000.0, step=1_000_000.0)
        jumlah = c2.number_input("Jumlah ketetapan", min_value=1, value=10, step=1)
        median = c3.number_input("Median ketetapan (Rp)", min_value=0.0, value=5_000_000.0, step=500_000.0)
        maksimum = c4.number_input("Ketetapan maksimum (Rp)", min_value=0.0, value=50_000_000.0, step=1_000_000.0)
        c1, c2, c3, c4 = st.columns(4)
        n_pajak = c1.number_input("Jumlah jenis pajak", min_value=1, value=2, step=1)
        n_ketetapan = c2.number_input("Jumlah jenis ketetapan", min_value=1, value=2, step=1)
        umur_rata = c3.number_input("Rata-rata umur utang (hari)", min_value=0.0, value=1000.0, step=30.0)
        umur_tua = c4.number_input("Umur utang tertua (hari)", min_value=0.0, value=2000.0, step=30.0)
        c1, c2 = st.columns(2)
        sisa_daluwarsa = c1.number_input("Sisa daluwarsa terdekat (hari; boleh negatif)", value=365.0, step=30.0)
        selisih_terbit = c2.number_input("Rata-rata selisih tahun terbit", value=1.0, step=0.5)

        st.markdown("#### SPT, omzet, dan aktivitas faktur")
        spt_available = st.checkbox("Data SPT tersedia", value=True)
        c1, c2, c3 = st.columns(3)
        rasio_spt = c1.number_input("Rasio lapor SPT 3 tahun", min_value=0.0, max_value=1.0, value=0.67, step=0.01)
        lapor_terakhir = c2.checkbox("Lapor SPT terakhir", value=True)
        omzet = c3.number_input("Peredaran bruto (Rp)", min_value=0.0, value=500_000_000.0, step=1_000_000.0)
        c1, c2, c3 = st.columns(3)
        n_customer = c1.number_input("Jumlah customer", min_value=0.0, value=10.0, step=1.0)
        dpp_customer = c2.number_input("DPP customer (Rp)", min_value=0.0, value=200_000_000.0, step=1_000_000.0)
        faktur_customer = c3.number_input("Jumlah faktur customer", min_value=0.0, value=50.0, step=1.0)
        c1, c2, c3 = st.columns(3)
        n_supplier = c1.number_input("Jumlah supplier", min_value=0.0, value=8.0, step=1.0)
        dpp_supplier = c2.number_input("DPP supplier (Rp)", min_value=0.0, value=150_000_000.0, step=1_000_000.0)
        faktur_supplier = c3.number_input("Jumlah faktur supplier", min_value=0.0, value=40.0, step=1.0)

        submitted = st.form_submit_button("🔎 Prediksi WP", type="primary", width="stretch")

    if submitted:
        st.session_state.pop("manual_result", None)
        if maksimum < median:
            st.warning("Ketetapan maksimum lebih kecil daripada median; periksa kembali input.")
        values = {
            "NPWP16": npwp,
            "NAMA_WP": nama,
            "STS_WP": sts,
            "JENIS_WP": jenis_wp,
            "JENIS_KPP_BKM": jenis_kpp,
            "KD_KANWIL": kanwil,
            "SEKTOR_KLU": sektor,
            "JENIS_PAJAK_DOMINAN": pajak,
            "KODE_KETETAPAN_DOMINAN": ketetapan,
            "TOTAL_TUNGGAKAN_POKOK": total,
            "RATA_TUNGGAKAN": total / jumlah,
            "MEDIAN_TUNGGAKAN": median,
            "MAX_TUNGGAKAN": maksimum,
            "JML_KETETAPAN": jumlah,
            "JML_JENIS_PAJAK": n_pajak,
            "JML_JENIS_KETETAPAN": n_ketetapan,
            "UMUR_UTANG_RATA": umur_rata,
            "UMUR_UTANG_TERTUA": umur_tua,
            "SISA_DALUWARSA_TERDEKAT": sisa_daluwarsa,
            "RATA_SELISIH_TAHUN_TERBIT": selisih_terbit,
            "RASIO_LAPOR_SPT_3THN": rasio_spt if spt_available else np.nan,
            "FLAG_LAPOR_SPT_TERAKHIR": int(lapor_terakhir) if spt_available else np.nan,
            "PEREDARAN_BRUTO": omzet if spt_available else np.nan,
            "JML_CUSTOMER": n_customer,
            "DPP_CUSTOMER": dpp_customer,
            "FAKTUR_CUSTOMER": faktur_customer,
            "JML_SUPPLIER": n_supplier,
            "DPP_SUPPLIER": dpp_supplier,
            "FAKTUR_SUPPLIER": faktur_supplier,
        }
        try:
            manual_features = build_manual_feature(values)
            manual_result = predict_wp(model, manual_features)
            st.session_state["manual_result"] = manual_result
        except (InputValidationError, ValueError) as exc:
            st.error(f"Prediksi gagal: {exc}")
        except Exception:
            show_unexpected_error("manual prediction")

    if "manual_result" in st.session_state:
        st.divider()
        render_manual_result(st.session_state["manual_result"])

with guide_tab:
    st.subheader("Panduan penggunaan")
    st.markdown(
        """
        ### 1. Pilih jalur input
        - **Kohir mentah:** paling praktis; aplikasi mengagregasi per NPWP dan menghitung fitur turunan.
        - **Fitur WP siap model:** untuk pipeline data yang sudah menghasilkan satu baris per WP.
        - **Prediksi manual:** untuk simulasi satu WP.

        ### 2. Lengkapi data eksternal
        Untuk hasil terbaik, sertakan metrik SPT/omzet serta customer dan supplier. Jika tidak
        tersedia, model memakai statistik imputasi yang dipelajari saat training dan aplikasi
        menampilkan peringatan kualitas.

        ### 3. Interpretasi
        - **Rendah (0):** belum tampak sinyal kapasitas/kemauan yang kuat.
        - **Sedang (1):** pola mirip pembayaran setelah tindakan lanjutan.
        - **Tinggi (2):** pola mirip pembayaran pada tahap teguran/quick win.

        Probabilitas merupakan skor relatif yang **belum dikalibrasi**. Jangan menggunakannya
        sebagai satu-satunya dasar tindakan penagihan.
        """
    )
    with st.expander("Kolom minimal kohir mentah"):
        st.code("NPWP16, NO_STPSKP, JENIS_PAJAK, NILAI_STPSKP, TGL_UTANG_DPT_DITAGIH")
        st.caption(
            "Kolom profil, tanggal produk hukum/daluwarsa, TH_PJK, dan data eksternal sangat disarankan. "
            "Kontrak model menghitung UMUR_UTANG_* sejak TGL_PRODUK_HUKUM."
        )
    with st.expander("Kontrak 36 fitur model"):
        st.write("**Numerik**")
        st.code(", ".join(NUMERIC_FEATURES))
        st.write("**Kategorikal**")
        st.code(", ".join(CATEGORICAL_FEATURES))
    with st.expander("Batasan model"):
        st.markdown(
            """
            - Baseline cross-sectional; belum diuji dengan out-of-time/prospective validation.
            - Target WP berasal dari modus LABEL kohir berlabel.
            - Data aset, sengketa, reachability, dan kondisi hukum belum tersedia.
            - Test F1-macro artefak V2 saat ini: 0,562; performa dapat berubah pada populasi/waktu baru.
            - Model sengaja tidak memakai pembayaran dan tindakan penagihan sebagai fitur.
            """
        )
    st.markdown("### Menjalankan secara lokal")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
    st.caption("Data upload diproses di memori proses Streamlit; aplikasi ini tidak menyimpan upload secara otomatis.")
