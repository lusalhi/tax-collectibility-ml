# Prediksi Ketertagihan Wajib Pajak

Aplikasi website Streamlit untuk menjalankan model ketertagihan WP V2 yang telah dilatih. Model menghasilkan tiga kelas pada level WP:

- **0 — Rendah**
- **1 — Sedang**
- **2 — Tinggi**

Model V2 hanya memakai fitur profil, tunggakan, SPT/omzet, faktur, dan kronologi dasar utang. Kolom pembayaran serta tindakan penagihan seperti `SETOR_*`, `NILAI_SISA`, dan `LABEL` tidak digunakan sebagai prediktor.

## Menjalankan aplikasi

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser. Konfigurasi default hanya bind ke `127.0.0.1` karena aplikasi memproses data rahasia.

Model harus tersedia di:

```text
models/ketertagihan_wp_v2.joblib
```

Folder `models/` dan `dataset/` sengaja di-ignore Git karena berisi artefak/data internal. Dependensi serving dipin agar cocok dengan artefak scikit-learn 1.9.0.

Aplikasi ini **default local-only**. Sebelum membuka akses jaringan/shared deployment, wajib sediakan reverse proxy TLS, SSO, role-based authorization untuk identitas/unduhan, audit log, serta kebijakan retensi. Provision model melalui penyimpanan privat dan jangan memasukkan dataset rahasia ke image/repository publik.

## Jalur input

### 1. Kohir mentah

Aplikasi mengagregasi data menjadi satu baris per WP. Kolom minimal:

```text
NPWP16
NO_STPSKP
JENIS_PAJAK
NILAI_STPSKP
TGL_UTANG_DPT_DITAGIH
```

Kolom yang sangat disarankan:

```text
NAMA_WP, STS_WP, JENIS_WP, JENIS_KPP_BKM, KD_KANWIL, KD_KLU,
TGL_PRODUK_HUKUM, TGL_DALUWARSA, TH_PJK
```

Data tambahan SPT/omzet, customer, dan supplier dapat diunggah terpisah. CSV dengan pemisah koma atau titik koma, Excel `.xlsx`, dan JSON didukung sesuai jenis upload. Batasnya 100 MB per file dan 140 MB secara kumulatif.

Gunakan tanggal `YYYY-MM-DD` atau `DD/MM/YYYY`. Angka mesin disarankan; parser juga menerima format ribuan konsisten seperti `1,000,000.50` atau `1.000.000,50`. Tanggal snapshot wajib merepresentasikan tanggal resmi tarikan/prediksi. Untuk menjaga kontrak artefak model V2, fitur historis bernama `UMUR_UTANG_*` dihitung sejak `TGL_PRODUK_HUKUM`, sedangkan sisa daluwarsa dihitung terhadap snapshot.

### 2. Fitur WP siap model

Unggah satu baris per WP dengan kontrak 36 fitur. Template dapat diunduh langsung dari halaman aplikasi. Kolom yang tidak tersedia akan diimputasi oleh pipeline training, tetapi input lengkap lebih disarankan.

### 3. Prediksi manual

Form manual menghitung fitur log, rasio tunggakan terhadap omzet, flag missing SPT, dan total mitra secara otomatis.

## Output

Aplikasi menampilkan dan dapat mengunduh:

- kelas prediksi;
- probabilitas relatif Rendah/Sedang/Tinggi;
- confidence model;
- rekomendasi triase singkat.

NPWP dimasking secara default pada layar. Hasil CSV unduhan tetap memuat identitas lengkap untuk pengguna lokal yang berwenang; nilai teks diamankan dari spreadsheet formula injection.

## Menjalankan test

```powershell
python -m unittest discover -s tests -v
```

## Batasan

- Baseline cross-sectional, belum validasi prospektif/out-of-time.
- Test F1-macro artefak V2 saat ini sekitar **0,562**.
- Probabilitas belum dikalibrasi sebagai peluang absolut.
- Data aset, sengketa, reachability, dan kondisi hukum belum tersedia.
- Hasil model harus digabungkan dengan informasi lapangan dan judgment petugas.

Data upload diproses di memori proses Streamlit dan tidak disimpan otomatis oleh aplikasi.
