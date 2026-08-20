from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.inference import (
    FEATURE_COLUMNS,
    FORBIDDEN_OUTCOME_COLUMNS,
    InputValidationError,
    aggregate_raw_kohir,
    load_model,
    predict_wp,
    prepare_feature_input,
)


class InferencePreparationTests(unittest.TestCase):
    def test_prepare_feature_input_normalizes_and_derives(self):
        data = pd.DataFrame(
            {
                "npwp16": ["123"],
                "nama_wp": ["Contoh WP"],
                "total_tunggakan_pokok": [100_000_000],
                "max_tunggakan": [60_000_000],
                "jml_ketetapan": [5],
                "peredaran_bruto": [500_000_000],
                "jml_customer": [4],
                "jml_supplier": [6],
                "sts_wp": ["AKTIF"],
                "kd_kanwil": [40],
            }
        )
        prepared, report = prepare_feature_input(data)

        self.assertEqual(prepared.loc[0, "NPWP16"], "NPWP_REDACTED")
        self.assertEqual(list(prepared.columns[2:]), FEATURE_COLUMNS)
        self.assertAlmostEqual(prepared.loc[0, "RATA_TUNGGAKAN"], 20_000_000)
        self.assertAlmostEqual(prepared.loc[0, "RASIO_TUNGGAKAN_PEREDARAN"], 0.2)
        self.assertEqual(prepared.loc[0, "TOTAL_MITRA"], 10)
        self.assertEqual(prepared.loc[0, "KD_KANWIL"], "040")
        self.assertIn("LOG_TOTAL_TUNGGAKAN", report.missing_model_features)

    def test_duplicate_wp_feature_rows_are_rejected(self):
        with self.assertRaises(InputValidationError):
            prepare_feature_input(pd.DataFrame({"NPWP16": ["123", "123"]}))

    @staticmethod
    def _raw_row(**overrides):
        row = {
            "NPWP16": "123",
            "NAMA_WP": "A",
            "STS_WP": "AKTIF",
            "JENIS_WP": "BADAN",
            "JENIS_KPP_BKM": "P",
            "KD_KANWIL": "040",
            "KD_KLU": "41000",
            "NO_STPSKP": "X/107/A",
            "JENIS_PAJAK": "411211",
            "NILAI_STPSKP": "10000000",
            "TGL_PRODUK_HUKUM": "2024-01-01",
            "TGL_UTANG_DPT_DITAGIH": "2024-02-01",
            "TGL_DALUWARSA": "2029-01-01",
            "TH_PJK": "2023",
        }
        row.update(overrides)
        return row

    def test_raw_kohir_aggregates_per_wp_and_ignores_outcome(self):
        raw = pd.DataFrame(
            {
                "NPWP16": ["123", "123", "456"],
                "NAMA_WP": ["A", "A", "B"],
                "STS_WP": ["AKTIF", "AKTIF", "NE"],
                "JENIS_WP": ["BADAN", "BADAN", "OP"],
                "JENIS_KPP_BKM": ["P", "P", "M"],
                "KD_KANWIL": ["040", "040", "100"],
                "KD_KLU": ["41000", "41000", "52000"],
                "NO_STPSKP": ["X/107/A", "X/101/B", "X/107/C"],
                "JENIS_PAJAK": ["411211", "411121", "411211"],
                "NILAI_STPSKP": [10_000_000, 30_000_000, 5_000_000],
                "TGL_PRODUK_HUKUM": ["2024-01-01", "2024-02-01", "2024-03-01"],
                "TGL_UTANG_DPT_DITAGIH": ["2024-02-01", "2024-03-01", "2024-04-01"],
                "TGL_DALUWARSA": ["2029-01-01", "2029-02-01", "2029-03-01"],
                "TH_PJK": [2023, 2023, 2023],
                "SETOR_TEGURAN": [1, 0, 10],
                "LABEL": [2, 0, 2],
            }
        )
        prepared, report = aggregate_raw_kohir(raw, snapshot_date="2025-01-01")

        self.assertEqual(len(prepared), 2)
        wp_a = prepared.loc[prepared["NPWP16"] == "NPWP_REDACTED"].iloc[0]
        self.assertEqual(wp_a["TOTAL_TUNGGAKAN_POKOK"], 40_000_000)
        self.assertEqual(wp_a["JML_KETETAPAN"], 2)
        self.assertEqual(wp_a["JML_JENIS_PAJAK"], 2)
        self.assertIn("LABEL", report.ignored_outcome_columns)
        self.assertIn("SETOR_TEGURAN", report.ignored_outcome_columns)

    def test_outcome_values_cannot_change_deduplication_or_features(self):
        baseline, _ = aggregate_raw_kohir(
            pd.DataFrame([self._raw_row()]), snapshot_date="2025-01-01"
        )
        duplicate_rows = [
            self._raw_row(NPWP16="123"),
            self._raw_row(NPWP16="NPWP_REDACTED"),
        ]
        for column in FORBIDDEN_OUTCOME_COLUMNS:
            duplicate_rows[0][column] = "OUTCOME_A"
            duplicate_rows[1][column] = "OUTCOME_B"
        with_outcomes, report = aggregate_raw_kohir(
            pd.DataFrame(duplicate_rows), snapshot_date="2025-01-01"
        )

        pd.testing.assert_frame_equal(baseline, with_outcomes, check_dtype=False)
        self.assertEqual(report.duplicate_rows_removed, 1)
        self.assertEqual(set(report.ignored_outcome_columns), FORBIDDEN_OUTCOME_COLUMNS)

    def test_locale_numeric_and_dmy_date_are_parsed(self):
        raw = pd.DataFrame([
            self._raw_row(
                NPWP16="123.456",
                NILAI_STPSKP="1.000.000,50",
                TGL_PRODUK_HUKUM="01/02/2024",
                TGL_UTANG_DPT_DITAGIH="02/03/2024",
                TGL_DALUWARSA="01/02/2029",
            )
        ])
        prepared, _ = aggregate_raw_kohir(raw, snapshot_date="2025-01-01")

        self.assertEqual(prepared.loc[0, "NPWP16"], "NPWP_REDACTED")
        self.assertAlmostEqual(prepared.loc[0, "TOTAL_TUNGGAKAN_POKOK"], 1_000_000.5)
        self.assertEqual(prepared.loc[0, "UMUR_UTANG_RATA"], 335)

    def test_malformed_or_scientific_npwp_is_rejected(self):
        for value in ("abc123", "1.234E+15"):
            with self.subTest(value=value), self.assertRaises(InputValidationError):
                aggregate_raw_kohir(
                    pd.DataFrame([self._raw_row(NPWP16=value)]),
                    snapshot_date="2025-01-01",
                )

    def test_partial_invalid_or_non_finite_monetary_value_is_rejected(self):
        for invalid_value in ("bukan angka", "inf", "-inf"):
            rows = [
                self._raw_row(),
                self._raw_row(NO_STPSKP="X/101/B", NILAI_STPSKP=invalid_value),
            ]
            with self.subTest(value=invalid_value), self.assertRaises(InputValidationError):
                aggregate_raw_kohir(pd.DataFrame(rows), snapshot_date="2025-01-01")

    def test_latest_external_year_is_selected(self):
        metrics = pd.DataFrame(
            {
                "NPWP": ["123", "123"],
                "tahun": [2023, 2024],
                "rasio_lapor_spt_3thn": [0.3, 0.9],
                "flag_lapor_spt_terakhir": [0, 1],
                "peredaran_bruto": [100_000_000, 250_000_000],
            }
        )
        prepared, _ = aggregate_raw_kohir(
            pd.DataFrame([self._raw_row()]),
            snapshot_date="2025-01-01",
            metrics=metrics,
        )
        self.assertEqual(prepared.loc[0, "PEREDARAN_BRUTO"], 250_000_000)
        self.assertEqual(prepared.loc[0, "RASIO_LAPOR_SPT_3THN"], 0.9)


class ModelSmokeTests(unittest.TestCase):
    MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ketertagihan_wp_v2.joblib"

    @unittest.skipUnless(MODEL_PATH.exists(), "Local model artifact is not available")
    def test_model_returns_three_probabilities(self):
        model = load_model(self.MODEL_PATH)
        frame, _ = prepare_feature_input(
            pd.DataFrame(
                {
                    "NPWP16": ["123"],
                    "TOTAL_TUNGGAKAN_POKOK": [100_000_000],
                    "MAX_TUNGGAKAN": [50_000_000],
                    "JML_KETETAPAN": [5],
                    "STS_WP": ["AKTIF"],
                }
            )
        )
        result = predict_wp(model, frame)
        probabilities = result.loc[0, ["P_RENDAH", "P_SEDANG", "P_TINGGI"]].astype(float)

        self.assertEqual(int(result.loc[0, "PREDIKSI"]), int(np.argmax(probabilities.to_numpy())))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
