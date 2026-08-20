from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


@unittest.skipUnless(
    (Path(__file__).resolve().parents[1] / "models" / "ketertagihan_wp_v2.joblib").exists(),
    "Local model artifact is not available",
)
class StreamlitStateTests(unittest.TestCase):
    VALID_RAW = (
        "NPWP16,NO_STPSKP,JENIS_PAJAK,NILAI_STPSKP,TGL_UTANG_DPT_DITAGIH,"
        "TGL_PRODUK_HUKUM,TGL_DALUWARSA,TH_PJK,STS_WP,JENIS_WP,"
        "JENIS_KPP_BKM,KD_KANWIL,KD_KLU\n"
        "123,X/107/A,411211,10000000,2024-02-01,2024-01-01,2029-01-01,"
        "2023,AKTIF,BADAN,P,040,41000\n"
    ).encode()

    @staticmethod
    def _has_prediction_success(app: AppTest) -> bool:
        return any(str(item.value).startswith("Prediksi selesai") for item in app.success)

    def test_changed_or_invalid_upload_never_shows_stale_result(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()

        app.file_uploader[0].upload("raw.csv", self.VALID_RAW, "text/csv").run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(self._has_prediction_success(app))

        changed = self.VALID_RAW.replace(b"10000000", b"20000000")
        app.file_uploader[0].clear().upload("changed.csv", changed, "text/csv").run()
        self.assertFalse(self._has_prediction_success(app))

        app.file_uploader[0].clear().upload("invalid.csv", b"foo\nbar\n", "text/csv").run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertFalse(self._has_prediction_success(app))
        self.assertTrue(any("Kolom wajib" in str(item.value) for item in app.error))


if __name__ == "__main__":
    unittest.main()
