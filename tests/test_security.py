from __future__ import annotations

import unittest

import pandas as pd

from src.security import neutralize_spreadsheet_formulas, to_csv_bytes


class CsvSecurityTests(unittest.TestCase):
    def test_formula_markers_and_control_prefixes_are_neutralized(self):
        dangerous = ["=CMD()", "+SUM(1,2)", "-1+2", "@IMPORT", "\t=CMD()", "\r@SUM(A1)", "  =1+1"]
        frame = pd.DataFrame({"NAMA_WP": dangerous + ["Nama Aman"], "VALUE": range(8)})

        safe = neutralize_spreadsheet_formulas(frame)

        for value in safe["NAMA_WP"].iloc[:-1]:
            self.assertTrue(value.startswith("'"), value)
        self.assertEqual(safe["NAMA_WP"].iloc[-1], "Nama Aman")
        exported = to_csv_bytes(frame).decode("utf-8-sig")
        self.assertIn("'=CMD()", exported)
        self.assertIn("'\t=CMD()", exported)

    def test_numeric_negative_values_remain_numeric(self):
        frame = pd.DataFrame({"VALUE": [-10.0, 5.0]})
        safe = neutralize_spreadsheet_formulas(frame)
        pd.testing.assert_frame_equal(frame, safe)


if __name__ == "__main__":
    unittest.main()
