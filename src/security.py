"""Security helpers for exported tabular data."""
from __future__ import annotations

import pandas as pd

# Spreadsheet engines may ignore leading whitespace/control characters before a
# formula marker. Prefixing the full cell with an apostrophe forces text mode.
_FORMULA_PREFIX = r"^[\s\x00-\x1f]*[=+\-@]"


def neutralize_spreadsheet_formulas(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy safe from common CSV/spreadsheet formula injection."""
    safe = frame.copy()
    for column in safe.select_dtypes(include=["object", "string"]).columns:
        text = safe[column].astype("string")
        formula = text.str.match(_FORMULA_PREFIX, na=False)
        if formula.any():
            values = safe[column].astype(object)
            values.loc[formula] = "'" + text.loc[formula]
            safe[column] = values
    return safe


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Encode a formula-neutralized CSV with an Excel-friendly UTF-8 BOM."""
    return neutralize_spreadsheet_formulas(frame).to_csv(index=False).encode("utf-8-sig")
