
"""
Universal file loader for Excel/CSV/PDF with friendly errors.
- Excel: .xlsx, .xls via pandas.read_excel
- CSV/TXT: via pandas.read_csv with delimiter sniff
- PDF: via pdfplumber (best effort) extracting first table
Returns: (df, meta) or (None, {"error": "..."})
"""
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import io

def _read_excel(file) -> pd.DataFrame:
    xls = pd.ExcelFile(file)
    # Prefer a known sheet but fallback to first
    preferred = 'Ausgewählte LISTE - Final'
    sheet = preferred if preferred in xls.sheet_names else xls.sheet_names[0]
    return pd.read_excel(xls, sheet_name=sheet), {"sheet": sheet}

def _detect_delimiter(sample: bytes) -> str:
    text = sample.decode("utf-8", errors="ignore")
    # naive sniff
    if text.count(";") > text.count(",") and text.count(";") > text.count("\t"):
        return ";"
    if text.count("\t") > text.count(","):
        return "\t"
    return ","

def _read_csv(file) -> pd.DataFrame:
    # Read small sample to guess delimiter
    if hasattr(file, "read"):
        pos = file.tell()
        sample = file.read(4096)
        file.seek(pos)
    else:
        sample = open(file, "rb").read(4096)
    delim = _detect_delimiter(sample)
    return pd.read_csv(file, sep=delim), {"delimiter": delim}

def _read_pdf(file) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    meta = {}
    try:
        import pdfplumber
    except Exception:
        return None, {"error": "PDF-Unterstützung benötigt 'pdfplumber'. Bitte installieren: pip install pdfplumber"}
    try:
        if hasattr(file, "read"):
            # pdfplumber expects a path or file-like object; ensure bytes
            data = file.read()
            fobj = io.BytesIO(data)
        else:
            fobj = file
        with pdfplumber.open(fobj) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    # Take first non-empty table
                    tbl = tables[0]
                    # Assume first row headers
                    if len(tbl) >= 2:
                        headers = tbl[0]
                        rows = tbl[1:]
                        df = pd.DataFrame(rows, columns=headers)
                        meta["page"] = page.page_number
                        meta["pdf_tables_found"] = True
                        return df, meta
            return None, {"error": "Keine Tabelle im PDF gefunden."}
    except Exception as e:
        return None, {"error": f"PDF konnte nicht gelesen werden: {e}"}

import pandas as pd
import io

def load_any(file, filename="upload", preview_only=False):
    """
    Lädt Datei (Excel, CSV, TXT, PDF) als DataFrame + Meta-Info.
    Falls preview_only=True, werden nur die ersten paar Zeilen gelesen (für schnelleres Mapping).
    """
    try:
        ext = filename.split(".")[-1].lower()

        if ext in ["xlsx", "xls"]:
            xls = pd.ExcelFile(file)
            sheet = xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=sheet, nrows=20 if preview_only else None)
            return df, {"type": "excel", "sheet": sheet}

        elif ext in ["csv", "txt"]:
            # Sniff delimiter
            if hasattr(file, "read"):
                pos = file.tell()
                sample = file.read(4096)
                file.seek(pos)
            else:
                with open(file, "rb") as f:
                    sample = f.read(4096)
            delim = _detect_delimiter(sample)
            df = pd.read_csv(file, sep=delim, nrows=20 if preview_only else None)
            return df, {"type": ext, "delimiter": delim}

        elif ext == "pdf":
            df, meta = _read_pdf(file)
            if df is not None and preview_only:
                df = df.head(20)
            return df, {"type": "pdf", **meta}

        else:
            return None, {"error": f"Nicht unterstütztes Format: {ext}"}

    except Exception as e:
        return None, {"error": str(e)}
