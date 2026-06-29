from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


_CANDIDATE_SEPS: tuple[str, ...] = (",", ";", "|", "\t")
_SNIFF_BYTES = 65536


class AmbiguousCsvSeparatorError(ValueError):
    """Raised when the CSV separator cannot be reliably detected.

    Typical symptom: the parsed DataFrame collapses to a single column whose
    header still contains a candidate separator (``;``, ``|``, tab or ``,``),
    which means pandas could not split the rows correctly.
    """


_AMBIGUOUS_SEP_MESSAGE = (
    "Impossible de détecter de manière fiable le séparateur de ce fichier CSV. "
    "Le système a essayé ',', ';', '|' et la tabulation, mais le fichier semble "
    "utiliser un séparateur ambigu ou mélangé : une seule colonne a été produite. "
    "Merci de ré-enregistrer le fichier avec un séparateur unique et standard "
    "(virgule recommandée) puis de le ré-uploader."
)


def _validate_parse_result(df: pd.DataFrame) -> pd.DataFrame:
    """Detect the classic 'wrong separator' symptom and fail loudly.

    A genuine single-column CSV is valid, so we only raise when the unique
    column header still contains one of our candidate separators — a clear
    sign that the row never got split.
    """
    if df.shape[1] == 1:
        header = str(df.columns[0])
        if any(sep in header for sep in _CANDIDATE_SEPS):
            raise AmbiguousCsvSeparatorError(_AMBIGUOUS_SEP_MESSAGE)
    return df


def _whitespace_only_to_na(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols) == 0:
        return df

    df = df.copy()
    for c in obj_cols:
        # remplace uniquement les valeurs qui matchent "^\s*$" (espaces uniquement)
        df[c] = df[c].replace(to_replace=r"^\s*$", value=pd.NA, regex=True)
    return df


def _sniff_separator(sample: str) -> Optional[str]:
    """Detect the most likely CSV delimiter from a text sample.

    Returns one of ``,``, ``;``, ``|``, ``\\t`` or ``None`` if undetermined.
    Strategy:
      1. csv.Sniffer (handles quoted fields correctly).
      2. Fallback heuristic: pick the candidate whose occurrence count is
         highest *and* most consistent across the first non-empty lines.
    """
    if not sample:
        return None

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(_CANDIDATE_SEPS))
        if dialect.delimiter in _CANDIDATE_SEPS:
            return dialect.delimiter
    except csv.Error:
        pass

    lines = [ln for ln in sample.splitlines() if ln.strip()]
    lines = lines[:20]
    if len(lines) < 2:
        # With a single line we cannot judge consistency — fall back to "most
        # frequent" if the header clearly uses one candidate.
        if len(lines) == 1:
            counts = {sep: lines[0].count(sep) for sep in _CANDIDATE_SEPS}
            best = max(counts, key=counts.get)
            return best if counts[best] > 0 else None
        return None

    best_sep: Optional[str] = None
    best_score = -1.0
    for sep in _CANDIDATE_SEPS:
        counts = [ln.count(sep) for ln in lines]
        if max(counts) == 0:
            continue
        header_count = counts[0]
        if header_count == 0:
            continue
        consistency = sum(1 for c in counts if c == header_count) / len(counts)
        # Reward many fields AND consistent column count across rows.
        score = header_count * consistency
        if score > best_score:
            best_score = score
            best_sep = sep
    return best_sep


def _decode_head(raw: bytes, encoding: str) -> str:
    return raw.decode(encoding, errors="replace")


def _read_csv_with_autosep(
    source,
    *,
    encoding: str,
    sample: str,
    nrows: Optional[int],
    usecols: Optional[Iterable[str]],
    on_bad_lines: Optional[str] = None,
    encoding_errors: Optional[str] = None,
) -> pd.DataFrame:
    """Single pandas read using the sniffed separator (or python-engine auto)."""
    sep = _sniff_separator(sample)
    kwargs: dict = {"nrows": nrows, "encoding": encoding, "usecols": usecols}
    if sep is not None:
        kwargs["sep"] = sep
    else:
        # Last resort: let pandas' python engine try to detect.
        kwargs["sep"] = None
        kwargs["engine"] = "python"
    if on_bad_lines is not None:
        kwargs["on_bad_lines"] = on_bad_lines
        kwargs["engine"] = "python"
    if encoding_errors is not None:
        kwargs["encoding_errors"] = encoding_errors
        kwargs["engine"] = "python"
    return pd.read_csv(source, **kwargs)


def read_df(
    path: Path,
    nrows: Optional[int] = None,
    usecols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:

    path = Path(path)

    # Excel files must be read with read_excel, never read_csv
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, nrows=nrows, usecols=usecols)
        return _whitespace_only_to_na(df)

    # SPSS .sav files — read via pandas.read_spss (backed by pyreadstat).
    # read_spss has no nrows/usecols args, so we slice after loading.
    if path.suffix.lower() == ".sav":
        df = pd.read_spss(str(path))
        if usecols is not None:
            wanted = set(usecols)
            df = df[[c for c in df.columns if c in wanted]]
        if nrows is not None:
            df = df.head(nrows)
        return _whitespace_only_to_na(df)

    with open(path, "rb") as f:
        head_bytes = f.read(_SNIFF_BYTES)

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    for enc in encodings:
        try:
            sample = _decode_head(head_bytes, enc)
            df = _read_csv_with_autosep(
                path,
                encoding=enc,
                sample=sample,
                nrows=nrows,
                usecols=usecols,
            )
            return _whitespace_only_to_na(_validate_parse_result(df))
        except AmbiguousCsvSeparatorError:
            raise
        except Exception:
            pass

    # Fallback: keep going even if characters are weird
    try:
        sample = _decode_head(head_bytes, "latin1")
        df = _read_csv_with_autosep(
            path,
            encoding="latin1",
            sample=sample,
            nrows=nrows,
            usecols=usecols,
            on_bad_lines="skip",
        )
        return _whitespace_only_to_na(_validate_parse_result(df))
    except AmbiguousCsvSeparatorError:
        raise
    except Exception:
        pass

    # Final fallback: replace invalid bytes (never crash API)
    sample = _decode_head(head_bytes, "utf-8")
    df = _read_csv_with_autosep(
        path,
        encoding="utf-8",
        sample=sample,
        nrows=nrows,
        usecols=usecols,
        on_bad_lines="skip",
        encoding_errors="replace",
    )
    return _whitespace_only_to_na(_validate_parse_result(df))


def read_csv_bytes(
    content: bytes,
    nrows: Optional[int] = None,
    usecols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Read a CSV file from raw bytes (e.g. an uploaded file) with the same
    separator-auto-detection and encoding-fallback logic as ``read_df``.
    """
    if not content:
        raise RuntimeError("Empty CSV content.")

    head_bytes = content[:_SNIFF_BYTES]
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    for enc in encodings:
        try:
            sample = _decode_head(head_bytes, enc)
            df = _read_csv_with_autosep(
                io.BytesIO(content),
                encoding=enc,
                sample=sample,
                nrows=nrows,
                usecols=usecols,
            )
            return _whitespace_only_to_na(_validate_parse_result(df))
        except AmbiguousCsvSeparatorError:
            raise
        except Exception:
            pass

    try:
        sample = _decode_head(head_bytes, "latin1")
        df = _read_csv_with_autosep(
            io.BytesIO(content),
            encoding="latin1",
            sample=sample,
            nrows=nrows,
            usecols=usecols,
            on_bad_lines="skip",
        )
        return _whitespace_only_to_na(_validate_parse_result(df))
    except AmbiguousCsvSeparatorError:
        raise
    except Exception:
        pass

    sample = _decode_head(head_bytes, "utf-8")
    df = _read_csv_with_autosep(
        io.BytesIO(content),
        encoding="utf-8",
        sample=sample,
        nrows=nrows,
        usecols=usecols,
        on_bad_lines="skip",
        encoding_errors="replace",
    )
    return _whitespace_only_to_na(_validate_parse_result(df))
