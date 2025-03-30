import re
from pathlib import Path

from classifier.data import FileData

"""replacement character for invalid characters in filenames"""
REPLACE_CHAR = "-"

"""
This module renames PDF documents based on metadata extracted from a CSV file.
"""

def find_pdf_in(folder: Path) -> list[Path]:
    """
    Finds all PDF files within a specified folder.

    Args:
        folder: The path to the folder to search within.

    Returns:
        A list of Path objects, each representing a PDF file found in the folder.
        Returns an empty list if no PDF files are found.
    """
    return list(folder.glob("*.pdf"))


def is_date_string(candidate) -> str:
    """checks if candidate is a date string, returns true if it is, false if not"""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
        return True
    if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", candidate):
        return True
    if re.fullmatch(r"\d{2}\.\d{2}\.\d{2}", candidate):
        return True
    if re.fullmatch(r"\d{4}\d{2}\d{2}", candidate):
        return True
    return False

def sortable_date(datestr) -> str:
    """    Converts a date string to a sortable format (YYYY-MM-DD).

    for xx-xx-xx the code tries to identify if the date is yy-mm-dd, dd-mm-yy, mm-dd-yy, yy-dd-mm

    the heuristics are:
    - if month > 12 then month and day are switched
    - if day > 31 then year and day are switched

    Args:
        datestr: The date string to convert.

    Returns:
        The date string in YYYY-MM-DD format, or the original string if it's not a recognized date format.
    """

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", datestr):
        return datestr
    if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", datestr):
        return datestr[6:10] + "-" + datestr[3:5] + "-" + datestr[0:2]

    if re.fullmatch(r"\d{2}\.\d{2}\.\d{2}", datestr):
        (year, month, day) = datestr.split(".")
        if int(month) > 12 : (month, day) = (day, month)
        if int(day) > 31 : (year, day) = (day, year)
        return f"20{year}-{month}-{day}"

    if re.fullmatch(r"\d{4}\d{2}\d{2}", datestr):
        return datestr[0:4] + "-" + datestr[4:6] + "-" + datestr[6:8]
    return datestr

def sanitize_string_for_filename(raw_filename: str) -> str:
    """
    Sanitizes a filename by replacing invalid characters with REPLACE_CHARs.

    Args:
        raw_filename: The filename to sanitize.

    Returns:
        The sanitized filename.
    """
    # check that raw_filename is actually a string or make it a string
    if not isinstance(raw_filename, str):
        raw_filename = str(raw_filename)

    sanitized = re.sub(r'[\\/*?:"<>|]', REPLACE_CHAR, raw_filename)
    # Remove leading/trailing spaces and dots and replacement chars
    sanitized = sanitized.strip(f" .{REPLACE_CHAR}")
    # Replace multiple REPLACE_CHARs with a single REPLACE_CHAR
    sanitized = re.sub(rf"{REPLACE_CHAR}+", REPLACE_CHAR, sanitized)

    sanitized = sortable_date(sanitized) if is_date_string(sanitized) else sanitized

    return sanitized

def classify_pdf(data: FileData, out_path : Path = None) -> Path | None:
    """
    Classifies a PDF file based on the provided classification data.

    Args:
        data: The classification data for the PDF.
        out_path: The base path where the classified PDF should be moved to.

    Returns:
        The new path of the classified PDF, or None if the classification data is incomplete.
    """
    if not data.is_complete:
        return None

    new_filename = f"{data.docdate}_{data.doctype}_{data.sendername}_{data.docid}_{data.receivername}_{data.dateoffile}.{data.extension}"
    sanitized_filename = sanitize_string_for_filename(new_filename)
    new_path = out_path / sanitized_filename if out_path is not None else Path(data.id).parent / sanitized_filename
    return new_path

def sanitize_filename_data(raw_filename_data: dict[str, FileData]) -> dict[str, FileData]:
    """    Sanitizes all values in the classification data.

    Args:
        raw_filename_data: The raw classification data.

    Returns:
        The sanitized classification data.
    """
    sanitized_data: dict[str, FileData] = {}
    for key, value in raw_filename_data.items():
        sanitized_data[key] = value.get_sanitized(sanitize_string_for_filename)

    return sanitized_data

