from __future__ import annotations

import csv
from re import match
from pathlib import Path
from typing import Callable

"""
This module provides classes and functions for loading and managing classification data from a CSV file.
"""

class FileData :
    docdate : str
    doctype : str
    sendername : str
    docid : str
    receivername : str
    dateoffile : str
    extension : str
    is_complete : bool
    """
    Represents a single row of classification data.

    Attributes:
        docdate (str): The document date.
        doctype (str): The document type.
        sendername (str): The sender's name.
        docid (str): The document ID.
        receivername (str): The receiver's name.
        dateoffile (str): The date of the file.
        extension (str): The file extension.
    """
    def init_from_dict(self, id: str, row: dict[str, str]) -> tuple[FileData, bool]:
        self.id = id
        for key in ['docdate', 'doctype', 'sendername', 'docid', 'receivername', 'dateoffile', 'extension']:
            setattr(self, key, row.get(key))

        self.is_complete = all(x is not None for x in [self.docdate, self.doctype, self.sendername, self.docid, self.receivername, self.dateoffile, self.extension])
        return self,self.is_complete

    def get_sanitized(self, sanitize_func:Callable[[str], str]) -> FileData:
        return FileData().init_from_dict(self.id, {
            k: sanitize_func(v) if v is not None else None
            for k, v in self.__dict__.items()
            if k not in ['is_complete', 'id']
        })[0]

    def __init__(self):
        self.is_complete = False
        """
        Initializes a new instance of the FileData class.
        """

    def init_from_features(self, features) -> FileData:
        # if the quality score is <=0.9, should append the score in the value field in square brackets
        for key in ["Document Date", "Document Type", "Sender", "Invoice Number"]:
            if features.loc[key, "quality"] < 0.9:
                features.loc[key, "value"] = f"{features.loc[key, 'value']}[{features.loc[key, 'quality']}]"

        self.id = features.loc["id", "value"]
        self.docdate = features.loc["Document Date", "value"] or "unknown"
        self.doctype = features.loc["Document Type", "value"] or "unknown"
        self.sendername = features.loc["Sender", "value"] or "unknown"
        self.docid = features.loc["Invoice Number", "value"] or "unknown"
        self.receivername = ""
        self.dateoffile = ""
        self.extension = "pdf"

        self.is_complete = all(x is not None for x in [self.docdate, self.doctype, self.sendername, self.docid, self.receivername, self.dateoffile, self.extension])
        return self


def load_classification_data(csv_file: Path) -> dict[str, FileData]:
    """
    Loads classification data from a CSV file.

    Args:
        csv_file: The path to the CSV file.

    Returns:
        A dictionary where keys are PDF filenames and values are dictionaries
        containing the classification data.
    """
    data = {}
    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            row_obj, is_complete = FileData().init_from_dict(row['scanfile'], row)
            if is_complete:
                data[row_obj.id] = row_obj
            else:
                print(f"{row_obj.id} is not complete")
    return data
