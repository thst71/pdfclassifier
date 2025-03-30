import shutil
import tempfile
import unittest
from pathlib import Path

from classifier.renamer import (
    find_pdf_in,
    is_date_string,
    sortable_date,
    sanitize_string_for_filename,
    classify_pdf,
    sanitize_filename_data,
    REPLACE_CHAR
)
from classifier.data import FileData

class TestRenamer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_find_pdf_in(self):
        # Create some dummy files
        (self.temp_dir / "test1.pdf").touch()
        (self.temp_dir / "test2.pdf").touch()
        (self.temp_dir / "test3.txt").touch()

        pdf_files = find_pdf_in(self.temp_dir)
        self.assertEqual(len(pdf_files), 2)
        self.assertTrue(all(f.suffix == ".pdf" for f in pdf_files))

    def test_find_pdf_in_empty(self):
        pdf_files = find_pdf_in(self.temp_dir)
        self.assertEqual(len(pdf_files), 0)

    def test_is_date_string(self):
        self.assertTrue(is_date_string("2023-10-27"))
        self.assertTrue(is_date_string("27.10.2023"))
        self.assertTrue(is_date_string("27.10.23"))
        self.assertTrue(is_date_string("20231027"))
        self.assertFalse(is_date_string("not a date"))
        self.assertFalse(is_date_string("2023-10-"))
        self.assertFalse(is_date_string("2023-10-277"))

    def test_sortable_date(self):
        self.assertEqual("2023-10-27", sortable_date("2023-10-27"))
        self.assertEqual("2023-10-27", sortable_date("27.10.2023"))
        self.assertEqual("2027-10-23", sortable_date("27.10.23"))
        self.assertEqual("2023-10-27", sortable_date("20231027"))
        self.assertEqual("2010-23-27", sortable_date("10.27.23"))
        self.assertEqual("2027-10-23", sortable_date("27.10.23"))
        self.assertEqual("2023-10-27", sortable_date("23.10.27"))
        self.assertEqual("2023-10-27", sortable_date("23.27.10"))
        self.assertEqual("not a date", sortable_date("not a date"))

    def test_sanitize_string_for_filename(self):
        self.assertEqual("test-file.pdf", sanitize_string_for_filename("test/file.pdf"))
        self.assertEqual("test-file.pdf", sanitize_string_for_filename("test\\file.pdf"))
        self.assertEqual("test-file.pdf", sanitize_string_for_filename("test*file.pdf"))
        self.assertEqual("test-file.pdf", sanitize_string_for_filename("test?file.pdf"))
        self.assertEqual("test-file.pdf", sanitize_string_for_filename("test:file.pdf"))
        self.assertEqual("test-file.pdf", sanitize_string_for_filename("test\"file.pdf"))
        self.assertEqual("test-file.pdf", sanitize_string_for_filename("test<file.pdf"))
        self.assertEqual("test-file.pdf", sanitize_string_for_filename("test>file.pdf"))
        self.assertEqual("test-file.pdf", sanitize_string_for_filename("test|file.pdf"))
        self.assertEqual("test  file.pdf", sanitize_string_for_filename("test  file.pdf"))
        self.assertEqual("test..file.pdf", sanitize_string_for_filename("test..file.pdf"))
        self.assertEqual("test-file.pdf", sanitize_string_for_filename("test--file.pdf"))
        self.assertEqual("test file.pdf", sanitize_string_for_filename(" test file.pdf "))
        self.assertEqual("2023-10-27", sanitize_string_for_filename("2023-10-27"))
        self.assertEqual("2023-10-27", sanitize_string_for_filename("27.10.2023"))
        self.assertEqual("2027-10-23", sanitize_string_for_filename("27.10.23"))
        self.assertEqual("2033-10-27", sanitize_string_for_filename("27.10.33"))
        self.assertEqual("2025-10-27", sanitize_string_for_filename("25.27.10"))
        self.assertEqual("2010-33-27", sanitize_string_for_filename("10.27.33")) # cannot interpret this correctly
        self.assertEqual("2023-10-27", sanitize_string_for_filename("20231027"))
        self.assertEqual("123", sanitize_string_for_filename(123))

    def test_classify_pdf(self):
        # Create a dummy FileData object
        file_data = FileData()
        file_data.id = str(self.temp_dir / "test.pdf")
        file_data.docdate = "2023-10-27"
        file_data.doctype = "invoice"
        file_data.sendername = "Test Sender"
        file_data.docid = "123"
        file_data.receivername = "Test Receiver"
        file_data.dateoffile = "2023-10-28"
        file_data.extension = "pdf"
        file_data.is_complete = True

        new_path = classify_pdf(file_data, self.temp_dir)
        self.assertEqual(self.temp_dir / "2023-10-27_invoice_Test Sender_123_Test Receiver_2023-10-28.pdf", new_path)

    def test_classify_pdf_incomplete(self):
        # Create a dummy FileData object
        file_data = FileData()
        file_data.id = str(self.temp_dir / "test.pdf")
        file_data.docdate = "2023-10-27"
        file_data.doctype = "invoice"
        file_data.sendername = "Test Sender"
        file_data.docid = "123"
        file_data.receivername = None
        file_data.dateoffile = "2023-10-28"
        file_data.extension = "pdf"
        file_data.is_complete = False

        new_path = classify_pdf(file_data, self.temp_dir)
        self.assertIsNone(new_path)

    def test_classify_pdf_no_outpath(self):
        # Create a dummy FileData object
        file_data = FileData()
        file_data.id = str(self.temp_dir / "test.pdf")
        file_data.docdate = "2023-10-27"
        file_data.doctype = "invoice"
        file_data.sendername = "Test Sender"
        file_data.docid = "123"
        file_data.receivername = "Test Receiver"
        file_data.dateoffile = "2023-10-28"
        file_data.extension = "pdf"
        file_data.is_complete = True
        (self.temp_dir / "test.pdf").touch()

        new_path = classify_pdf(file_data)
        self.assertEqual(self.temp_dir / "2023-10-27_invoice_Test Sender_123_Test Receiver_2023-10-28.pdf", new_path)

    def test_sanitize_filename_data(self):
        # Create some dummy FileData objects
        file_data1 = FileData()
        file_data1.id = "test1.pdf"
        file_data1.docdate = "2023-10-27"
        file_data1.doctype = "invoice"
        file_data1.sendername = "Test/Sender"
        file_data1.docid = "123"
        file_data1.receivername = "Test Receiver"
        file_data1.dateoffile = "2023-10-28"
        file_data1.extension = "pdf"
        file_data1.is_complete = True

        file_data2 = FileData()
        file_data2.id = "test2.pdf"
        file_data2.docdate = "2023-10-28"
        file_data2.doctype = "account statement"
        file_data2.sendername = "Test\\Sender"
        file_data2.docid = "456"
        file_data2.receivername = "Test Receiver"
        file_data2.dateoffile = "2023-10-29"
        file_data2.extension = "pdf"
        file_data2.is_complete = True

        raw_data = {"test1.pdf": file_data1, "test2.pdf": file_data2}
        sanitized_data = sanitize_filename_data(raw_data)

        self.assertEqual("2023-10-27", sanitized_data["test1.pdf"].docdate)
        self.assertEqual("Test-Sender", sanitized_data["test1.pdf"].sendername)
        self.assertEqual("Test-Sender", sanitized_data["test2.pdf"].sendername)
        self.assertEqual("2023-10-28", sanitized_data["test2.pdf"].docdate)


if __name__ == "__main__":
    unittest.main()
