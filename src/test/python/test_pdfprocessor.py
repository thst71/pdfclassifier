import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pandas as pd

from classifier.pdfprocessor import PdfProcessor, PdfData


class TestPdfData(unittest.TestCase):
    def setUp(self):
        self.temp_base_dir = Path(tempfile.mkdtemp())
        self.pdf_file = self.temp_base_dir / "test.pdf"
        with open(self.pdf_file, "w") as f:
            f.write("dummy pdf content")
        self.pdf_data = PdfData(self.pdf_file, self.temp_base_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_base_dir)

    def test_init(self):
        self.assertTrue(self.pdf_data.temp_dir.exists())
        self.assertEqual(self.pdf_data.pdf_file, self.pdf_file)
        self.assertEqual(self.pdf_data.image_files, [])
        self.assertEqual(self.pdf_data.data_files, [])
        self.assertEqual(self.pdf_data.data, [])

    def test_init_existing_temp_dir(self):
        # Create an existing temp dir
        self.pdf_data.temp_dir.mkdir(parents=True, exist_ok=True)
        # Create a file in the temp dir to check if it gets removed
        (self.pdf_data.temp_dir / "dummy.txt").touch()
        # Create a new PdfData object
        pdf_data = PdfData(self.pdf_file, self.temp_base_dir, force=True)
        self.assertTrue(pdf_data.temp_dir.exists())
        self.assertFalse((pdf_data.temp_dir / "dummy.txt").exists())

    def test_cleanup(self):
        self.pdf_data.temp_dir.mkdir(parents=True, exist_ok=True)
        (self.pdf_data.temp_dir / "dummy.txt").touch()
        self.pdf_data.cleanup()
        self.assertFalse(self.pdf_data.temp_dir.exists())

    @patch("classifier.pdfprocessor.convert_from_path")
    def test_extract_images(self, mock_convert_from_path):
        mock_image = MagicMock()
        mock_convert_from_path.return_value = [mock_image]
        self.pdf_data.extract_images()
        self.assertEqual(len(self.pdf_data.image_files), 1)
        mock_image.save.assert_called_once()

    @patch("classifier.pdfprocessor.convert_from_path")
    def test_extract_images_error(self, mock_convert_from_path):
        mock_convert_from_path.side_effect = Exception("Test Error")
        self.pdf_data.extract_images()
        self.assertEqual(len(self.pdf_data.image_files), 0)

    @patch("classifier.pdfprocessor.pytesseract.image_to_data")
    @patch("classifier.pdfprocessor.Image.open")
    def test_extract_text(self, mock_image_open, mock_image_to_data):
        mock_image = MagicMock()
        mock_image_open.return_value.__enter__.return_value = mock_image
        mock_image_to_data.return_value = pd.DataFrame({"text": ["test"]})
        self.pdf_data.image_files = [self.pdf_data.temp_dir / "page_1.png"]
        self.pdf_data.extract_text()
        self.assertEqual(len(self.pdf_data.data_files), 1)
        self.assertEqual(len(self.pdf_data.data), 1)
        mock_image_to_data.assert_called_once()

    @patch("classifier.pdfprocessor.pytesseract.image_to_data")
    @patch("classifier.pdfprocessor.Image.open")
    def test_extract_text_no_images(self, mock_image_open, mock_image_to_data):
        self.pdf_data.extract_text()
        self.assertEqual(len(self.pdf_data.data_files), 0)
        self.assertEqual(len(self.pdf_data.data), 0)
        mock_image_to_data.assert_not_called()

    @patch("classifier.pdfprocessor.pytesseract.image_to_data")
    @patch("classifier.pdfprocessor.Image.open")
    def test_extract_text_error(self, mock_image_open, mock_image_to_data):
        mock_image = MagicMock()
        mock_image_open.return_value.__enter__.return_value = mock_image
        mock_image_to_data.side_effect = Exception("Test Error")
        self.pdf_data.image_files = [self.pdf_data.temp_dir / "page_1.png"]
        self.pdf_data.extract_text()
        self.assertEqual(len(self.pdf_data.data_files), 0)
        self.assertEqual(len(self.pdf_data.data), 0)

    @patch("classifier.pdfprocessor.pytesseract.image_to_data")
    @patch("classifier.pdfprocessor.Image.open")
    def test_extract_text_existing_data(self, mock_image_open, mock_image_to_data):
        mock_image = MagicMock()
        mock_image_open.return_value.__enter__.return_value = mock_image
        mock_image_to_data.return_value = pd.DataFrame({"text": ["test"]})
        self.pdf_data.image_files = [self.pdf_data.temp_dir / "page_1.png"]
        self.pdf_data.extract_text()
        self.assertEqual(len(self.pdf_data.data_files), 1)
        self.assertEqual(len(self.pdf_data.data), 1)
        mock_image_to_data.assert_called_once()
        mock_image_to_data.reset_mock()
        self.pdf_data.extract_text()
        mock_image_to_data.assert_not_called()


class TestPdfProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_base_dir = Path(tempfile.mkdtemp())
        self.pdf_processor = PdfProcessor(self.temp_base_dir)
        self.pdf_folder = self.temp_base_dir / "pdf_folder"
        self.pdf_folder.mkdir()
        self.pdf_file = self.pdf_folder / "test.pdf"
        with open(self.pdf_file, "w") as f:
            f.write("dummy pdf content")

    def tearDown(self):
        shutil.rmtree(self.temp_base_dir, ignore_errors=True)

    @patch("classifier.pdfprocessor.PdfData")
    def test_process_pdf(self, mock_data):
        mock_data.return_value = mm_pdf_data = MagicMock()
        mm_pdf_data.data_files = ["one"]
        mm_pdf_data.extract_text = MagicMock()
        mm_pdf_data.extract_images = MagicMock()

        pdf_data = self.pdf_processor.process_pdf(self.pdf_file)
        self.assertIsNotNone(pdf_data)
        mm_pdf_data.extract_images.assert_called_once()
        mm_pdf_data.extract_text.assert_called_once()

    def test_process_pdf_not_found(self):
        pdf_data = self.pdf_processor.process_pdf(self.pdf_folder / "not_found.pdf")
        self.assertIsNone(pdf_data)

    @patch("classifier.pdfprocessor.PdfProcessor.process_pdf")
    def test_process_pdfs(self, mock_process_pdf):
        mock_process_pdf.return_value = MagicMock()
        pdf_data_list = self.pdf_processor.process_pdfs(self.pdf_folder)
        self.assertEqual(len(pdf_data_list), 1)
        mock_process_pdf.assert_called_once()

    def test_process_pdfs_no_pdf(self):
        pdf_data_list = self.pdf_processor.process_pdfs(self.temp_base_dir)
        self.assertEqual(len(pdf_data_list), 0)

    def test_cleanup(self):
        self.pdf_processor.cleanup()
        self.assertFalse(self.temp_base_dir.exists())


if __name__ == "__main__":
    unittest.main()
