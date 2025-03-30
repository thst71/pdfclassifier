import unittest
from unittest.mock import patch
from pathlib import Path
from classifier.data import FileData, load_classification_data
import pandas as pd


class TestFileData(unittest.TestCase):

    def test_init_from_dict_complete(self):
        row = {
            'docdate': '2023-01-01',
            'doctype': 'invoice',
            'sendername': 'Test Sender',
            'docid': '123',
            'receivername': 'Test Receiver',
            'dateoffile': '2023-01-02',
            'extension': 'pdf'
        }
        file_data, is_complete = FileData().init_from_dict('test.pdf', row)
        self.assertTrue(is_complete)
        self.assertEqual(file_data.docdate, '2023-01-01')
        self.assertEqual(file_data.doctype, 'invoice')
        self.assertEqual(file_data.sendername, 'Test Sender')
        self.assertEqual(file_data.docid, '123')
        self.assertEqual(file_data.receivername, 'Test Receiver')
        self.assertEqual(file_data.dateoffile, '2023-01-02')
        self.assertEqual(file_data.extension, 'pdf')
        self.assertEqual(file_data.id, 'test.pdf')

    def test_init_from_dict_incomplete(self):
        row = {
            'docdate': '2023-01-01',
            'doctype': 'invoice',
            'sendername': 'Test Sender',
            'docid': '123',
            'receivername': None,
            'dateoffile': '2023-01-02',
            'extension': 'pdf'
        }
        file_data, is_complete = FileData().init_from_dict('test.pdf', row)
        self.assertFalse(is_complete)
        self.assertEqual(file_data.docdate, '2023-01-01')
        self.assertEqual(file_data.doctype, 'invoice')
        self.assertEqual(file_data.sendername, 'Test Sender')
        self.assertEqual(file_data.docid, '123')
        self.assertIsNone(file_data.receivername)
        self.assertEqual(file_data.dateoffile, '2023-01-02')
        self.assertEqual(file_data.extension, 'pdf')
        self.assertEqual(file_data.id, 'test.pdf')

    def test_get_sanitized(self):
        row = {
            'docdate': '2023-01-01',
            'doctype': 'invoice',
            'sendername': 'Test Sender',
            'docid': '123',
            'receivername': 'Test Receiver',
            'dateoffile': '2023-01-02',
            'extension': 'pdf'
        }
        file_data, _ = FileData().init_from_dict('test.pdf', row)
        sanitized_data = file_data.get_sanitized(lambda x: x.upper() if x is not None else None)
        self.assertEqual(sanitized_data.docdate, '2023-01-01')
        self.assertEqual(sanitized_data.doctype, 'INVOICE')
        self.assertEqual(sanitized_data.sendername, 'TEST SENDER')
        self.assertEqual(sanitized_data.docid, '123')
        self.assertEqual(sanitized_data.receivername, 'TEST RECEIVER')
        self.assertEqual(sanitized_data.dateoffile, '2023-01-02')
        self.assertEqual(sanitized_data.extension, 'PDF')
        self.assertEqual(sanitized_data.id, 'test.pdf')

    def test_init_from_features(self):
        features = pd.DataFrame({
            "key": ["id", "Document Date", "Document Type", "Sender", "Invoice Number"],
            "value": ["test.pdf", "2023-01-01", "invoice", "Test Sender", "123"],
            "quality": [1.0, 0.9, 0.8, 0.95, 0.7]
        })
        features.set_index("key", inplace=True)
        file_data = FileData().init_from_features(features)
        self.assertEqual(file_data.id, 'test.pdf')
        self.assertEqual(file_data.docdate, '2023-01-01')
        self.assertEqual(file_data.doctype, 'invoice[0.8]')
        self.assertEqual(file_data.sendername, 'Test Sender')
        self.assertEqual(file_data.docid, '123[0.7]')
        self.assertEqual(file_data.receivername, '')
        self.assertEqual(file_data.dateoffile, '')
        self.assertEqual(file_data.extension, 'pdf')

    def test_init_from_features_missing_values(self):
        features = pd.DataFrame({
            "key": ["id", "Document Date", "Document Type", "Sender", "Invoice Number"],
            "value": ["test.pdf", None, None, None, None],
            "quality": [1.0, 0.9, 0.8, 0.95, 0.7]
        })
        features.set_index("key", inplace=True)
        file_data = FileData().init_from_features(features)
        self.assertEqual(file_data.id, 'test.pdf')
        self.assertEqual(file_data.docdate, 'unknown')
        self.assertEqual(file_data.doctype, 'unknown[0.8]')
        self.assertEqual(file_data.sendername, 'unknown')
        self.assertEqual(file_data.docid, 'unknown[0.7]')
        self.assertEqual(file_data.receivername, '')
        self.assertEqual(file_data.dateoffile, '')
        self.assertEqual(file_data.extension, 'pdf')


class TestLoadClassificationData(unittest.TestCase):

    @patch("builtins.open", new_callable=unittest.mock.mock_open,
           read_data="scanfile,docdate,doctype,sendername,docid,receivername,dateoffile,extension\ntest.pdf,2023-01-01,invoice,Test Sender,123,Test Receiver,2023-01-02,pdf")
    def test_load_classification_data_complete(self, mock_open):
        data = load_classification_data(Path("test.csv"))
        self.assertIn("test.pdf", data)
        self.assertEqual(data["test.pdf"].docdate, "2023-01-01")
        self.assertEqual(data["test.pdf"].doctype, "invoice")
        self.assertEqual(data["test.pdf"].sendername, "Test Sender")
        self.assertEqual(data["test.pdf"].docid, "123")
        self.assertEqual(data["test.pdf"].receivername, "Test Receiver")
        self.assertEqual(data["test.pdf"].dateoffile, "2023-01-02")
        self.assertEqual(data["test.pdf"].extension, "pdf")

    @patch("builtins.open", new_callable=unittest.mock.mock_open,
           read_data="scanfile,docdate,doctype,sendername,docid,receivername,dateoffile,extension\ntest.pdf,2023-01-01,invoice,Test Sender,123,,2023-01-02,pdf")
    def test_load_classification_data_incomplete(self, mock_open):
        data = load_classification_data(Path("test.csv"))
        self.assertNotIn("test.pdf", data)


if __name__ == '__main__':
    unittest.main()
