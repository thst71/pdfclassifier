import os
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from classifier.llm_classifier import LLMDataExtractor


class TestLLMDataExtractor(unittest.TestCase):

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    def setUp(self):
        self.extractor = LLMDataExtractor()
        self.mock_model = MagicMock()
        self.extractor.model = self.mock_model

    def test_init_with_api_key(self):
        extractor = LLMDataExtractor(api_key="test_api_key")
        self.assertIsNotNone(extractor.model)

    def test_init_without_api_key(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"}):
            extractor = LLMDataExtractor()
            self.assertIsNotNone(extractor.model)

    def test_init_no_api_key_raises_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                LLMDataExtractor()

    def test_extract_features_empty_df(self):
        empty_df = pd.DataFrame()
        result_df = self.extractor.extract_features(empty_df)
        self.assertTrue(result_df.empty)
        self.assertListEqual(list(result_df.columns), ["key", "value", "quality"])

    def test_extract_features_api_call_success(self):
        mock_response = MagicMock()
        mock_response.text = "Document Date: 2023-01-15 (0.9)\nDocument Type: Rechnung (0.8)\nSender: Test GmbH (0.95)\nInvoice Number: 12345 (0.7)"
        self.mock_model.generate_content.return_value = mock_response
        mock_response.resolve.return_value = None
        test_df = pd.DataFrame({"text": ["some", "text"]})
        result_df = self.extractor.extract_features(test_df)
        self.assertFalse(result_df.empty)
        self.assertEqual(len(result_df), 4)
        self.assertListEqual(list(result_df.columns), ["key", "value", "quality"])
        self.assertEqual(result_df.loc[0, "key"], "Document Date")
        self.assertEqual(result_df.loc[0, "value"], "2023-01-15")
        self.assertEqual(result_df.loc[0, "quality"], 0.9)
        self.assertEqual(result_df.loc[1, "key"], "Document Type")
        self.assertEqual(result_df.loc[1, "value"], "Rechnung")
        self.assertEqual(result_df.loc[1, "quality"], 0.8)
        self.assertEqual(result_df.loc[2, "key"], "Sender")
        self.assertEqual(result_df.loc[2, "value"], "Test-GmbH")
        self.assertEqual(result_df.loc[2, "quality"], 0.95)
        self.assertEqual(result_df.loc[3, "key"], "Invoice Number")
        self.assertEqual(result_df.loc[3, "value"], "12345")
        self.assertEqual(result_df.loc[3, "quality"], 0.7)

    def test_extract_features_api_call_failure(self):
        self.mock_model.generate_content.side_effect = Exception("API Error")
        test_df = pd.DataFrame({"text": ["some", "text"]})
        result_df = self.extractor.extract_features(test_df)
        self.assertTrue(result_df.empty)
        self.assertListEqual(list(result_df.columns), ["key", "value", "quality"])

    def test_parse_response_invalid_response(self):
        invalid_response = "This is not a valid response"
        result_df = self.extractor._parse_response(invalid_response)
        self.assertFalse(result_df.empty)
        self.assertEqual(len(result_df), 4)
        self.assertListEqual(list(result_df.columns), ["key", "value", "quality"])
        self.assertListEqual(list(result_df["quality"]), [0.1, 0.1, 0.1, 0.1])

    def test_parse_response_valid_response(self):
        valid_response = "Document Date: 2023-01-15 (0.9)\nDocument Type: Rechnung (0.8)\nSender: Test GmbH (0.95)\nInvoice Number: 12345 (0.7)"
        result_df = self.extractor._parse_response(valid_response)
        self.assertFalse(result_df.empty)
        self.assertEqual(len(result_df), 4)
        self.assertListEqual(list(result_df.columns), ["key", "value", "quality"])
        self.assertEqual(result_df.loc[0, "key"], "Document Date")
        self.assertEqual(result_df.loc[0, "value"], "2023-01-15")
        self.assertEqual(result_df.loc[0, "quality"], 0.9)
        self.assertEqual(result_df.loc[1, "key"], "Document Type")
        self.assertEqual(result_df.loc[1, "value"], "Rechnung")
        self.assertEqual(result_df.loc[1, "quality"], 0.8)
        self.assertEqual(result_df.loc[2, "key"], "Sender")
        self.assertEqual(result_df.loc[2, "value"], "Test-GmbH")
        self.assertEqual(result_df.loc[2, "quality"], 0.95)
        self.assertEqual(result_df.loc[3, "key"], "Invoice Number")
        self.assertEqual(result_df.loc[3, "value"], "12345")
        self.assertEqual(result_df.loc[3, "quality"], 0.7)

    def test_parse_response_invalid_quality(self):
        valid_response = "Document Date: 2023-01-15 (abc)\nDocument Type: Rechnung (0.8)\nSender: Test GmbH (0.95)\nInvoice Number: 12345 (0.7)"
        result_df = self.extractor._parse_response(valid_response)
        self.assertFalse(result_df.empty)
        self.assertEqual(len(result_df), 4)
        self.assertListEqual(list(result_df.columns), ["key", "value", "quality"])
        self.assertEqual(result_df.loc[0, "key"], "Document Date")
        self.assertEqual(result_df.loc[0, "value"], "")
        self.assertEqual(result_df.loc[0, "quality"], 0.1)

    def test_parse_response_invalid_key(self):
        valid_response = "Document Date: 2023-01-15 (0.9)\nDocument Type: Rechnung (0.8)\nSender: Test GmbH (0.95)\nInvoice Number: 12345 (0.7)\nInvalid Key: Invalid Value (0.5)"
        result_df = self.extractor._parse_response(valid_response)
        self.assertFalse(result_df.empty)
        self.assertEqual(len(result_df), 4)
        self.assertListEqual(list(result_df.columns), ["key", "value", "quality"])

    def test_parse_response_invalid_value(self):
        valid_response = "Document Date: 2023-01-15 (0.9)\nDocument Type: Rechnu(ng (0.8)\nSender: Test G mbH (0.95)\nInvoice Number: 12345 (0.7)"
        result_df = self.extractor._parse_response(valid_response)
        self.assertFalse(result_df.empty)
        self.assertEqual(len(result_df), 4)
        self.assertListEqual(list(result_df.columns), ["key", "value", "quality"])
        self.assertEqual(result_df.loc[1, "value"], "")
        self.assertEqual(result_df.loc[2, "value"], "")

    def test_create_prompt(self):
        test_df = pd.DataFrame({"text": ["some", "text"]})
        prompt = self.extractor._create_prompt(test_df)
        self.assertIn("You are a document processing expert", prompt)
        self.assertIn("Here is the document text:", prompt)
        self.assertIn("some", prompt)
        self.assertIn("text", prompt)


if __name__ == '__main__':
    unittest.main()
