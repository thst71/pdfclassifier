import logging
import os
import re
import time

import google.generativeai as genai
import pandas as pd

# time between two calls in seconds
THROTTLE_SECONDS = 2.0

class LLMDataExtractor:
    """
    A class to extract key data elements from a Tesseract OCR DataFrame using a Large Language Model (LLM).
    """

    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash"):
        """
        Initializes the LLMDataExtractor with the specified API key and model name.

        Args:
            api_key: The API key for the LLM service. If None, it will try to get it from the environment variable "GOOGLE_API_KEY".
            model_name: The name of the LLM model to use (default: "gemini-pro").
        """
        self.last_call = 0
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError("No API key provided and GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def extract_features(self, tesseract_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts key features (document date, document type, sender, invoice number) from a Tesseract DataFrame using an LLM.

        Args:
            tesseract_df: The Tesseract DataFrame containing OCR output.

        Returns:
            A pandas DataFrame with extracted features (key, value, quality).
        """
        if time.time() - self.last_call < THROTTLE_SECONDS:
            sleep_time = THROTTLE_SECONDS - (time.time() - self.last_call)
            logging.info(f"API call throttle - sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)

        if tesseract_df.empty:
            return pd.DataFrame(columns=["key", "value", "quality"])

        # Prepare the prompt for the LLM
        prompt = self._create_prompt(tesseract_df)

        # Get the response from the LLM
        try:
            self.last_call = time.time()
            response = self.model.generate_content(prompt)
            response.resolve()
        except Exception as e:
            print(f"Error during LLM call: {e}")
            return pd.DataFrame(columns=["key", "value", "quality"])

        # Parse the response and create the output DataFrame
        return self._parse_response(response.text)

    def _create_prompt(self, tesseract_df: pd.DataFrame) -> str:
        """
        Creates the prompt for the LLM based on the Tesseract DataFrame.

        Args:
            tesseract_df: The Tesseract DataFrame.

        Returns:
            The prompt string.
        """
        # Extract relevant columns from the DataFrame
        text_content = "\n".join(tesseract_df["text"].dropna().astype(str).tolist())
        prompt = f"""
        You are a document processing expert. Your task is to extract key information from a document.
        The document is represented as tabular text from a tesseract ocr in pandas dataframe format. The table contains information on page, line, word and position on page. Use the given locations to improve data extraction.
        Extract the following information:
        - Document Date: The date of the document must be in the format "yyyy-mm-dd".
        - Document Type: The type of the document. If it is an invoice, the type is "rechnung", if it is an account statement use "ktoauszg", otherwise "Other".
        - Sender: The name of the sender of the document.
        - Invoice Number: If the document type is "Other", leave blank. If the document is an invoice, the invoice number. If it is an account statement, first determine the type of account, for a bank account use the IBAN, for a credit card or Corporate card statement use the credit card number. Prefix the month of the statement as number (01,...,12) followed by a dash to the result. Make sure you do not use the IBAN of the bank as the account number, the bank IBAN is usually in the footer of the document, the account number is in the upper half of the document. Make sure to not use a random sequence of numbers found on the document as account number, IBAN or credit card numbers. IBANs and credit card numbers never start with a 0 and have no separators other than spaces or dashes, or no separators at all. Credit card numbers and corporate card numbers are exactly 16 digits long and in 4 groups of 4 digits each. A statement for a credit card or corporate card contains the words "credit card" or "corporate card".
        
        Provide the extracted information in the following plain-text, line-based format:
        key: value (quality)
        key: value (quality)
        ...

        where "key" is the extracted information, "value" is the corresponding value, and "quality" is the confidence level of the extraction.
        Use plain text in UTF-8 encoding. Do not use another structure for the output as described above.
        Each key-value-quality tuple is on one line. 
        Use for key only the keywords "Document Date", "Document Type", "Sender", and "Invoice Number". 
        Do not add other keys to the result. 
        Use all defined keys in every result.
        Replace the characters " " and "(" and ")" with "-" in the value. 

        The quality is a number between 0 and 1, where 1 means that you are very sure about the value and 0 means that you are not sure at all.

        Here is the document text:
        {text_content}
        """
        return prompt

    def _parse_response(self, response_text: str) -> pd.DataFrame:
        """
        Parses the LLM's response and creates the output DataFrame.

        Args:
            response_text: The text response from the LLM.

        Returns:
            A pandas DataFrame with extracted features (key, value, quality).
        """
        data = []
        lines = response_text.strip().split("\n")
        valid_keys = ["Document Date", "Document Type", "Sender", "Invoice Number"]

        pattern = r"(Document Date|Document Type|Sender|Invoice Number)[:]\s*([^(]+)\s*\(([0-9.]+)\)"

        for line in lines:
            match = re.match(pattern, line)
            if match is not None:
                key = match.group(1)
                value = match.group(2).strip()
                value = value.replace(" ", "-").replace("(", "-").replace(")", "-")
                quality = float(match.group(3))
                data.append({"key": key, "value": value, "quality": quality})
            else:
                logging.debug(f"skipping line: {line}")

        if len(data) != 4:
            logging.warning(f"received invalid response: {response_text}")
            return pd.DataFrame(
                    {"key": valid_keys, "value": [""] * len(valid_keys), "quality": [0.1] * len(valid_keys)},
                    columns=["key", "value", "quality"])

        df = pd.DataFrame(data)
        return df