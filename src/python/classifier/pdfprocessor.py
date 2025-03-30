"""
This module processes pdf files in a folder into images per page and feeds the images into tesseract for OCR.

For each PDF, there will be a temp folder containing
- one image per page
- a data file with the text per page

The PDF extraction is managed with PdfData objects that contains the Paths to the image files and datafiles.

The datafiles are all written in a basefolder that is given into the constructor.
"""
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import pandas
from pytesseract.pytesseract import cleanup


class PdfData:
    pdf_file: Path
    image_files: List[Path]
    data_files: List[Path]
    temp_dir: Path
    data: List[pandas.DataFrame] = []

    def __init__(self, pdf_file: Path, temp_base_dir: Path, force:bool=False):
        self.pdf_file = pdf_file
        self.image_files = []
        self.data_files = []
        self.data = []
        self.force = force
        self.temp_dir = temp_base_dir / self.pdf_file.stem
        self._mtime_of_pdf = os.path.getmtime(self.pdf_file)
        if self.temp_dir.exists() :
            # check if the tmp folder is older than the pdf_file or force is true
            if os.path.getmtime(self.temp_dir) < self._mtime_of_pdf or force:
                logging.warning(f"removing existing workfolder {self.temp_dir}")
                self.cleanup()
        else:
            self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _is_overwrite(self, check_file: Path):
        """
        Checks if a file should be overwritten based on its last modification time and the force flag.

        :param check_file: The file to check.
        :return: True if the file should be overwritten, False otherwise.
        """
        if not check_file.exists():
            return True

        return os.path.getmtime(check_file) < self._mtime_of_pdf or self.force

    def cleanup(self):
        """Removes the temporary directory and its contents."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def extract_images(self):
        """Extracts images from the PDF file."""
        try:
            logging.debug(f"Extracting _images from {self.pdf_file}")
            _images = convert_from_path(self.pdf_file, dpi=200)
            for i, _image in enumerate(_images):
                logging.debug(f"Extracting image {i+1}/{len(_images)} from {self.pdf_file}")
                _image_file: Path = self.temp_dir / f"page_{i+1}.png"
                if self._is_overwrite(_image_file):
                    _image.save(_image_file, "PNG")
                    logging.debug(f"overwrite existing image {_image_file}")
                else:
                    logging.info(f"using existing image {_image_file}")
                self.image_files.append(_image_file)
        except Exception as e:
            logging.error(f"Error extracting _images from {self.pdf_file}: {e}")

    def extract_text(self):
        """Extracts text from the images using Tesseract OCR."""
        if not self.image_files:
            logging.warning(f"No images found for {self.pdf_file}. Skipping text extraction.")
            return

        logging.debug("Extracting text from images for {self.pdf_file}")
        for image_path in self.image_files:
            try:
                data_path = self.temp_dir / f"{image_path.stem}.csv"
                if not self._is_overwrite(data_path):
                    logging.info(f"using existing data {data_path}")
                    text:pandas.DataFrame = pandas.read_csv(data_path, sep="\t")
                else:
                    with Image.open(image_path) as image:
                        logging.debug(f"Extracting text as Dataframe from {image_path}")
                        text:pandas.DataFrame = pytesseract.image_to_data(image, lang="deu", config="--dpi 200", output_type=pytesseract.Output.DATAFRAME)
                        text.to_csv(data_path, sep="\t", index=False)
                        logging.debug(f"Text extracted from image {image_path} to {data_path}")

                self.data.append(text)
                self.data_files.append(data_path)
            except Exception as e:
                logging.error(f"Error extracting text from {image_path}: {e}")


class PdfProcessor:
    def __init__(self, temp_base_dir: Path = None, force:bool = False):
        """
        :param temp_base_dir: the working folder
        :param force: if true, removes all existing files, otherwise, checks if the source file is newer than the target file and only reprocesses newer files
        """

        self.temp_base_dir = temp_base_dir or Path(tempfile.gettempdir()) / "pdf_processing"
        self.temp_base_dir.mkdir(parents=True, exist_ok=True)
        self.force = force

    def process_pdf(self, pdf_file: Path) -> PdfData | None:
        """Processes a single PDF file.
        :param force: if true, removes all existing files, otherwise, checks if the source file is newer than the target file and only reprocesses newer files
        """
        if not pdf_file.exists():
            logging.error(f"PDF file not found: {pdf_file}")
            return None

        _pdf_data = PdfData(pdf_file, self.temp_base_dir, self.force)
        _pdf_data.extract_images()
        _pdf_data.extract_text()

        return _pdf_data if len(_pdf_data.data_files) > 0 else None

    def process_pdfs(self, pdf_folder: Path) -> List[PdfData]:
        """Processes all PDF files in a folder.
        :param pdf_folder: the folder containing the PDF files
        :returns: a list of PdfData objects
        """
        _pdf_files = list(pdf_folder.glob("*.pdf"))
        _pdf_data_list: List[PdfData] = []
        for _pdf_file in _pdf_files:
            logging.info(f"Processing PDF: {_pdf_file}")
            _pdf_data = self.process_pdf(_pdf_file)
            if _pdf_data is not None:
                _pdf_data_list.append(_pdf_data)
            else:
                logging.debug(f"No PDF data found for {_pdf_file}, skipped")
        return _pdf_data_list

    def cleanup(self):
        """Removes the temporary base directory and its contents."""
        shutil.rmtree(self.temp_base_dir, ignore_errors=True)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create a PdfProcessor
    processor = PdfProcessor()

    # Process all PDFs in a folder
    pdf_folder = Path("path/to/your/pdf/folder")  # Replace with your folder
    pdf_data_list = processor.process_pdfs(pdf_folder, do_force)

    # Iterate through the results
    for pdf_data in pdf_data_list:
        logging.info(f"Processed PDF: {pdf_data.pdf_file}")
        for image_file in pdf_data.image_files:
            logging.info(f"  Image: {image_file}")
        for data_file in pdf_data.data_files:
            logging.info(f"  Data: {data_file}")
        pdf_data.cleanup()

    # Clean up the temporary base directory
    processor.cleanup()
