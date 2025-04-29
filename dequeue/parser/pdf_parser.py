import asyncio
import os.path
import subprocess
import tempfile
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
from parser.docx_to_text import DocxExtractor
from parser.imagepdf_to_text import ImageGrabber
from parser.misc import Toolkit
from backend.settings import logger





class CorruptFileException(Exception):
    """Raised when a file is found to be corrupt."""
    pass

class EmptyFileException(Exception):
    """Raised when a file is empty."""
    pass

class FileSystemException(Exception):
    """Raised when a file system operation fails."""
    pass

class ImageToTextException(Exception):
    """Raised when image to text conversion fails."""
    pass

class PDFConversionFailedException(Exception):
    """Raised when PDF conversion to another format fails."""
    pass

class PdfToImageException(Exception):
    """Raised when PDF to image conversion fails."""
    pass





class _ExtractTextFromFile:
    """Handles extraction of text from PDF files"""

    def __init__(self):
        # self.pdftotext_path = pdftotext_exe_path()
        self.pdftotext_path = r"C:\Program Files\xpdf-tools\bin64\pdftotext.exe"

    async def _convert_pdf_to_text(self, pdf_file_path: str, output_text_file_name: str) -> str:
        """Convert PDF file to text using pdftotext utility"""
        if not os.path.isfile(pdf_file_path):
            logger.error("PDF file does not exist: %s", pdf_file_path)
            return "error: File not found"

        command = [
            self.pdftotext_path,
            "-layout",
            "-table",
            "-nopgbrk",
            "-enc",
            "UTF-8",
            pdf_file_path,
            output_text_file_name,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await process.communicate()

            if process.returncode != 0:
                # Simulate the behavior of check=True
                e = subprocess.CalledProcessError(process.returncode, command)
                e.stderr = stderr_bytes.decode("utf-8")
                raise e

            return output_text_file_name

        except subprocess.CalledProcessError as e:
            error_message = e.stderr.lower()
            if "warning" in error_message:
                if "may not be a pdf file" in error_message:
                    return "error: Malformed pdf file"
                logger.warning("Warning in pdf_to_text conversion: %s", error_message)
            else:
                logger.error("Error converting pdf to text: %s", error_message)
                return "error_converting_pdf_to_text"

    async def extract_text(self, file_path: str, user_id: str) -> str:
        """Extract text from Files (pdf or docx)"""
        txt_temp = None
        is_image_pdf = False
        try:
            if file_path.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as txt_temp:
                    result = await self._convert_pdf_to_text(file_path, txt_temp.name)

                    logger.debug(f"txt_temp type: {type(txt_temp)} - txt_temp.name: {txt_temp.name}")

                    if result.startswith("error:"):
                        raise PDFConversionFailedException(f"PDF conversion failed: {result}")

                    with open(txt_temp.name, "r", encoding="utf-8") as f:
                        text = f.read()

                    if not text:
                        # for pdfs we'll see if it is really scanned pdf or a empty file or a corrupt pdf.
                        pdf_signal = Toolkit.ocr_pdf(file_path)
                        if pdf_signal == "corrupt_pdf":
                            raise CorruptFileException(f"{os.path.basename(file_path)} is corrupted.")
                        elif pdf_signal == "text_pdf":
                            raise EmptyFileException(f"{os.path.basename(file_path)} is empty.")
                        else:
                            # For scanned pdfs 1st we'll grab the images
                            logger.info(f"Scanned pdf detected, file_name: {file_path}, initialing fallback...")
                            try:
                                base64_images_list = await ImageGrabber.extract_pages_parallel(file_path)
                            except Exception as e:
                                raise PdfToImageException(f"Can't pull images from {file_path}, \nerror: {e}")
                            # Initiate image to json
                            try:
                                # text = await imagetranslator._parse_img_text(base64_images_list, user_id) # commented
                                is_image_pdf = True
                                logger.info(f"Extracted text from scanned pdf, candidate name: {text.name}")
                            except (RateLimitError, APIError, APIConnectionError, APITimeoutError, asyncio.TimeoutError):
                                raise
                            except Exception as e:
                                raise ImageToTextException(e)

            elif file_path.lower().endswith((".docx", ".doc")):
                text = await DocxExtractor.extract_docx_structure_async(file_path)
            return text, is_image_pdf

        except Exception as e:
            logger.error("Error in extract_text: %s", str(e))
            raise
        finally:
            if txt_temp:
                try:
                    os.unlink(txt_temp.name)
                except Exception as e:
                    logger.error(f"Can't delete file: {txt_temp.name}")
                    raise FileSystemException(e)


extract = _ExtractTextFromFile()
