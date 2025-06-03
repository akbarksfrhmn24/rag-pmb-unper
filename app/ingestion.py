# app/ingestion.py

import os
from pathlib import Path
import fitz            # PyMuPDF for PDF processing and image extraction
import pdfplumber     # for table extraction
import pytesseract    # for OCR on images
from PIL import Image
from io import BytesIO

from langchain.text_splitter import CharacterTextSplitter
from app.vector_store import get_vector_store
from app.embedding import LocalEmbedding

def extract_text_from_pdf(file_path: str) -> str:
    """Extract plain text from a PDF file using PyMuPDF."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_tables_from_pdf(file_path: str) -> str:
    """
    Extract tables from a PDF using pdfplumber.
    Returns a concatenated string of CSV-formatted tables.
    """
    tables_text = ""
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            for table in page.extract_tables():
                # Convert each row to CSV line
                for row in table:
                    # join cell values with commas
                    line = ",".join(cell or "" for cell in row)
                    tables_text += f"TABLE_ROW: {line}\n"
                tables_text += "\n"  # separate tables
    return tables_text

def extract_images_and_ocr(file_path: str) -> str:
    """
    Extract images from PDF via PyMuPDF, run OCR via pytesseract,
    and return concatenated OCR text.
    """
    ocr_text = ""
    doc = fitz.open(file_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha < 4:  # this is GRAY or RGB
                img_data = pix.get_image_data("png")
            else:  # CMYK: convert to RGB first
                pix = fitz.Pixmap(fitz.csRGB, pix)
                img_data = pix.get_image_data("png")
            # load into PIL
            image = Image.open(BytesIO(img_data))
            # perform OCR
            try:
                text = pytesseract.image_to_string(image)
                if text.strip():
                    ocr_text += f"IMAGE_OCR_PAGE{page_index+1}_{img_index+1}: {text}\n"
            except Exception:
                # if pytesseract not available or fails, skip
                pass
            pix = None
    return ocr_text

def extract_text_from_markdown(file_path: str) -> str:
    """Extract text from a Markdown file."""
    return Path(file_path).read_text(encoding="utf-8")

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a plain .txt file."""
    return Path(file_path).read_text(encoding="utf-8")

def process_document(file_path: str) -> list[str]:
    """
    Process the document based on its file extension:
    - PDFs: extract text + tables + OCR from images
    - Markdown / TXT: extract text
    Then split into chunks.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        # 1) plain text
        text = extract_text_from_pdf(file_path)
        # 2) tables
        tables = extract_tables_from_pdf(file_path)
        # 3) images OCR
        ocr   = extract_images_and_ocr(file_path)
        combined = "\n\n".join([text, tables, ocr])
    elif ext in [".md", ".markdown"]:
        combined = extract_text_from_markdown(file_path)
    elif ext == ".txt":
        combined = extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file format")

    # Split into overlapping chunks
    splitter = CharacterTextSplitter(separator="\n", chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(combined)
    return [chunk for chunk in chunks if chunk.strip()]

def ingest_document(file_path: str):
    """
    Ingest a document by:
      1. Extracting text (+ tables + OCR) and splitting into chunks.
      2. Generating embeddings for each chunk.
      3. Adding the chunks with embeddings to the vector store.
    """
    chunks = process_document(file_path)
    if not chunks:
        raise ValueError("No valid text chunks found in the document.")

    embedding_model = LocalEmbedding()
    vector_store    = get_vector_store()

    for chunk in chunks:
        try:
            embedding = embedding_model.embed_query(chunk)
        except Exception as e:
            raise ValueError(f"Failed to embed chunk: {chunk[:30]}... Error: {e}")
        if not embedding:
            raise ValueError(f"Empty embedding for chunk: {chunk[:30]}...")

        print(f"Indexing chunk: {chunk[:30]}... [len={len(embedding)}]")
        vector_store.add_texts(
            texts=[chunk],
            metadatas=[{"source": os.path.basename(file_path)}],
            embeddings=[embedding]
        )
    vector_store.persist()
