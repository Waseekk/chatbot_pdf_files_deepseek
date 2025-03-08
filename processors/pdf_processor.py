import streamlit as st
import fitz  # PyMuPDF
from typing import Dict
from .utils import _is_table

class PDFProcessor:
    """Handles multiple PDF files processing and text extraction with page tracking."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> Dict[str, any]:
        """Extracts text and metadata from PDF with page numbers."""
        try:
            pdf_bytes = pdf_file.read()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Dictionary to store text content with page numbers
            pages_content = {}
            full_text = ""
            tables = []
            images = []
            
            # Extract text and content page by page
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                
                # Extract tables with better error handling
                try:
                    blocks = page.get_text("blocks")
                    page_tables = []
                    for block in blocks:
                        if _is_table(block):
                            page_tables.append({
                                'page': page_num + 1,
                                'content': block[4] if isinstance(block, tuple) and len(block) > 4 else str(block)
                            })
                    tables.extend(page_tables)
                except Exception as e:
                    st.warning(f"Could not extract tables from page {page_num + 1}: {str(e)}")
                    continue
                
                # Extract images
                for img_index, img in enumerate(page.get_images(full=True)):
                    images.append({
                        'page': page_num + 1,
                        'index': img_index,
                        'type': img[1],
                    })
                
                pages_content[page_num + 1] = {
                    'text': page_text,
                    'tables': [t for t in tables if t['page'] == page_num + 1],
                    'images': [i for i in images if i['page'] == page_num + 1]
                }
                full_text += f"\n[Page {page_num + 1}]\n{page_text}"
            
            metadata = {
                "title": pdf_document.metadata.get("title", "Unknown"),
                "author": pdf_document.metadata.get("author", "Unknown"),
                "total_pages": len(pdf_document),
                "total_tables": len(tables),
                "total_images": len(images),
                "creation_date": pdf_document.metadata.get("creationDate", "Unknown"),
                "modification_date": pdf_document.metadata.get("modDate", "Unknown")
            }
            
            return {
                "text": full_text,
                "pages_content": pages_content,
                "metadata": metadata,
                "tables": tables,
                "images": images
            }
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return {"text": "", "pages_content": {}, "metadata": {}, "tables": [], "images": []}