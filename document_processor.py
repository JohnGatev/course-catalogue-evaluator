import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
import io
import os

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == '.pdf':
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text("text") + "\n"
        except Exception as e:
            text = f"Error reading PDF: {e}"
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = "Unsupported format."
    return text

def extract_url_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()
        # get text
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        return f"Error scraping URL: {e}"

def highlight_pdf(file_path, exact_quotes):
    """
    Takes a PDF file path and a list of exact quotes.
    Highlights the quotes in red and returns the modified PDF as bytes.
    """
    try:
        doc = fitz.open(file_path)
        for quote in exact_quotes:
            if not quote.strip(): continue
            # Attempt to highlight. fitz.search_for works within a page.
            # Long quotes spanning pages won't be easily highlighted.
            # We truncate long quotes for search if needed, but exact quotes are best.
            for page in doc:
                instances = page.search_for(quote)
                for inst in instances:
                    annot = page.add_highlight_annot(inst)
                    annot.set_colors(stroke=(1, 0, 0))  # Red highlight
                    annot.update()
        
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    except Exception as e:
        print(f"Error highlighting: {e}")
        return b""
