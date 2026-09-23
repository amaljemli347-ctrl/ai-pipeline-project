import re
import unicodedata
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Ensure consistent language detection
DetectorFactory.seed = 0

def strip_html(text: str) -> str:
    """Removes HTML tags from the given text."""
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")

def normalize_unicode(text: str) -> str:
    """Normalizes unicode characters to NFKC form."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)

def cleanup_whitespace(text: str) -> str:
    """Removes extra whitespace, tabs, and newlines."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def filter_language(text: str, target_lang: str = 'en') -> bool:
    """Returns True if the detected language matches the target_lang."""
    try:
        if not text or len(text.strip()) < 10:
            return False
        return detect(text) == target_lang
    except LangDetectException:
        return False

def clean_text(text: str) -> str:
    """Applies all text cleaning steps sequentially."""
    text = strip_html(text)
    text = normalize_unicode(text)
    text = cleanup_whitespace(text)
    return text

def process_document(text: str) -> dict:
    """Processes a document and determines if it should be kept."""
    cleaned = clean_text(text)
    is_target_lang = filter_language(cleaned, 'en')
    return {
        "cleaned_text": cleaned,
        "is_valid": is_target_lang
    }
