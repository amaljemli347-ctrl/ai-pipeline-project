import pytest
import sys
import os

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import (
    strip_html,
    normalize_unicode,
    cleanup_whitespace,
    filter_language,
    clean_text
)

def test_strip_html():
    html_text = "<p>Hello <b>World</b>!</p>"
    assert strip_html(html_text) == "Hello  World !"
    assert strip_html("No html here") == "No html here"

def test_normalize_unicode():
    text_with_accents = "Caf\u00e9"
    normalized = normalize_unicode(text_with_accents)
    assert normalized == "Café"
    
def test_cleanup_whitespace():
    messy_text = "  This   has  too \n much \t whitespace.  "
    assert cleanup_whitespace(messy_text) == "This has too much whitespace."

def test_filter_language():
    english_text = "This is a sentence written in English and it should be detected as such."
    french_text = "Ceci est une phrase écrite en français."
    
    assert filter_language(english_text, 'en') is True
    assert filter_language(french_text, 'en') is False

def test_clean_text():
    raw_text = "<p>   This \t is a <b>messy</b>  text.  </p>"
    # strip_html: "   This \t is a  messy   text.  "
    # normalize_unicode: same
    # cleanup: "This is a messy text."
    assert clean_text(raw_text) == "This is a messy text."
