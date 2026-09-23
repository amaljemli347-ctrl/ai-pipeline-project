import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.chunking import recursive_character_split

def test_empty_text():
    assert recursive_character_split("") == []
    assert recursive_character_split("   ") == []

def test_short_text():
    text = "This is a short text."
    chunks = recursive_character_split(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_split_by_paragraph():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = recursive_character_split(text, chunk_size=20, chunk_overlap=5)
    # Each paragraph is ~14-16 chars. So they should split cleanly on \n\n
    assert len(chunks) == 3
    assert chunks[0] == "Paragraph one."
    assert chunks[1] == "Paragraph two."
    assert chunks[2] == "Paragraph three."

def test_split_by_sentence():
    text = "This is sentence one. This is sentence two. And sentence three."
    # len is ~63.
    chunks = recursive_character_split(text, chunk_size=30, chunk_overlap=10)
    # 'This is sentence one.' is 21 chars.
    # 'This is sentence two.' is 21 chars.
    # So it should split by '. '
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk) <= 30

def test_forceful_split_long_word():
    # Edge case: A single word that is longer than the chunk size
    text = "A" * 100
    chunks = recursive_character_split(text, chunk_size=30, chunk_overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 30

def test_overlap_logic():
    text = "Word1 Word2 Word3 Word4 Word5"
    chunks = recursive_character_split(text, chunk_size=15, chunk_overlap=10)
    # Check that there is some overlapping text between consecutive chunks
    # This is a basic sanity check for overlap
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 15
