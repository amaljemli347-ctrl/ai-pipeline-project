import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm import LLMClient
from src.rag import format_rag_prompt, RAGPipeline

def test_format_rag_prompt_with_context():
    query = "What is Wikipedia?"
    chunks = ["Wikipedia is a free online encyclopedia."]
    messages = format_rag_prompt(query, chunks)
    
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "STRICT RULES" in messages[0]["content"]
    assert "[Source Chunk 1]:" in messages[1]["content"]
    assert "Wikipedia is a free online encyclopedia." in messages[1]["content"]

def test_format_rag_prompt_empty_context():
    query = "What is quantum computing?"
    messages = format_rag_prompt(query, [])
    
    assert "[NO CONTEXT RETRIEVED]" in messages[1]["content"]

def test_llm_client_mock_mode():
    client = LLMClient(provider="openrouter", api_key="")
    messages = [{"role": "user", "content": "Test prompt"}]
    res = client.generate(messages)
    
    assert res["status"] == "mock"
    assert res["text"] is not None
    assert res["latency"] >= 0

def test_verify_groundedness_out_of_domain():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    out_of_domain_answer = "The provided context does not contain enough information to answer your query."
    check = pipeline.verify_groundedness(out_of_domain_answer, ["Some random text"])
    
    assert check["grounded"] is True
    assert check["out_of_domain"] is True

def test_verify_groundedness_grounded_answer():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    chunks = ["Albert Einstein was born in Ulm, Germany in 1879."]
    answer = "Einstein was born in Germany in 1879."
    check = pipeline.verify_groundedness(answer, chunks)
    
    assert check["grounded"] is True
    assert check["out_of_domain"] is False
