import time
import os
import logging
from typing import Dict, Any, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer

from src.llm import LLMClient

logger = logging.getLogger("RAG_Pipeline")

SYSTEM_PROMPT = """You are an Enterprise AI Assistant specializing in factual question-answering based STRICTLY on retrieved documentation.

STRICT RULES:
1. Answer the user's question using ONLY the facts explicitly mentioned in the provided [CONTEXT].
2. Do NOT use any prior knowledge or make assumptions beyond what is stated in the context.
3. If the [CONTEXT] does not contain sufficient information to answer the question, or if the question is off-topic/out-of-domain relative to the context, you MUST reply with exactly:
   "The provided context does not contain enough information to answer your query."
4. Do NOT attempt to invent, extrapolate, or hallucinate facts not present in the context.
5. Keep your answer concise, precise, and directly grounded in the source text.
"""

def format_rag_prompt(query: str, context_chunks: List[str]) -> List[Dict[str, str]]:
    """
    Formats system and user messages with context injection.
    """
    if not context_chunks:
        formatted_context = "[NO CONTEXT RETRIEVED]"
    else:
        formatted_context = "\n\n---\n\n".join(
            [f"[Source Chunk {i+1}]:\n{chunk}" for i, chunk in enumerate(context_chunks)]
        )

    user_message = f"""[CONTEXT]
{formatted_context}

[USER QUESTION]
{query}

[ANSWER]"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]


class RAGPipeline:
    """
    End-to-End RAG Pipeline:
    1. Query embedding & ChromaDB vector retrieval
    2. Context injection & Prompt engineering
    3. LLM generation with error handling
    4. Hallucination detection & latency benchmarking
    """
    def __init__(
        self,
        chroma_path: str = "data/chroma_db",
        collection_name: str = "wikipedia_chunks",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_client: Optional[LLMClient] = None
    ):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.llm = llm_client or LLMClient()
        
        logger.info(f"Loading embedding model '{embedding_model_name}'...")
        self.embedder = SentenceTransformer(embedding_model_name)
        
        logger.info(f"Connecting to ChromaDB at '{chroma_path}'...")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        except Exception as e:
            logger.warning(f"Could not load collection '{collection_name}': {e}. Collection will need to be created.")
            self.collection = None

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieves top_k relevant chunks from ChromaDB for a given query.
        Returns retrieved chunks, distance scores, and retrieval latency.
        """
        start_time = time.time()
        
        if not self.collection:
            retrieval_latency = time.time() - start_time
            return {"chunks": [], "distances": [], "latency": retrieval_latency}

        query_embedding = self.embedder.encode([query])[0].tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieval_latency = time.time() - start_time
        
        chunks = results['documents'][0] if results.get('documents') else []
        distances = results['distances'][0] if results.get('distances') else []
        
        return {
            "chunks": chunks,
            "distances": distances,
            "latency": retrieval_latency
        }

    def verify_groundedness(self, answer: str, context_chunks: List[str]) -> Dict[str, Any]:
        """
        Checks if the generated answer is grounded in context or indicates out-of-domain/lack of context.
        """
        is_out_of_domain = "does not contain enough information" in answer.lower()
        
        if is_out_of_domain:
            return {
                "grounded": True,
                "out_of_domain": True,
                "reason": "Correctly identified missing context or out-of-domain query."
            }
            
        if not context_chunks:
            return {
                "grounded": False,
                "out_of_domain": True,
                "reason": "Answer generated despite empty context (potential hallucination)."
            }

        # Check keyword/semantic overlap for basic hallucination detection
        answer_words = set(answer.lower().split())
        context_words = set(" ".join(context_chunks).lower().split())
        overlap = answer_words.intersection(context_words)
        
        overlap_ratio = len(overlap) / max(len(answer_words), 1)
        
        is_grounded = overlap_ratio > 0.2
        
        return {
            "grounded": is_grounded,
            "out_of_domain": False,
            "overlap_ratio": round(overlap_ratio, 2),
            "reason": "Grounded in context" if is_grounded else "Low lexical overlap with context (potential hallucination)."
        }

    def query(self, user_query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes end-to-end RAG workflow:
        Retrieve -> Prompt -> Generate -> Verify -> Benchmark Latency
        """
        overall_start = time.time()
        
        # 1. Retrieval
        retrieval_res = self.retrieve(user_query, top_k=top_k)
        chunks = retrieval_res["chunks"]
        retrieval_latency = retrieval_res["latency"]
        
        # 2. Prompt Formatting
        messages = format_rag_prompt(user_query, chunks)
        
        # 3. LLM Generation
        llm_res = self.llm.generate(messages=messages, temperature=0.1, max_tokens=400)
        answer = llm_res["text"]
        generation_latency = llm_res["latency"]
        
        # 4. Hallucination & Groundedness Verification
        groundedness = self.verify_groundedness(answer, chunks)
        
        total_latency = time.time() - overall_start
        
        return {
            "query": user_query,
            "answer": answer,
            "retrieved_chunks": chunks,
            "groundedness_check": groundedness,
            "latency": {
                "retrieval_s": round(retrieval_latency, 4),
                "generation_s": round(generation_latency, 4),
                "total_s": round(total_latency, 4)
            },
            "llm_status": llm_res["status"],
            "model_used": llm_res["model"]
        }
