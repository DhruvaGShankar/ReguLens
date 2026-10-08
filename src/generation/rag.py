import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src import config
from src.generation.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


import os
from dotenv import load_dotenv


def get_openai_client() -> Optional[OpenAI]:
    """
    Initialize OpenAI-compatible client dynamically reading environment config.
    """
    load_dotenv(override=True)
    api_key = os.getenv("LLM_API_KEY", getattr(config, "LLM_API_KEY", ""))
    base_url = os.getenv("LLM_BASE_URL", getattr(config, "LLM_BASE_URL", "")) or "https://integrate.api.nvidia.com/v1"

    if not api_key or api_key == "your_openrouter_key_here":
        logger.warning("LLM API key not configured in .env. LLM generation will fallback to mock/dry-run mode.")
        return None

    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(prompt: str, model_name: Optional[str] = None) -> str:
    """
    Send prompt to OpenAI-compatible LLM endpoint (NVIDIA NIM / Nemotron).
    """
    load_dotenv(override=True)
    if not model_name:
        model_name = os.getenv("LLM_MODEL", getattr(config, "LLM_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"))

    client = get_openai_client()
    if client is None:
        return "Insufficient evidence was found in the indexed SEBI corpus to answer this question reliably."

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error calling LLM API: {e}")
        return f"Error executing LLM request: {str(e)}"


class ReguLensRAG:
    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None
    ):
        self.retriever = hybrid_retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()

    def retrieve_evidence(self, query: str, mode: str = "hybrid_reranked", top_k: int = config.FINAL_TOP_K) -> List[Dict[str, Any]]:
        """
        Retrieve evidence chunks based on mode: 'dense', 'bm25', 'hybrid', or 'hybrid_reranked'.
        """
        if mode == "dense":
            return self.retriever.search_dense(query, top_k=top_k)
        elif mode == "bm25":
            return self.retriever.search_bm25(query, top_k=top_k)
        elif mode == "hybrid":
            return self.retriever.search_hybrid(query, candidate_k=top_k)
        elif mode == "hybrid_reranked":
            candidates = self.retriever.search_hybrid(query, candidate_k=20)
            return self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

    def answer_query(self, query: str, mode: str = "hybrid_reranked", top_k: int = config.FINAL_TOP_K) -> Dict[str, Any]:
        """
        End-to-end RAG workflow: Retrieve evidence -> Construct prompt -> Generate grounded response.
        """
        evidence_chunks = self.retrieve_evidence(query, mode=mode, top_k=top_k)

        # Refusal check if no chunks returned
        if not evidence_chunks:
            refusal_msg = "Insufficient evidence was found in the indexed SEBI corpus to answer this question reliably."
            return {
                "query": query,
                "answer": refusal_msg,
                "evidence_chunks": [],
                "retrieval_mode": mode,
                "refused": True
            }

        prompt = build_rag_prompt(query, evidence_chunks)
        llm_output = call_llm(prompt)

        return {
            "query": query,
            "answer": llm_output,
            "evidence_chunks": evidence_chunks,
            "retrieval_mode": mode,
            "refused": "Insufficient evidence" in llm_output
        }
