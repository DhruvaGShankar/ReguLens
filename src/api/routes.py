import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, status
from openai import RateLimitError, APIConnectionError, APIStatusError

from src import config
from src.api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SystemStatus,
    ResponseMetadata,
    EvidenceChunkModel
)
from src.generation.rag import ReguLensRAG

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Regulatory Intelligence RAG API"])

# Global single instance of RAG pipeline
_rag_instance: Optional[ReguLensRAG] = None


def get_rag_service() -> ReguLensRAG:
    global _rag_instance
    if _rag_instance is None:
        logger.info("Initializing ReguLensRAG pipeline for FastAPI service...")
        _rag_instance = ReguLensRAG()
    return _rag_instance


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Verify backend API health, version, and index readiness without calling external LLM providers."
)
def get_health() -> HealthResponse:
    rag = get_rag_service()
    
    vector_store = rag.retriever.vector_store
    bm25 = rag.retriever.bm25_retriever
    
    faiss_loaded = vector_store.index is not None
    vector_count = vector_store.index.ntotal if faiss_loaded else 0
    bm25_loaded = bm25.bm25 is not None
    
    overall_healthy = faiss_loaded and bm25_loaded
    
    return HealthResponse(
        status="healthy" if overall_healthy else "degraded",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        system_status=SystemStatus(
            faiss_index_loaded=faiss_loaded,
            faiss_vector_count=vector_count,
            bm25_index_loaded=bm25_loaded,
            llm_provider=getattr(config, "LLM_PROVIDER", "nvidia"),
            llm_model=getattr(config, "LLM_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
        )
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit Regulatory Chat Query",
    description="Submit a natural language SEBI regulatory question to retrieve grounded circular evidence and LLM response."
)
def submit_chat_query(request: ChatRequest) -> ChatResponse:
    start_time = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    rag = get_rag_service()
    
    try:
        raw_result = rag.answer_query(
            query=request.query,
            mode=request.mode.value,
            top_k=request.top_k
        )
    except ValueError as ve:
        logger.warning(f"Validation error processing query [{request_id}]: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation Error: {str(ve)}"
        )
    except RateLimitError as rle:
        logger.error(f"LLM Rate Limit Error [{request_id}]: {rle}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="LLM provider rate limit exceeded. Please retry your query after a brief pause."
        )
    except (APIConnectionError, APIStatusError) as apie:
        logger.error(f"Upstream Provider Error [{request_id}]: {apie}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream LLM provider service returned an error. Please try again later."
        )
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Unexpected Backend Exception [{request_id}]: {err_msg}")
        
        if "429" in err_msg or "Rate limit" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="LLM provider rate limit exceeded. Please retry after a brief pause."
            )
        elif "503" in err_msg or "Service Unavailable" in err_msg or "502" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Upstream LLM provider is currently overloaded or unavailable. Please retry shortly."
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred processing your regulatory query."
        )
        
    execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    
    raw_evidence = raw_result.get("evidence_chunks", [])
    evidence_models = [EvidenceChunkModel(**chunk) for chunk in raw_evidence]
    
    answer_text = raw_result.get("answer", "")
    is_refused = raw_result.get("refused", False) or "Insufficient evidence" in answer_text
    
    return ChatResponse(
        request_id=request_id,
        query=request.query,
        answer=answer_text,
        evidence_chunks=evidence_models,
        retrieval_mode=raw_result.get("retrieval_mode", request.mode.value),
        refused=is_refused,
        metadata=ResponseMetadata(
            retrieved_count=len(evidence_models),
            execution_time_ms=execution_time_ms,
            model_used=getattr(config, "LLM_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
        )
    )
