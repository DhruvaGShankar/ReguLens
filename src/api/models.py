from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


class RetrievalMode(str, Enum):
    HYBRID_RERANKED = "hybrid_reranked"
    HYBRID = "hybrid"
    DENSE = "dense"
    BM25 = "bm25"


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User regulatory compliance or circular query text.",
        examples=["What restrictions apply to borrowing by mutual funds?"]
    )
    mode: Optional[RetrievalMode] = Field(
        default=RetrievalMode.HYBRID_RERANKED,
        description="Retrieval strategy mode. Supported: 'hybrid_reranked', 'hybrid', 'dense', 'bm25'."
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=15,
        description="Number of top evidence chunks to retrieve and feed to the LLM (1 to 15)."
    )
    alpha: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight given to dense retrieval scores during hybrid fusion (0.0 to 1.0)."
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query string cannot be empty or contain only whitespace.")
        return stripped

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What restrictions apply to borrowing by mutual funds?",
                "mode": "hybrid_reranked",
                "top_k": 5,
                "alpha": 0.5
            }
        }
    )


class EvidenceChunkModel(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    document_id: str = Field(..., description="Parent SEBI document ID.")
    circular_number: Optional[str] = Field(None, description="Official SEBI circular reference number.")
    subject: Optional[str] = Field(None, description="Subject / title of the circular.")
    issue_date: Optional[str] = Field(None, description="Publication date of the circular (YYYY-MM-DD).")
    effective_date: Optional[str] = Field(None, description="Effective date of the circular if stated.")
    issuing_department: Optional[str] = Field(None, description="SEBI department code or name.")
    circular_type: Optional[str] = Field(None, description="Type classification (CIRCULAR, MASTER_CIRCULAR).")
    supersession_status: Optional[str] = Field(None, description="Status (in_force, superseded).")
    validity_status: Optional[str] = Field(None, description="Validity status (current, archived).")
    source_url: Optional[str] = Field(None, description="Official SEBI webpage URL.")
    section: Optional[str] = Field(None, description="Detected section clause heading.")
    context_header: Optional[str] = Field(None, description="Injected context header string.")
    text: str = Field(..., description="Full text content of the chunk.")
    body_text: Optional[str] = Field(None, description="Body text without header.")
    word_count: Optional[int] = Field(None, description="Token or word count.")
    score: float = Field(0.0, description="Final relevance score computed by retriever.")
    retrieval_source: Optional[str] = Field(None, description="Source method (dense, bm25, hybrid, cross_encoder).")
    dense_score: Optional[float] = Field(None, description="Raw or normalized dense similarity score.")
    bm25_score: Optional[float] = Field(None, description="Raw or normalized BM25 score.")
    hybrid_score: Optional[float] = Field(None, description="Combined hybrid score.")
    rerank_score: Optional[float] = Field(None, description="Cross-encoder logit relevance score.")

    model_config = ConfigDict(extra="ignore")


class ResponseMetadata(BaseModel):
    retrieved_count: int = Field(..., description="Number of evidence chunks returned.")
    execution_time_ms: float = Field(..., description="Processing time in milliseconds.")
    model_used: str = Field(..., description="LLM model identifier used for generation.")


class ChatResponse(BaseModel):
    request_id: str = Field(..., description="Unique request tracing identifier.")
    query: str = Field(..., description="Original user query.")
    answer: str = Field(..., description="Generated answer grounded strictly in retrieved evidence.")
    evidence_chunks: List[EvidenceChunkModel] = Field(default_factory=list, description="List of supporting evidence chunks.")
    retrieval_mode: str = Field(..., description="Retrieval mode used.")
    refused: bool = Field(..., description="True if query triggered refusal due to insufficient evidence.")
    metadata: ResponseMetadata = Field(..., description="Execution and performance metadata.")


class SystemStatus(BaseModel):
    faiss_index_loaded: bool = Field(..., description="FAISS vector store readiness.")
    faiss_vector_count: int = Field(..., description="Total indexed vectors.")
    bm25_index_loaded: bool = Field(..., description="BM25 index readiness.")
    llm_provider: str = Field(..., description="Active LLM provider.")
    llm_model: str = Field(..., description="Active LLM model.")


class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Overall health status ('healthy' or 'degraded').")
    version: str = Field("1.0.0", description="API version string.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp.")
    system_status: SystemStatus = Field(..., description="RAG subsystem readiness status.")


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error classification code.")
    message: str = Field(..., description="Human-readable error description.")
    details: Optional[List[Dict[str, Any]]] = Field(default=None, description="Field-level validation error details.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp.")


class ErrorResponse(BaseModel):
    error: ErrorDetail = Field(..., description="Standardized error container.")
