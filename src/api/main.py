import os
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.responses import JSONResponse

from contextlib import asynccontextmanager
from src.api.routes import router as api_router, get_rag_service
from src.api.models import ErrorResponse, ErrorDetail

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to initialize and pre-warm RAG index resources on app startup.
    """
    logger.info("FastAPI App Lifespan Startup: Initializing ReguLens RAG pipeline...")
    try:
        rag = get_rag_service()
        vstore = rag.retriever.vector_store
        bm25_ret = rag.retriever.bm25_retriever
        
        if vstore.index is not None and bm25_ret.bm25 is not None:
            logger.info(f"RAG indexes successfully initialized! FAISS vector count: {vstore.index.ntotal}")
        else:
            logger.warning("RAG indexes missing or partially degraded on startup.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline during app startup: {e}", exc_info=True)
    yield
    logger.info("FastAPI App Lifespan Shutdown.")


app = FastAPI(
    title="ReguLens Regulatory Intelligence RAG API",
    description="Citation-Grounded Financial Regulatory Intelligence API powering SEBI circular question answering.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configure CORS Middleware for Frontend Development
cors_origins_str = os.getenv("CORS_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Format Pydantic request validation errors into standard ErrorResponse format.
    """
    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": " -> ".join(str(loc) for loc in error.get("loc", [])),
            "issue": error.get("msg", ""),
            "type": error.get("type", "")
        })
        
    response_payload = ErrorResponse(
        error=ErrorDetail(
            code="VALIDATION_ERROR",
            message="Request validation failed. Please check your query parameters or request body format.",
            details=error_details,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_payload.model_dump()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Format HTTP exceptions into standard ErrorResponse format.
    """
    code_map = {
        400: "BAD_REQUEST",
        404: "NOT_FOUND",
        422: "UNPROCESSABLE_ENTITY",
        429: "RATE_LIMIT_EXCEEDED",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
        500: "INTERNAL_SERVER_ERROR"
    }
    err_code = code_map.get(exc.status_code, "ERROR")
    
    response_payload = ErrorResponse(
        error=ErrorDetail(
            code=err_code,
            message=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=response_payload.model_dump()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all unhandled exception handler ensuring no sensitive stack traces are exposed.
    """
    logger.error(f"Unhandled Exception on path {request.url.path}: {exc}", exc_info=True)
    response_payload = ErrorResponse(
        error=ErrorDetail(
            code="INTERNAL_SERVER_ERROR",
            message="An internal server error occurred while processing your request.",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_payload.model_dump()
    )


# Include API Router
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
