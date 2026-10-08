import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from src import config

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_CACHE = None

def get_embedding_model(model_name: str = config.EMBEDDING_MODEL) -> SentenceTransformer:
    global _EMBEDDING_MODEL_CACHE
    if _EMBEDDING_MODEL_CACHE is None:
        logger.info(f"Loading SentenceTransformer model '{model_name}'...")
        _EMBEDDING_MODEL_CACHE = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL_CACHE


def generate_embeddings(texts: List[str], model_name: str = config.EMBEDDING_MODEL, batch_size: int = 64) -> np.ndarray:
    """
    Generate normalized dense vector embeddings for a list of text strings.
    Returns float32 numpy array of shape (N, vector_dim).
    """
    model = get_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings.astype(np.float32)
