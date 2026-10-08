import pytest
import numpy as np
from pathlib import Path
from src.retrieval.embeddings import generate_embeddings
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.bm25 import BM25Retriever, regulatory_tokenizer


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "chunk_0001",
            "document_id": "doc_0001",
            "circular_number": "SEBI/HO/CFD/2023/123",
            "subject": "Disclosure requirements for listed entities",
            "issue_date": "2023-07-15",
            "issuing_department": "CFD",
            "section": "Regulation 30",
            "text": "[SEBI Circular: SEBI/HO/CFD/2023/123] Listed entities must disclose material events within 24 hours under Regulation 30. Threshold limit is set at 15%.",
            "source_url": "https://www.sebi.gov.in/test1.html"
        },
        {
            "chunk_id": "chunk_0002",
            "document_id": "doc_0002",
            "circular_number": "SEBI/HO/IMD/2022/456",
            "subject": "Mutual fund borrowing restrictions",
            "issue_date": "2022-04-10",
            "issuing_department": "IMD",
            "section": "Borrowing Limits",
            "text": "[SEBI Circular: SEBI/HO/IMD/2022/456] Mutual funds shall not borrow except to meet temporary liquidity requirements for redemption. Maximum borrowing is 20% of net assets.",
            "source_url": "https://www.sebi.gov.in/test2.html"
        },
        {
            "chunk_id": "chunk_0003",
            "document_id": "doc_0003",
            "circular_number": "SEBI/HO/MIRSD/2021/789",
            "subject": "KYC requirements for stock brokers",
            "issue_date": "2021-09-05",
            "issuing_department": "MIRSD",
            "section": "Client Onboarding",
            "text": "[SEBI Circular: SEBI/HO/MIRSD/2021/789] Stock brokers must complete video KYC for resident individuals.",
            "source_url": "https://www.sebi.gov.in/test3.html"
        },
        {
            "chunk_id": "chunk_0004",
            "document_id": "doc_0004",
            "circular_number": "SEBI/HO/AFD/2024/101",
            "subject": "Alternative Investment Funds valuation norms",
            "issue_date": "2024-01-20",
            "issuing_department": "AFD",
            "section": "Valuation of Assets",
            "text": "[SEBI Circular: SEBI/HO/AFD/2024/101] AIF Category I and II funds shall appoint an independent valuer.",
            "source_url": "https://www.sebi.gov.in/test4.html"
        },
        {
            "chunk_id": "chunk_0005",
            "document_id": "doc_0005",
            "circular_number": "SEBI/HO/DDHS/2023/202",
            "subject": "Green debt securities issuance",
            "issue_date": "2023-11-12",
            "issuing_department": "DDHS",
            "section": "Green Bonds",
            "text": "[SEBI Circular: SEBI/HO/DDHS/2023/202] Issuers of green debt securities shall file annual disclosure statements.",
            "source_url": "https://www.sebi.gov.in/test5.html"
        }
    ]


def test_embedding_generation():
    texts = ["Mutual fund borrowing restriction", "SEBI regulatory circular"]
    vecs = generate_embeddings(texts)
    assert vecs.shape == (2, 384)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)


def test_regulatory_tokenizer():
    tokens = regulatory_tokenizer("SEBI/HO/CFD/2023/123 threshold limit 15% and ₹50 crore")
    assert "sebi/ho/cfd/2023/123" in tokens
    assert "15%" in tokens
    assert "₹50" in tokens


def test_faiss_vector_store(tmp_path, sample_chunks):
    index_file = tmp_path / "faiss.index"
    map_file = tmp_path / "faiss_map.json"

    store = FAISSVectorStore(index_path=index_file, map_path=map_file)
    store.build_index(sample_chunks)

    assert index_file.exists()
    assert map_file.exists()

    store2 = FAISSVectorStore(index_path=index_file, map_path=map_file)
    results = store2.search("borrowing by mutual funds", top_k=2)

    assert len(results) >= 1
    top_chunk = results[0]
    assert top_chunk["document_id"] == "doc_0002"
    assert "score" in top_chunk


def test_bm25_retriever(tmp_path, sample_chunks):
    bm25_file = tmp_path / "bm25.pkl"

    bm25 = BM25Retriever(index_path=bm25_file)
    bm25.build_index(sample_chunks)

    assert bm25_file.exists()

    bm25_2 = BM25Retriever(index_path=bm25_file)
    results = bm25_2.search("SEBI/HO/CFD/2023/123", top_k=2)

    assert len(results) >= 1
    assert results[0]["circular_number"] == "SEBI/HO/CFD/2023/123"
