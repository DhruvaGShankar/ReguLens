import sys
import logging
from pathlib import Path
import streamlit as st

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import config
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.rag import ReguLensRAG

# Page Configuration
st.set_page_config(
    page_title="ReguLens AI — SEBI Regulatory Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern Chatbot aesthetics
st.markdown("""
<style>
    /* Dark theme & chat styling */
    .stApp {
        background-color: #0E1117;
    }
    .main-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1E2638;
        margin-bottom: 16px;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #8B949E;
        margin-top: 2px;
    }
    .metric-card {
        background: #161B22;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #30363D;
        text-align: center;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #58A6FF;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8B949E;
    }
    .source-card {
        background-color: #161B22;
        border-left: 4px solid #58A6FF;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-top: 1px solid #21262D;
        border-right: 1px solid #21262D;
        border-bottom: 1px solid #21262D;
    }
    .source-card h5 {
        margin: 0 0 6px 0;
        color: #F0F6FC;
        font-size: 0.95rem;
    }
    .source-card p {
        margin: 4px 0;
        font-size: 0.85rem;
        color: #8B949E;
    }
    .refusal-card {
        background-color: #2D1517;
        border-left: 4px solid #F85149;
        padding: 14px;
        border-radius: 6px;
        color: #FF7B72;
        margin-bottom: 12px;
    }
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_pipeline():
    """
    Cache indexing resources to load FAISS, BM25, and models into memory on app startup.
    """
    vector_store = FAISSVectorStore()
    bm25 = BM25Retriever()
    
    try:
        vector_store.load()
    except Exception as e:
        st.sidebar.warning(f"FAISS index missing: {e}. Run scripts/build_index.py")

    try:
        bm25.load()
    except Exception as e:
        st.sidebar.warning(f"BM25 index missing: {e}. Run scripts/build_index.py")

    hybrid_retriever = HybridRetriever(vector_store=vector_store, bm25_retriever=bm25)
    reranker = CrossEncoderReranker()
    rag = ReguLensRAG(hybrid_retriever=hybrid_retriever, reranker=reranker)
    return rag, vector_store, bm25


# Load RAG system
rag_system, vector_store_obj, bm25_obj = load_rag_pipeline()

# Initialize Session Chat Messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am **ReguLens**, your SEBI financial regulatory intelligence assistant.\n\nAsk me any question regarding SEBI circulars, compliance guidelines, disclosure obligations, or investment norms.",
            "evidence": [],
            "refused": False
        }
    ]


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("## ⚖️ **ReguLens Assistant**")
    st.markdown("Citation-Grounded Regulatory AI")
    st.markdown("---")

    # Clear Chat History Button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat history cleared. How can I assist you with SEBI circulars today?",
                "evidence": [],
                "refused": False
            }
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ **Retrieval Settings**")
    
    mode_options = {
        "Hybrid + Reranking (Recommended)": "hybrid_reranked",
        "Hybrid Score Fusion": "hybrid",
        "Dense Vector Retrieval": "dense",
        "BM25 Lexical Retrieval": "bm25"
    }
    selected_mode_label = st.selectbox("Retrieval Strategy:", list(mode_options.keys()))
    retrieval_mode = mode_options[selected_mode_label]

    top_k_val = st.slider("Top-K Evidence Chunks:", min_value=1, max_value=15, value=5)
    alpha_val = st.slider("Hybrid Weight (Dense Alpha):", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    st.markdown("---")
    st.markdown("### 📊 **System Knowledge**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">300</div><div class="metric-label">SEBI Circulars</div></div>', unsafe_allow_html=True)
    with col2:
        num_vecs = vector_store_obj.index.ntotal if vector_store_obj.index else 0
        st.markdown(f'<div class="metric-card"><div class="metric-value">{num_vecs}</div><div class="metric-label">Vector Chunks</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**LLM Provider:** NVIDIA NIM")
    st.markdown(f"**Model:** `{config.LLM_MODEL}`")


# --- MAIN HEADER ---
st.markdown("""
<div class="main-header">
    <div>
        <div class="main-title">ReguLens AI Chat ⚖️</div>
        <div class="sub-title">Citation-Grounded SEBI Financial Regulatory Assistant powered by NVIDIA Nemotron 3 Nano Omni</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sample Query Shortcuts
st.markdown("##### 💡 **Quick Regulatory Queries:**")
demo_queries = [
    "What restrictions apply to borrowing by mutual funds?",
    "What disclosure requirements apply under Regulation 30?",
    "What are the valuation norms for Alternative Investment Funds (AIF)?",
    "What capital rules apply to crypto exchanges under EU MiCA?" # Refusal test
]

shortcut_clicked_query = None
cols = st.columns(len(demo_queries))
for idx, col in enumerate(cols):
    if col.button(f"📌 Query {idx+1}", help=demo_queries[idx], use_container_width=True):
        shortcut_clicked_query = demo_queries[idx]

st.markdown("---")

# Display Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("refused"):
            st.markdown(
                f'<div class="refusal-card">⚠️ <strong>CORPUS REFUSAL:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(msg["content"])

        # Render Evidence & Sources Expanders for Assistant Messages
        evidence = msg.get("evidence", [])
        if evidence:
            with st.expander(f"🔍 **View {len(evidence)} Cited SEBI Evidence Chunks & Official Sources**"):
                tab_sources, tab_chunks = st.tabs(["📚 Citation Sources", "📑 Full Evidence Passages"])

                with tab_sources:
                    for idx, chunk in enumerate(evidence, start=1):
                        circ_num = chunk.get("circular_number", "N/A")
                        subj = chunk.get("subject", "N/A")
                        date = chunk.get("issue_date", "N/A")
                        dept = chunk.get("issuing_department", "N/A")
                        section = chunk.get("section", "General")
                        score = chunk.get("score", 0.0)
                        url = chunk.get("source_url", "#")

                        st.markdown(f"""
                        <div class="source-card">
                            <h5>[{idx}] {subj}</h5>
                            <p><strong>Circular No:</strong> <code>{circ_num}</code> | <strong>Date:</strong> {date} | <strong>Dept:</strong> {dept} | <strong>Section:</strong> {section}</p>
                            <p><strong>Relevance Score:</strong> <code>{score:.4f}</code> | <a href="{url}" target="_blank">Official SEBI Link 🔗</a></p>
                        </div>
                        """, unsafe_allow_html=True)

                with tab_chunks:
                    for idx, chunk in enumerate(evidence, start=1):
                        st.markdown(f"**Chunk [{idx}] — {chunk.get('circular_number')} ({chunk.get('section')})**")
                        st.code(chunk.get("text"), language="text")


# Chat Input Handling
user_input = st.chat_input("Ask a SEBI regulatory compliance or disclosure question...")

prompt_to_process = shortcut_clicked_query or user_input

if prompt_to_process:
    # 1. Append & Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt_to_process, "evidence": [], "refused": False})
    with st.chat_message("user"):
        st.markdown(prompt_to_process)

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching SEBI regulatory corpus and generating grounded response..."):
            response = rag_system.answer_query(prompt_to_process, mode=retrieval_mode, top_k=top_k_val)

        answer_text = response.get("answer", "")
        evidence_chunks = response.get("evidence_chunks", [])
        is_refused = response.get("refused", False) or "Insufficient evidence" in answer_text

        if is_refused:
            st.markdown(
                f'<div class="refusal-card">⚠️ <strong>CORPUS REFUSAL:</strong><br>{answer_text}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(answer_text)

        if evidence_chunks:
            with st.expander(f"🔍 **View {len(evidence_chunks)} Cited SEBI Evidence Chunks & Official Sources**"):
                tab_sources, tab_chunks = st.tabs(["📚 Citation Sources", "📑 Full Evidence Passages"])

                with tab_sources:
                    for idx, chunk in enumerate(evidence_chunks, start=1):
                        circ_num = chunk.get("circular_number", "N/A")
                        subj = chunk.get("subject", "N/A")
                        date = chunk.get("issue_date", "N/A")
                        dept = chunk.get("issuing_department", "N/A")
                        section = chunk.get("section", "General")
                        score = chunk.get("score", 0.0)
                        url = chunk.get("source_url", "#")

                        st.markdown(f"""
                        <div class="source-card">
                            <h5>[{idx}] {subj}</h5>
                            <p><strong>Circular No:</strong> <code>{circ_num}</code> | <strong>Date:</strong> {date} | <strong>Dept:</strong> {dept} | <strong>Section:</strong> {section}</p>
                            <p><strong>Relevance Score:</strong> <code>{score:.4f}</code> | <a href="{url}" target="_blank">Official SEBI Link 🔗</a></p>
                        </div>
                        """, unsafe_allow_html=True)

                with tab_chunks:
                    for idx, chunk in enumerate(evidence_chunks, start=1):
                        st.markdown(f"**Chunk [{idx}] — {chunk.get('circular_number')} ({chunk.get('section')})**")
                        st.code(chunk.get("text"), language="text")

    # 3. Save Assistant Message to State
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "evidence": evidence_chunks,
        "refused": is_refused
    })

