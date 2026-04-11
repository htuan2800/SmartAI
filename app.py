import hashlib
import html
import logging
import re
import shutil
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from document_processor import load_and_split_document
from rag_engine import ask_question, build_retrievers, get_conversational_chain, get_llm, get_vector_store

# Phần nâng cao 7.2.5: Cấu hình logging để theo dõi quá trình truy xuất và xử lý
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


STOPWORDS = {
    "va", "là", "la", "the", "and", "for", "with", "from", "that", "this", "what", "how",
    "cua", "của", "cho", "để", "tren", "trên", "duoc", "được", "hay", "trong", "when", "where",
    "why", "who", "which", "một", "những", "các", "với", "about", "into", "yêu", "cầu",
}


def parse_cited_pages(answer: str):
    """Lấy danh sách trang được model nhắc trực tiếp trong câu trả lời, ví dụ (Page 12) hoặc (Trang 12)."""
    matches = re.findall(r"\((?:page|trang)\s*(\d+)\)", answer or "", flags=re.IGNORECASE)
    return sorted({int(page) for page in matches if int(page) >= 1})


def prettify_source_name(source: str):
    """Rút gọn tên source đang có dạng hash_tenfile để UI dễ đọc."""
    name = Path(source or "").name
    # Trường hợp upload đang lưu dạng <hash>_<original_filename>
    if re.match(r"^[a-f0-9]{64}_", name):
        return name[65:]
    return name


def group_sources_by_page(source_items):
    """Nhóm các chunk theo page để khi bấm vào 1 nguồn chỉ hiện đúng nhóm context của page đó."""
    grouped = {}
    for item in source_items or []:
        page = int(item.get("page", 0) or 0)
        if page < 1:
            continue

        source_name = prettify_source_name(item.get("source", ""))
        if page not in grouped:
            grouped[page] = {
                "source": source_name,
                "chunks": [],
                "upload_at": item.get("upload_at", "N/A"),
                "file_type": item.get("file_type", "N/A")
            }
        grouped[page]["chunks"].append(item.get("content", ""))

    return grouped


def extract_query_terms(query: str):
    """Tách từ khóa chính từ câu hỏi để highlight trong context nguồn."""
    tokens = re.findall(r"[A-Za-zÀ-ỹ0-9]{3,}", query.lower())
    terms = []
    for token in tokens:
        if token not in STOPWORDS and token not in terms:
            terms.append(token)
    return terms[:12]


def highlight_text(text: str, terms):
    """Bọc <mark> cho các từ khóa xuất hiện trong đoạn nguồn để người dùng nhìn nhanh phần liên quan."""
    safe_text = html.escape(text or "")
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(re.escape(html.escape(term)), flags=re.IGNORECASE)
        safe_text = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe_text)
    return safe_text


def init_session_state():
    defaults = {
        "doc_hash": None,
        "retriever": None,
        "vector_retriever": None,
        "bm25_retriever": None,
        "hybrid_retriever": None,
        "conversation_chain": None,
        "llm": None,
        "current_file_path": None,
        "current_file_bytes": None,
        "current_chunks": [],
        "chat_history": [],
        "latest_answer": "",
        "latest_question": "",
        "latest_sources": [],
        "latest_source_items": [],
        "selected_source_page": None,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "search_mode": "Hybrid" if config.USE_HYBRID_DEFAULT else "Vector",
        "use_hybrid": bool(config.USE_HYBRID_DEFAULT),
        "bm25_k": config.BM25_K,
        "hybrid_weight_vector": float(config.HYBRID_WEIGHTS[0]),
        "hybrid_weight_bm25": float(config.HYBRID_WEIGHTS[1]),
        "retriever_signature": None,
        "latest_retrieval_metrics": None,
        "compare_search": bool(config.SHOW_RETRIEVAL_COMPARISON),
        "benchmark_input": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_history_state():
    # Reset cả UI history lẫn memory trong conversational chain.
    chain = st.session_state.get("conversation_chain")
    if chain is not None and getattr(chain, "memory", None) is not None:
        chain.memory.clear()

    # Re-create chain mới để tránh giữ state nội bộ từ phiên trước.
    if st.session_state.get("retriever") is not None:
        if st.session_state.get("llm") is None:
            st.session_state.llm = get_llm()
        st.session_state.conversation_chain = get_conversational_chain(
            st.session_state.retriever,
            st.session_state.llm,
            window_size=config.MEMORY_WINDOW_SIZE,
        )

    st.session_state.chat_history = []
    st.session_state.latest_answer = ""
    st.session_state.latest_question = ""
    st.session_state.latest_sources = []
    st.session_state.latest_source_items = []
    st.session_state.selected_source_page = None


def clear_vector_store_state():
    for folder in (config.FAISS_DIR, config.UPLOAD_DIR):
        for item in folder.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)

    st.session_state.doc_hash = None
    st.session_state.retriever = None
    st.session_state.vector_retriever = None
    st.session_state.bm25_retriever = None
    st.session_state.hybrid_retriever = None
    st.session_state.conversation_chain = None
    st.session_state.llm = None
    st.session_state.current_file_path = None
    st.session_state.current_file_bytes = None
    st.session_state.current_chunks = []
    st.session_state.retriever_signature = None
    st.session_state.latest_retrieval_metrics = None


def parse_benchmark_cases(raw_text: str):
    cases = []
    for line in raw_text.splitlines():
        row = line.strip()
        if not row or "||" not in row:
            continue
        question, keywords_raw = row.split("||", 1)
        keywords = [kw.strip().lower() for kw in keywords_raw.split(",") if kw.strip()]
        if question.strip() and keywords:
            cases.append({"question": question.strip(), "keywords": keywords})
    return cases


def build_retriever_signature():
    """Tạo chữ ký cấu hình retrieval để biết khi nào cần rebuild retriever/chain."""
    return "|".join(
        [
            str(st.session_state.doc_hash),
            str(st.session_state.chunk_size),
            str(st.session_state.chunk_overlap),
            str(st.session_state.search_mode),
            str(int(st.session_state.bm25_k)),
            f"{float(st.session_state.hybrid_weight_vector):.3f}",
            f"{float(st.session_state.hybrid_weight_bm25):.3f}",
            str(st.session_state.get("active_filter")), # Thêm dòng này
        ]
    )


def configure_retrievers_and_chain(chunks, files_identifier: str):
    """Khởi tạo vector/BM25/hybrid retriever và gắn chain theo mode đang chọn.

    Hybrid search dùng EnsembleRetriever để phối hợp 2 tín hiệu:
    - Semantic (FAISS): hiểu ngữ nghĩa câu hỏi.
    - Keyword (BM25): bám từ khóa chính xác.
    """
    vector_db = get_vector_store(
        chunks,
        files_identifier,
        chunk_size=int(st.session_state.chunk_size),
        chunk_overlap=int(st.session_state.chunk_overlap),
    )

    # Tham số filter_metadata lấy từ session_state
    filter_meta = st.session_state.get("active_filter")

    retriever_bundle = build_retrievers(
        vector_db=vector_db,
        chunks=chunks,
        vector_k=int(config.RETRIEVER_K),
        bm25_k=int(st.session_state.bm25_k),
        weights=[
            float(st.session_state.hybrid_weight_vector),
            float(st.session_state.hybrid_weight_bm25),
        ],
        use_hybrid=True,
        filter_metadata=filter_meta, # Thêm dòng này
    )

    st.session_state.vector_retriever = retriever_bundle["vector"]
    st.session_state.bm25_retriever = retriever_bundle["bm25"]
    st.session_state.hybrid_retriever = retriever_bundle["hybrid"]

    use_hybrid_mode = st.session_state.search_mode == "Hybrid" and st.session_state.hybrid_retriever is not None
    st.session_state.use_hybrid = bool(use_hybrid_mode)
    st.session_state.retriever = st.session_state.hybrid_retriever if use_hybrid_mode else st.session_state.vector_retriever

    if st.session_state.llm is None:
        st.session_state.llm = get_llm()

    st.session_state.conversation_chain = get_conversational_chain(
        st.session_state.retriever,
        st.session_state.llm,
        window_size=config.MEMORY_WINDOW_SIZE,
    )
    st.session_state.retriever_signature = build_retriever_signature()


def compare_vector_vs_hybrid(query: str):
    """So sánh query time + số lượng docs truy xuất giữa Vector và Hybrid cho cùng câu hỏi."""
    metrics = {}

    def _safe_page_value(value):
        try:
            parsed = int(value)
            return parsed if parsed >= 1 else 1
        except (TypeError, ValueError):
            return 1

    def _measure(name, retriever_obj):
        if retriever_obj is None:
            return
        start = time.perf_counter()
        docs = retriever_obj.invoke(query)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        pages = sorted({_safe_page_value((doc.metadata or {}).get("page", 1)) for doc in docs})
        top_sources = []
        for doc in docs[:3]:
            metadata = doc.metadata or {}
            top_sources.append(f"{metadata.get('source', 'Unknown')} - Page {_safe_page_value(metadata.get('page', 1))}")
        metrics[name] = {
            "query_time_ms": elapsed_ms,
            "num_docs": len(docs),
            "pages": pages,
            "top_sources": top_sources,
        }

    _measure("vector", st.session_state.vector_retriever)
    _measure("hybrid", st.session_state.hybrid_retriever)

    if "vector" in metrics and "hybrid" in metrics:
        vector_pages = set(metrics["vector"]["pages"])
        hybrid_pages = set(metrics["hybrid"]["pages"])
        metrics["page_overlap"] = sorted(vector_pages.intersection(hybrid_pages))

    return metrics


@st.dialog("Confirm Clear History")
def confirm_clear_history_dialog():
    st.warning("Bạn có chắc muốn xóa toàn bộ lịch sử chat không?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, clear history", key="confirm_clear_history"):
            clear_history_state()
            st.rerun()
    with col2:
        if st.button("Cancel", key="cancel_clear_history"):
            st.rerun()


@st.dialog("Confirm Clear Vector Store")
def confirm_clear_vector_dialog():
    st.warning("Thao tác này sẽ xóa toàn bộ tài liệu đã upload và FAISS cache. Tiếp tục?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, clear vector store", key="confirm_clear_vector"):
            clear_vector_store_state()
            st.rerun()
    with col2:
        if st.button("Cancel", key="cancel_clear_vector"):
            st.rerun()


# 1. CẤU HÌNH UI
st.set_page_config(page_title="SmartDoc AI", page_icon="📚", layout="centered")

# Thêm Custom CSS cơ bản (Mô phỏng Color Palette)
st.markdown(
    """
<style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .stSpinner > div > div { border-top-color: #007BFF !important; }
</style>
""",
    unsafe_allow_html=True,
)

init_session_state()

# 2. SIDEBAR (Bên trái - giữ layout gốc, chỉ bổ sung gọn gàng)
with st.sidebar:
    st.header("Instructions")
    st.markdown(
        """
    1. **Upload:** Click hoặc kéo thả file PDF/DOCX.
    2. **Processing:** Hệ thống tự động xử lý tài liệu.
    3. **Query:** Đặt câu hỏi vào ô nhập liệu.
    """
    )

    st.markdown("---")
    st.header("Settings Information")
    st.markdown(
        f"""
    * **Chunk Size:** {st.session_state.chunk_size} ký tự
    * **Chunk Overlap:** {st.session_state.chunk_overlap} ký tự
    * **Retriever K:** {config.RETRIEVER_K} đoạn văn
    """
    )

    st.markdown("---")
    st.header("Retrieval Strategy")
    st.caption("Hybrid Search = Semantic (FAISS) + Keyword (BM25) để tăng độ bám ngữ nghĩa và từ khóa.")

    st.session_state.search_mode = st.radio(
        "Search Mode",
        options=["Hybrid", "Vector"],
        index=0 if st.session_state.search_mode == "Hybrid" else 1,
        horizontal=True,
    )

    st.session_state.bm25_k = st.number_input(
        "BM25 Top-k",
        min_value=1,
        max_value=20,
        step=1,
        value=int(st.session_state.bm25_k),
    )

    vector_w = st.slider(
        "Weight: Vector",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.hybrid_weight_vector),
        step=0.05,
    )
    bm25_w = st.slider(
        "Weight: BM25",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.hybrid_weight_bm25),
        step=0.05,
    )

    if vector_w + bm25_w == 0:
        st.warning("Tổng weights đang bằng 0, hệ thống tự fallback về [0.5, 0.5].")

    st.session_state.hybrid_weight_vector = float(vector_w)
    st.session_state.hybrid_weight_bm25 = float(bm25_w)
    st.session_state.compare_search = st.checkbox(
        "So sánh Vector vs Hybrid theo từng query",
        value=bool(st.session_state.compare_search),
    )

    st.markdown("---")
    st.header("Filter Settings")
    
    # Lấy danh sách file duy nhất từ các chunks hiện có, hiển thị tên gốc kèm thời gian upload
    if st.session_state.current_chunks:
        from pathlib import Path
        import re
        # Tạo danh sách tuple (label, source_full)
        file_infos = {}
        for chunk in st.session_state.current_chunks:
            meta = chunk.metadata or {}
            source = meta.get("source")
            upload_at = meta.get("upload_at", "?")
            name = Path(source or "").name
            if re.match(r"^[a-f0-9]{64}_", name):
                pretty = name[65:]
            else:
                pretty = name
            label = f"{pretty} ({upload_at})"
            # Nếu trùng label (file trùng tên, cùng thời gian), chỉ lấy 1
            file_infos[label] = source

        labels_sorted = sorted(file_infos.keys())
        selected_labels = st.multiselect("Lọc theo tài liệu:", options=labels_sorted, default=[])

        # Tạo filter dictionary
        if selected_labels:
            # Lấy danh sách source_full tương ứng các label được chọn
            selected_sources = [file_infos[label] for label in selected_labels]
            # Nếu chỉ chọn 1 file thì truyền string, nếu nhiều file thì truyền list
            if len(selected_sources) == 1:
                st.session_state.active_filter = {"source": selected_sources[0]}
            else:
                st.session_state.active_filter = {"source": selected_sources}
            st.info(f"Đang lọc: {', '.join(selected_labels)}")
        else:
            st.session_state.active_filter = None
    else:
        st.session_state.active_filter = None

    with st.expander("Chunk Strategy", expanded=False):
        st.caption("Tùy chỉnh chunk parameters để tối ưu retrieval cho từng tài liệu.")
        use_custom = st.checkbox("Use custom values", value=False)

        if use_custom:
            st.session_state.chunk_size = st.number_input(
                "Chunk Size",
                min_value=200,
                max_value=4000,
                step=50,
                value=int(st.session_state.chunk_size),
            )
            st.session_state.chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=1000,
                step=10,
                value=int(st.session_state.chunk_overlap),
            )
        else:
            size_options = [500, 1000, 1500, 2000]
            overlap_options = [50, 100, 200]
            size_index = size_options.index(st.session_state.chunk_size) if st.session_state.chunk_size in size_options else 1
            overlap_index = overlap_options.index(st.session_state.chunk_overlap) if st.session_state.chunk_overlap in overlap_options else 1

            st.session_state.chunk_size = st.selectbox("Chunk Size", options=size_options, index=size_index)
            st.session_state.chunk_overlap = st.selectbox("Chunk Overlap", options=overlap_options, index=overlap_index)

    st.markdown("---")
    st.header("Model Configuration")
    st.info(f"**LLM:** {config.LLM_MODEL}\n\n**Embedding:** Multilingual MPNet\n\n**Vector DB:** FAISS")

    st.markdown("---")
    st.header("Chat History")
    if st.session_state.chat_history:
        for idx, item in enumerate(reversed(st.session_state.chat_history), start=1):
            with st.expander(f"Q{len(st.session_state.chat_history) - idx + 1}: {item['question'][:70]}"):
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer']}")
                if item.get("source_items"):
                    with st.expander("Nguồn tham chiếu (đã highlight)", expanded=False):
                        terms = extract_query_terms(item.get("question", ""))
                        for source_item in item["source_items"]:
                            st.markdown(
                                f"**{source_item['source']} - Page {source_item['page']}**"
                            )
                            rendered = highlight_text(source_item.get("content", ""), terms)
                            st.markdown(rendered, unsafe_allow_html=True)
    else:
        st.caption("Chưa có lịch sử hội thoại.")

    if st.button("Clear Chat History", use_container_width=True):
        confirm_clear_history_dialog()
    if st.button("Clear Vector Store", use_container_width=True):
        confirm_clear_vector_dialog()


# 3. MAIN AREA (Chính giữa)
st.title("Intelligent Document Q&A System")
st.caption("RAG System với LLMs - Tối ưu hóa: Caching, Fallback, LangDetect")

# FEATURE 1: File Upload (PDF + DOCX)

# --- VALIDATE UPLOAD ---
MAX_FILES = 10
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
ALLOWED_EXTS = {".pdf", ".docx"}

uploaded_files = st.file_uploader(
    "Upload File (Limit 200MB per file • PDF, DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) > 0:
    # Validate số lượng file
    if len(uploaded_files) > MAX_FILES:
        st.error(f"Chỉ được upload tối đa {MAX_FILES} file/lần.")
        st.stop()

    # Validate từng file
    file_errors = []
    for f in uploaded_files:
        ext = Path(f.name).suffix.lower()
        if ext not in ALLOWED_EXTS:
            file_errors.append(f"❌ {f.name}: Định dạng không hợp lệ (chỉ nhận PDF/DOCX)")
        if f.size > MAX_FILE_SIZE:
            file_errors.append(f"❌ {f.name}: Dung lượng vượt quá 200MB")
    if file_errors:
        st.error("\n".join(file_errors))
        st.stop()

    # 1. Tạo một chuỗi định danh duy nhất cho nhóm file này
    files_identifier = "-".join(sorted([f"{f.name}_{f.size}" for f in uploaded_files]))
    strategy_key = f"{st.session_state.chunk_size}:{st.session_state.chunk_overlap}"
    new_hash = hashlib.sha256((files_identifier + strategy_key).encode("utf-8")).hexdigest()


    total_size = sum(f.size for f in uploaded_files)
    if total_size > 200 * 1024 * 1024:
        st.warning(f"Tổng dung lượng tất cả file ({total_size / (1024*1024):.2f} MB) vượt quá 200MB. Có thể gây quá tải bộ nhớ hoặc lỗi xử lý.")
    elif total_size > 50 * 1024 * 1024:
        st.warning(f"Tổng dung lượng file ({total_size / (1024*1024):.2f} MB) lớn hơn 50MB. Quá trình xử lý có thể kéo dài.")

    if st.session_state.doc_hash != new_hash or st.session_state.retriever is None:
        with st.spinner(f"Processing document (Splitting & Creating embeddings)..."):
            all_chunks = []
            current_file_paths = []
            file_process_errors = []
            for uploaded_file in uploaded_files:
                try:
                    file_bytes = uploaded_file.getvalue()
                    raw_hash = hashlib.sha256(file_bytes).hexdigest()
                    clean_name = re.sub(r'[\\/*?:"<>|]', "_", uploaded_file.name)
                    file_path = config.UPLOAD_DIR / f"{raw_hash}_{uploaded_file.name}"
                    if not file_path.exists():
                        file_path.write_bytes(file_bytes)
                    current_file_paths.append(str(file_path))

                    chunks = load_and_split_document(
                        str(file_path),
                        chunk_size=int(st.session_state.chunk_size),
                        chunk_overlap=int(st.session_state.chunk_overlap),
                    )
                    logger.info(f"Processing {len(chunks)} chunks")
                    all_chunks.extend(chunks)
                except Exception as e:
                    file_process_errors.append(f"❌ {uploaded_file.name}: {str(e)}")

            # Validate tổng số chunk tối đa
            MAX_TOTAL_CHUNKS = 10000
            if len(all_chunks) > MAX_TOTAL_CHUNKS:
                st.error(f"Tổng số chunk ({len(all_chunks)}) vượt quá giới hạn {MAX_TOTAL_CHUNKS}. Hãy giảm số lượng hoặc dung lượng file.")
                st.stop()

            if file_process_errors:
                st.error("Một số file bị lỗi khi xử lý:\n" + "\n".join(file_process_errors))
                if not all_chunks:
                    st.stop()

            # Tạo đồng thời retriever vector/BM25/hybrid để có thể đổi mode ngay trên UI.
            st.session_state.doc_hash = new_hash
            vector_db = get_vector_store(
                all_chunks,
                files_identifier, 
                chunk_size=int(st.session_state.chunk_size),
                chunk_overlap=int(st.session_state.chunk_overlap),
            )
            configure_retrievers_and_chain(all_chunks, files_identifier)
            st.session_state.current_file_path = current_file_paths[0] if current_file_paths else None
            st.session_state.current_chunks = all_chunks
            st.session_state.current_files_identifier = files_identifier
            st.session_state.latest_retrieval_metrics = None

            st.success(f"Đã upload thành công {len(all_chunks)} chunk từ {len(uploaded_files)} file.")

# Nếu user đổi mode retrieval/weights/BM25-k sau khi upload, rebuild retriever + chain tương ứng.
if st.session_state.doc_hash and st.session_state.current_chunks:
    latest_signature = build_retriever_signature()
    if latest_signature != st.session_state.retriever_signature:
        try:
            configure_retrievers_and_chain(st.session_state.current_chunks, st.session_state.current_files_identifier)
            logger.info("Retriever pipeline has been rebuilt due to retrieval settings changes")
        except Exception as e:
            st.error("Không thể cập nhật retrieval strategy với cấu hình hiện tại.")
            st.code(str(e))

# FEATURE 2: Question Answering
if st.session_state.retriever is not None:
    st.markdown("---")
    st.subheader("Conversational Q&A")
    st.caption(f"Active retrieval mode: **{st.session_state.search_mode}**")

    # Nếu app reload nhưng chain chưa có (hoặc đang dùng memory format cũ), tự dựng lại.
    chain = st.session_state.conversation_chain
    needs_rebuild = chain is None
    if chain is not None and getattr(chain, "memory", None) is not None:
        # Tránh lỗi Unsupported chat history format khi chain cũ trả history dạng string.
        if getattr(chain.memory, "return_messages", True) is False:
            needs_rebuild = True

    if needs_rebuild:
        if st.session_state.llm is None:
            st.session_state.llm = get_llm()
        st.session_state.conversation_chain = get_conversational_chain(
            st.session_state.retriever,
            st.session_state.llm,
            window_size=config.MEMORY_WINDOW_SIZE,
        )

    # Hiển thị lịch sử theo dạng chat app.
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            # Hiển thị Self-eval nếu có
            if item.get("self_eval_score"):
                st.info(f"**Self-Eval:** {item['self_eval_score']}")
            # Hiển thị câu hỏi đã rewrite nếu có
            if item.get("rewritten_query") and item["rewritten_query"] != item["question"]:
                st.caption(f"Câu hỏi đã rewrite: {item['rewritten_query']}")
            # Hiển thị các bước multi-hop nếu có
            if item.get("multihop_steps"):
                with st.expander("Multi-hop reasoning steps", expanded=False):
                    for idx, step in enumerate(item["multihop_steps"], 1):
                        st.markdown(f"**Bước {idx}:** {step['question']}")
                        st.markdown(f"Trả lời: {step['answer']}")


    query = st.chat_input("Nhập câu hỏi về tài liệu (hỗ trợ follow-up như: 'nó là gì?')")

    if query and query.strip():
        # Validate độ dài câu hỏi
        MAX_QUERY_LEN = 500
        if len(query.strip()) > MAX_QUERY_LEN:
            st.error(f"Câu hỏi quá dài (>{MAX_QUERY_LEN} ký tự). Vui lòng rút ngắn câu hỏi.")
        else:
            logger.info(f"Query: {query}")
            with st.spinner("Processing your query..."):
                try:
                    if st.session_state.compare_search:
                        st.session_state.latest_retrieval_metrics = compare_vector_vs_hybrid(query.strip())
                        logger.info(f"Retrieval compare metrics: {st.session_state.latest_retrieval_metrics}")
                    else:
                        st.session_state.latest_retrieval_metrics = None

                    # Conversational chain tự dùng memory + retrieval để xử lý câu hỏi follow-up.
                    result = ask_question(
                        query.strip(),
                        chain=st.session_state.conversation_chain,
                        llm=st.session_state.llm,
                        return_sources=True,
                        return_source_items=True,
                    )
                    # Hỗ trợ backward compatibility (nếu trả về 3 giá trị cũ)
                    if len(result) == 6:
                        answer, source_pages, source_items, self_eval_score, rewritten_query, multihop_steps = result
                    else:
                        answer, source_pages, source_items = result
                        self_eval_score = None
                        rewritten_query = query.strip()
                        multihop_steps = None
                    st.session_state.latest_answer = answer
                    st.session_state.latest_question = query.strip()
                    st.session_state.latest_sources = sorted({int(page) for page in source_pages if int(page) >= 1})
                    st.session_state.latest_source_items = source_items
                    st.session_state.selected_source_page = None
                    st.session_state.chat_history.append(
                        {
                            "question": query.strip(),
                            "answer": answer,
                            "sources": st.session_state.latest_sources,
                            "source_items": source_items,
                            "self_eval_score": self_eval_score,
                            "rewritten_query": rewritten_query,
                            "multihop_steps": multihop_steps,
                        }
                    )
                except Exception as e:
                    st.error("Không có response từ Model!")
                    st.info("Kiểm tra Ollama đang chạy, sau đó thử lại.")
                    st.code(str(e))

    if st.session_state.latest_retrieval_metrics:
        st.markdown("**Retrieval Performance (Vector vs Hybrid):**")
        rows = []
        for mode in ["vector", "hybrid"]:
            if mode in st.session_state.latest_retrieval_metrics:
                item = st.session_state.latest_retrieval_metrics[mode]
                rows.append(
                    {
                        "mode": mode,
                        "query_time_ms": item["query_time_ms"],
                        "num_docs": item["num_docs"],
                        "pages": ", ".join(str(p) for p in item["pages"][:8]),
                        "top_sources": " | ".join(item["top_sources"]),
                    }
                )

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        overlap = st.session_state.latest_retrieval_metrics.get("page_overlap", [])
        st.caption(f"Page overlap giữa Vector và Hybrid: {overlap if overlap else 'không có'}")

    if st.session_state.latest_answer:
        st.subheader("Response:")
        st.write(st.session_state.latest_answer)
        # Hiển thị Self-eval nếu có
        if st.session_state.chat_history and st.session_state.chat_history[-1].get("self_eval_score"):
            st.info(f"**Self-Eval:** {st.session_state.chat_history[-1]['self_eval_score']}")
        # Hiển thị câu hỏi đã rewrite nếu có
        if st.session_state.chat_history and st.session_state.chat_history[-1].get("rewritten_query") and st.session_state.chat_history[-1]["rewritten_query"] != st.session_state.chat_history[-1]["question"]:
            st.caption(f"Câu hỏi đã rewrite: {st.session_state.chat_history[-1]['rewritten_query']}")
        # Hiển thị các bước multi-hop nếu có
        if st.session_state.chat_history and st.session_state.chat_history[-1].get("multihop_steps"):
            with st.expander("Multi-hop reasoning steps", expanded=False):
                for idx, step in enumerate(st.session_state.chat_history[-1]["multihop_steps"], 1):
                    st.markdown(f"**Bước {idx}:** {step['question']}")
                    st.markdown(f"Trả lời: {step['answer']}")

        if st.session_state.latest_source_items:
            with st.expander("Chi tiết điểm số Re-ranking (Cross-Encoder)", expanded=False):
                st.info("💡 Cross-Encoder đánh giá lại độ liên quan. Điểm càng cao, đoạn văn càng sát với câu hỏi của bạn.")
                
                items = st.session_state.latest_source_items
                
                # 1. Tạo DataFrame với đầy đủ metadata và làm tròn số
                df_scores = pd.DataFrame([
                    {
                        "Nguồn": f"Trang {item['page']} - {item.get('source', '')[:15]}...",
                        "Score": round(item.get('rerank_score', 0), 4), # Làm tròn 4 chữ số
                        "Loại": item.get('file_type', 'N/A'),
                        "Ngày tải": item.get('upload_at', 'N/A'),
                        "Nội dung": item['content'][:150] + "..."
                    } for item in items
                ])

                # 2. Sắp xếp theo Score giảm dần
                df_scores = df_scores.sort_values(by="Score", ascending=False)
                
                # 3. Hiển thị bảng với style
                st.dataframe(
                    df_scores.style.highlight_max(axis=0, subset=['Score'], color='#2e7d32'),
                    use_container_width=True,
                    hide_index=True
                )
                
                # 4. Biểu đồ cột so sánh
                if not df_scores.empty:
                    # Dùng cột Score làm trục Y
                    st.bar_chart(df_scores, x="Nguồn", y="Score", color="#2e7d32")
                    st.caption("Biểu đồ thể hiện mức độ tự tin (Confidence Score) của mô hình Re-ranker.")

        # Ưu tiên trang được nhắc trực tiếp trong answer. Nếu model chưa gắn citation, fallback theo trang retrieve.
        cited_pages = parse_cited_pages(st.session_state.latest_answer)
        pages_for_buttons = cited_pages or st.session_state.latest_sources

        grouped_sources = group_sources_by_page(st.session_state.latest_source_items)

        if pages_for_buttons:
            st.markdown("**Nguồn tham chiếu:**")
            for page in pages_for_buttons:
                info = grouped_sources.get(page, {})
                source_name = info.get("source", "document")
                u_at = info.get("upload_at", "N/A")
                f_type = info.get("file_type", "Unknown")
                chip_label = f"📄 {source_name} • Page {page}"
                col1, col2 = st.columns([0.4, 0.6])
                with col1:
                    if st.button(chip_label, key=f"show_source_page_{page}", use_container_width=True):
                        st.session_state.selected_source_page = page
                with col2:
                    st.caption(f"📅 Upload: {u_at} | 📁 Loại: {f_type}")

        selected_page = st.session_state.selected_source_page
        if selected_page and st.session_state.latest_source_items:
            st.markdown("**Highlight các đoạn văn được sử dụng để trả lời:**")
            st.markdown(f"Trang đang xem: **Page {selected_page}**")
            terms = extract_query_terms(st.session_state.latest_question)
            chunks = grouped_sources.get(int(selected_page), {}).get("chunks", [])
            source_name = grouped_sources.get(int(selected_page), {}).get("source", "document")

            if not chunks:
                st.info("Không tìm thấy chunk tương ứng cho trang này trong top-k retrieval hiện tại.")
            else:
                for idx, chunk in enumerate(chunks, start=1):
                    with st.expander(f"{source_name} - Page {selected_page} - Chunk {idx}", expanded=True):
                        rendered = highlight_text(chunk, terms)
                        st.markdown(rendered, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Chunk Strategy Evaluation")
    st.caption(
        "So sánh chunk strategy bằng Context Hit Rate (độ phủ keyword kỳ vọng trong top-k context truy xuất)."
    )
    st.text_area(
        "Nhập test cases theo định dạng: question || keyword1, keyword2",
        key="benchmark_input",
        height=140,
        placeholder=(
            "Ví dụ:\n"
            "Tài liệu nói gì về mục tiêu dự án? || mục tiêu, dự án\n"
            "Các công nghệ chính là gì? || streamlit, langchain, faiss"
        ),
    )

    if st.button("Run Chunk Benchmark"):
        cases = parse_benchmark_cases(st.session_state.benchmark_input)
        if not cases:
            st.warning("Chưa có test case hợp lệ. Vui lòng nhập đúng định dạng question || keywords.")
        elif not st.session_state.current_file_path or not Path(st.session_state.current_file_path).exists():
            st.warning("Không tìm thấy file hiện tại để benchmark. Vui lòng upload lại tài liệu.")
        else:
            test_configs = [(size, overlap) for size in [500, 1000, 1500, 2000] for overlap in [50, 100, 200]]
            progress = st.progress(0)
            results = []

            with st.spinner("Đang chạy benchmark cho các cấu hình chunk..."):
                for idx, (chunk_size, chunk_overlap) in enumerate(test_configs, start=1):
                    chunks = load_and_split_document(
                        st.session_state.current_file_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    vector_db = get_vector_store(
                        chunks,
                        st.session_state.current_file_bytes,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    retriever = vector_db.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

                    hits = 0
                    for case in cases:
                        docs = retriever.invoke(case["question"])
                        context = "\n".join(doc.page_content for doc in docs).lower()
                        if any(keyword in context for keyword in case["keywords"]):
                            hits += 1

                    accuracy = round((hits / len(cases)) * 100, 2)
                    results.append(
                        {
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "num_chunks": len(chunks),
                            "context_hit_rate_%": accuracy,
                        }
                    )
                    progress.progress(idx / len(test_configs))

            df = pd.DataFrame(results).sort_values(by=["context_hit_rate_%", "num_chunks"], ascending=[False, True])
            st.dataframe(df, use_container_width=True)

            best = df.iloc[0].to_dict()
            st.success(
                "Best config: "
                f"chunk_size={int(best['chunk_size'])}, "
                f"chunk_overlap={int(best['chunk_overlap'])}, "
                f"context_hit_rate={best['context_hit_rate_%']}%"
            )