import hashlib
import logging
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from document_processor import load_and_split_document
from rag_engine import ask_question, get_llm, get_vector_store

# Phần nâng cao 7.2.5: Cấu hình logging để theo dõi quá trình truy xuất và xử lý
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_session_state():
    defaults = {
        "doc_hash": None,
        "retriever": None,
        "current_file_path": None,
        "current_file_bytes": None,
        "chat_history": [],
        "latest_answer": "",
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "benchmark_input": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_history_state():
    st.session_state.chat_history = []
    st.session_state.latest_answer = ""


def clear_vector_store_state():
    for folder in (config.FAISS_DIR, config.UPLOAD_DIR):
        for item in folder.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)

    st.session_state.doc_hash = None
    st.session_state.retriever = None
    st.session_state.current_file_path = None
    st.session_state.current_file_bytes = None


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
    else:
        st.caption("Chưa có lịch sử hội thoại.")

    if st.button("Clear History", use_container_width=True):
        confirm_clear_history_dialog()
    if st.button("Clear Vector Store", use_container_width=True):
        confirm_clear_vector_dialog()


# 3. MAIN AREA (Chính giữa)
st.title("Intelligent Document Q&A System")
st.caption("RAG System với LLMs - Tối ưu hóa: Caching, Fallback, LangDetect")

# FEATURE 1: File Upload (PDF + DOCX)
uploaded_file = st.file_uploader(
    "Upload File (Limit 200MB per file • PDF, DOCX)",
    type=["pdf", "docx"],
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    strategy_key = f"{st.session_state.chunk_size}:{st.session_state.chunk_overlap}".encode("utf-8")
    new_hash = hashlib.sha256(file_bytes + strategy_key).hexdigest()

    if uploaded_file.size > 50 * 1024 * 1024:
        st.warning("File tải lên lớn hơn 50MB. Quá trình xử lý có thể kéo dài, vui lòng chờ.")

    if st.session_state.doc_hash != new_hash or st.session_state.retriever is None:
        with st.spinner("Processing document (Splitting & Creating embeddings)..."):
            try:
                raw_hash = hashlib.sha256(file_bytes).hexdigest()
                file_path = config.UPLOAD_DIR / f"{raw_hash}_{uploaded_file.name}"
                file_path.write_bytes(file_bytes)

                chunks = load_and_split_document(
                    str(file_path),
                    chunk_size=int(st.session_state.chunk_size),
                    chunk_overlap=int(st.session_state.chunk_overlap),
                )
                logger.info(f"Processing {len(chunks)} chunks")

                vector_db = get_vector_store(
                    chunks,
                    file_bytes,
                    chunk_size=int(st.session_state.chunk_size),
                    chunk_overlap=int(st.session_state.chunk_overlap),
                )
                st.session_state.retriever = vector_db.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
                st.session_state.doc_hash = new_hash
                st.session_state.current_file_bytes = file_bytes
                st.session_state.current_file_path = str(file_path)

                st.success(f"{uploaded_file.name} uploaded successfully!")
            except Exception as e:
                st.error("Upload/Processing fail!")
                st.info("Kiểm tra file hợp lệ (PDF/DOCX) hoặc thử lại.")
                st.code(str(e))
                st.stop()

# FEATURE 2: Question Answering
if st.session_state.retriever is not None:
    st.markdown("---")
    st.subheader("Ask a Question")
    with st.form("qa_form", clear_on_submit=True):
        query = st.text_input("Enter your question based on the document:")
        submitted = st.form_submit_button("Ask")

    if submitted and query.strip():
        logger.info(f"Query: {query}")
        with st.spinner("Processing your query..."):
            try:
                llm = get_llm()
                answer = ask_question(query.strip(), st.session_state.retriever, llm)
                st.session_state.latest_answer = answer
                st.session_state.chat_history.append({"question": query.strip(), "answer": answer})
            except Exception as e:
                st.error("Không có response từ Model!")
                st.info("Kiểm tra Ollama đang chạy, sau đó thử lại.")
                st.code(str(e))

    if st.session_state.latest_answer:
        st.subheader("Response:")
        st.write(st.session_state.latest_answer)

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