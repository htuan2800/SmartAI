import streamlit as st
from pathlib import Path
import config
from document_processor import load_and_split_pdf
from rag_engine import get_vector_store, ask_question, get_llm
import hashlib
import logging

# Phần nâng cao 7.2.5: Cấu hình logging để theo dõi quá trình truy xuất và xử lý
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. CẤU HÌNH UI
st.set_page_config(page_title="SmartDoc AI", page_icon="📚", layout="centered")

# Thêm Custom CSS cơ bản (Mô phỏng Color Palette)
st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    /* Primary color cho các highlight nhỏ */
    .stSpinner > div > div { border-top-color: #007BFF !important; }
</style>
""", unsafe_allow_html=True)

# 2. SIDEBAR (Bên trái)
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. **Upload:** Click hoặc kéo thả file PDF.
    2. **Processing:** Hệ thống tự động xử lý tài liệu.
    3. **Query:** Đặt câu hỏi vào ô nhập liệu.
    """)
    
    st.markdown("---")
    # Settings information
    st.header("Settings Information")
    st.markdown(f"""
    * **Chunk Size:** {config.CHUNK_SIZE} ký tự
    * **Chunk Overlap:** {config.CHUNK_OVERLAP} ký tự
    * **Retriever K:** {config.RETRIEVER_K} đoạn văn
    """)

    st.markdown("---")
    st.header("Model Configuration")
    st.info(f"**LLM:** {config.LLM_MODEL}\n\n**Embedding:** Multilingual MPNet\n\n**Vector DB:** FAISS")


# 3. MAIN AREA (Chính giữa)
st.title("Intelligent Document Q&A System")
st.caption("RAG System với LLMs - Tối ưu hóa: Caching, Fallback, LangDetect")

# Khởi tạo bộ nhớ tạm (Session State)
if "doc_hash" not in st.session_state:
    st.session_state.doc_hash = None
    st.session_state.retriever = None

# FEATURE 1: File Upload
uploaded_file = st.file_uploader("Upload PDF File (Limit 200MB per file • PDF)", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    new_hash = hashlib.sha256(file_bytes).hexdigest()
    
    if uploaded_file.size > 50 * 1024 * 1024: # Đổi 50MB ra bytes
        st.warning("File tải lên lớn hơn 50MB. Quá trình xử lý (processing) có thể diễn ra lâu, vui lòng kiên nhẫn!")

    # Xử lý nếu là file mới hoàn toàn
    if st.session_state.doc_hash != new_hash or st.session_state.retriever is None:
        # Match user flow: "Xem progress Splitting... và Creating embeddings..."
        with st.spinner("Processing document (Splitting & Creating embeddings)..."):
            try:
                # Lưu file tạm
                pdf_path = config.UPLOAD_DIR / f"{new_hash}_{uploaded_file.name}"
                pdf_path.write_bytes(file_bytes)

                # Xử lý PDF & Vector Store
                chunks = load_and_split_pdf(str(pdf_path))
                
                # Phần nâng cao 7.2.5: Cấu hình logging để theo dõi số lượng chunks được tạo ra
                logger.info(f"Processing {len(chunks)} chunks")
                
                vector_db = get_vector_store(chunks, file_bytes)
                
                # Định nghĩa cách tìm kiếm khi up pdf
                st.session_state.retriever = vector_db.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
                
                # Mục 7.2.4: Tìm kiếm Đa dạng hóa (MMR - Nâng cao)
                # st.session_state.retriever = vector_db.as_retriever(
                #     search_type=config.SEARCH_TYPE,
                #     search_kwargs={
                #         "k": config.RETRIEVER_K,
                #         "fetch_k": config.FETCH_K,
                #         "lambda_mult": config.LAMBDA_MULT
                #     }
                # )
                
                st.session_state.doc_hash = new_hash
                
                st.success("PDF uploaded successfully!")
            except Exception as e:
                st.error("Upload/Processing fail!")
                st.info("**Xử lý lỗi:** Check file format (phải là PDF hợp lệ không bị hỏng). Nếu lỗi khác: Xem error message bên dưới và retry.")
                st.code(str(e)) # In rõ error message ra
                st.stop()

# FEATURE 2: Question Answering 
if st.session_state.retriever is not None:
    st.markdown("---")
    st.subheader("Ask a Question")
    query = st.text_input("Enter your question based on the document:")
    
    if query:
        # Phần nâng cao 7.2.5: Cấu hình logging để theo dõi câu hỏi và số lượng tài liệu được truy xuất
        logger.info(f"Query: {query}") 
        # Match user flow: Chờ spinner "Processing your query..."
        with st.spinner("Processing your query..."):
            try:
                llm = get_llm()
                answer = ask_question(query, st.session_state.retriever, llm)
                
                st.subheader("Response:")
                st.write(answer)
            except Exception as e:
                # Error Handling: Model connection errors
                st.error("Không có response từ Model!")
                st.info("**Xử lý lỗi:** Kiểm tra Ollama đang chạy. Nếu lỗi khác: Xem error message bên dưới và retry.")
                st.code(str(e))