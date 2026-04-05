import hashlib
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import config
import logging

#Phần nâng cao 7.2.5
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embedding():
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

#Khởi tạo cầu nối giao tiếp với phần mềm Ollama (nơi đang chạy mô hình Qwen2.5:7b).
def get_llm():
    try:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=config.LLM_MODEL, temperature=0.7)
    except ImportError:
        return Ollama(model=config.LLM_MODEL, temperature=0.7)

def get_vector_store(chunks, file_bytes: bytes):
    """Sử dụng mã Hash để lưu và tải lại Vector Database từ ổ cứng."""
    doc_hash = hashlib.sha256(file_bytes).hexdigest()
    persist_dir = config.FAISS_DIR / doc_hash
    embedder = get_embedding()

    # Nếu đã có cache trên ổ cứng, Load lên ngay lập tức
    if persist_dir.exists() and any(persist_dir.iterdir()):
        return FAISS.load_local(str(persist_dir), embedder, allow_dangerous_deserialization=True)
    
    # Nếu chưa có, tính toán Embedding và lưu xuống ổ cứng
    vector_db = FAISS.from_documents(chunks, embedder)
    vector_db.save_local(str(persist_dir))
    return vector_db

def detect_language(text: str) -> str:
    """Tự động phát hiện ngôn ngữ."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        vietnamese_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
        return "vi" if any(char in text.lower() for char in vietnamese_chars) else "en"

def ask_question(query: str, retriever, llm):
    """Tạo Prompt và giới hạn Context an toàn."""
    lang = detect_language(query)
    #Thực thi tìm kiếm : FAISS tìm top-k chunks liên quan nhất
    docs = retriever.invoke(query)
    #Phần 7.2.5
    logger.info(f"Retrieved {len(docs)} documents")
    # Gom văn bản và ngắt ở mức 12000 ký tự để tránh lỗi 500 của Ollama
    context = "\n\n".join([d.page_content for d in docs])
    if len(context) > 12000:
        context = context[:12000]

    #Tạo prompt
    if lang == "vi":
        prompt = f"""Sử dụng ngữ cảnh sau đây để trả lời câu hỏi.
Nếu bạn không biết, chỉ cần nói là bạn không biết.
Trả lời ngắn gọn (3-4 câu) BẮT BUỘC bằng tiếng Việt.

Ngữ cảnh: {context}
Câu hỏi: {query}
Trả lời:"""
    else:
        prompt = f"""Use the following context to answer the question.
If you don't know the answer, just say you don't know. Keep answer concise (3-4 sentences).

Context: {context}
Question: {query}
Answer:"""
    #Sinh câu trả lời
    return llm.invoke(prompt)