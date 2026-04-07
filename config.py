import os
from pathlib import Path

# Cấu hình đường dẫn lưu trữ Cache
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = DATA_DIR / "faiss_cache"
UPLOAD_DIR = DATA_DIR / "uploads"

# Tự động tạo thư mục nếu chưa có
FAISS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình tham số RAG
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL = "qwen2.5:7b"
RETRIEVER_K = 4  # Top-k cho semantic/vector retrieval
BM25_K = 4       # Top-k cho keyword retrieval bằng BM25
USE_HYBRID_DEFAULT = True
HYBRID_WEIGHTS = (0.5, 0.5)  # (vector_weight, bm25_weight)
SHOW_RETRIEVAL_COMPARISON = True
MEMORY_WINDOW_SIZE = 10  # Số lượt hội thoại gần nhất giữ trong Conversational RAG memory



#phần nâng cao 7.2.4
SEARCH_TYPE = "mmr"   # Thuật toán Maximum Marginal Relevance
FETCH_K = 30          # Lấy 30 kết quả thô ban đầu
LAMBDA_MULT = 0.7     # Độ đa dạng của kết quả