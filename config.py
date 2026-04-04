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
RETRIEVER_K = 4 # Tăng lên 4 để lấy nhiều ngữ cảnh hơn



#phần nâng cao 7.2.4
SEARCH_TYPE = "mmr"   # Thuật toán Maximum Marginal Relevance
RETRIEVER_K = 5       # Lấy 5 kết quả
FETCH_K = 30          # Lấy 30 kết quả thô ban đầu
LAMBDA_MULT = 0.7     # Độ đa dạng của kết quả