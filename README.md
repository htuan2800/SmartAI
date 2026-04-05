# SmartDoc AI - Intelligent Document Q&A System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![LangChain](https://img.shields.io/badge/LangChain-LLM-green)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-orange)

---

## Giới thiệu

SmartDoc AI là hệ thống **RAG (Retrieval-Augmented Generation)** cho phép:

- Hỏi đáp trực tiếp trên file PDF  
- Sử dụng AI (LLM)  

Ứng dụng kết hợp:
- Embeddings
- Vector Search
- Local LLM (Ollama)

---

## Tính năng nổi bật

- Upload PDF trực tiếp
- Tự động xử lý & tạo embeddings
- Tìm kiếm ngữ nghĩa (semantic search)
- Hỏi đáp bằng ngôn ngữ tự nhiên
- UI đơn giản (Streamlit)
- Không cần internet sau khi tải model

---

## Công nghệ sử dụng

| Công nghệ  | Ứng dụng |
|------------|----------|
| Streamlit   | Framework chính để xây dựng Web App |
| HTML/CSS    | Custom styling (tinh chỉnh màu sắc, giao diện). |
| LangChain         | Framework cốt lõi cho ứng dụng LLM. |
| langchain-text-splitters | Phân đoạn văn bản.  |
| LangChain Community  | Các tiện ích mở rộng của cộng đồng. |
| FAISS  | Cơ sở dữ liệu Vector dùng để tìm kiếm sự tương đồng (Vector similarity search) |
| HuggingFace Transformers | Mô hình embedding `Multilingual MPNet` (768-dim) xử lý đa ngôn ngữ. |
|Ollama |Môi trường chạy LLM nội bộ (Local LLM runtime). |
| PDFPlumber  | Trích xuất văn bản từ PDF (Ưu tiên). |
| PyPDF   | Xử lý PDF dự phòng (Alternative). |
| NumPy    | Tính toán số học.|
| Pandas         | Thao tác và xử lý dữ liệu.|
| Torch    | Backend cho Deep learning. |
---

## Yêu cầu hệ thống

- Python **3.8+**
- pip
- Ollama runtime

---

## Cài đặt

### 1. Clone project
```bash
git clone <repository-url>
cd Project-LLMs-Rag-Agent
```

### 2️. Tạo virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

### 3️. Cài dependencies
```bash
pip install -r requirements.txt
```

### 4️. Cài model (Ollama)
```bash
# Download from https://ollama.ai
ollama pull qwen2.5:7b
```

### 5️. Chạy ứng dụng
```bash
streamlit run app.py
```
Mở trình duyệt tại: http://localhost:8501