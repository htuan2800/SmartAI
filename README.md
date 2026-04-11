# SmartDoc AI - Intelligent Document Q&A System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![LangChain](https://img.shields.io/badge/LangChain-LLM-green)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-orange)

---

## Giới thiệu

SmartDoc AI là hệ thống **RAG (Retrieval-Augmented Generation)** cho phép:

- Hỏi đáp trực tiếp trên file PDF/DOCX  
- Sử dụng AI (LLM)  

Ứng dụng kết hợp:
- Embeddings
- Vector Search
- Local LLM (Ollama)

---

## Tính năng nổi bật

- Upload PDF/DOCX trực tiếp
- Tự động xử lý & tạo embeddings
- Tìm kiếm ngữ nghĩa (semantic search)
- Hỏi đáp bằng ngôn ngữ tự nhiên
- Lưu lịch sử hội thoại theo session
- Hiển thị lịch sử chat trong sidebar
- Nút `Clear History` và `Clear Vector Store` có xác nhận trước khi xóa
- Tùy chỉnh `chunk_size`, `chunk_overlap` trên UI
- Benchmark chunk strategy với các cấu hình (500/1000/1500/2000) x (50/100/200)
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
| python-docx | Trích xuất văn bản từ DOCX (bao gồm đoạn văn và bảng). |
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

---

## Tổng quan 10 câu hỏi phát triển thêm

Đánh giá hiện tại: mức độ hoàn thiện tổng thể đạt trên 90% theo nhóm yêu cầu 8.2, với đầy đủ chức năng cốt lõi cho cả 10 câu hỏi và đã có cải tiến bổ sung về độ an toàn, đo hiệu năng, và kiểm thử hồi quy.

### Kiến trúc mở rộng tổng thể

- Presentation Layer: Streamlit UI cho upload, chat, history, benchmark, filter, và bảng metric.
- Processing Layer: loader PDF/DOCX, splitter, metadata enrichment, validation.
- Retrieval Layer: Vector retrieval (FAISS), Keyword retrieval (BM25), Hybrid Ensemble, metadata filtering.
- Ranking Layer: Cross-Encoder re-ranking sau retrieval để tăng độ chính xác ngữ cảnh.
- Reasoning Layer: Conversational RAG memory, query rewriting, self-eval, multi-hop reasoning.
- Storage Layer: local cache FAISS và thư mục uploads cho xử lý nhiều tài liệu.

### Câu hỏi 1: Hỗ trợ file DOCX

- Nội dung: Mở rộng upload và xử lý DOCX song song với PDF.
- Kiến trúc: Nhánh loader riêng cho DOCX trong pipeline xử lý tài liệu.
- Công nghệ: python-docx, LangChain Document abstraction.
- Workflow:
	1. User upload DOCX.
	2. Hệ thống trích xuất đoạn văn và bảng.
	3. Chuẩn hóa metadata source/page/type.
	4. Split thành chunks và đưa vào embedding/indexing.

### Câu hỏi 2: Lưu trữ lịch sử hội thoại

- Nội dung: Lưu các cặp hỏi đáp trong session và hiển thị lại.
- Kiến trúc: Session state làm nơi lưu lịch sử, sidebar là lớp hiển thị.
- Công nghệ: Streamlit session_state.
- Workflow:
	1. User gửi câu hỏi.
	2. Hệ thống sinh câu trả lời.
	3. Lưu question/answer/source vào session.
	4. Sidebar render danh sách hội thoại đã có.

### Câu hỏi 3: Nút xóa lịch sử và vector store

- Nội dung: Cung cấp thao tác dọn trạng thái chat và dữ liệu index.
- Kiến trúc: Dialog xác nhận tách biệt, hàm reset state riêng.
- Công nghệ: Streamlit dialog, thao tác file system local.
- Workflow:
	1. User bấm Clear Chat History hoặc Clear Vector Store.
	2. Dialog xác nhận hiển thị.
	3. Nếu confirm, hệ thống xóa memory/session/cache tương ứng.
	4. UI rerun về trạng thái sạch.

### Câu hỏi 4: Cải thiện chunk strategy

- Nội dung: Cho phép chỉnh chunk_size/chunk_overlap và benchmark nhiều cấu hình.
- Kiến trúc: Configurable splitter + benchmark loop + bảng so sánh kết quả.
- Công nghệ: RecursiveCharacterTextSplitter, Pandas.
- Workflow:
	1. User nhập bộ test question và keywords.
	2. Hệ thống chạy các cặp cấu hình chunk.
	3. Mỗi cấu hình thực hiện retrieval và tính Context Hit Rate.
	4. Trả bảng kết quả và gợi ý cấu hình tốt nhất.

### Câu hỏi 5: Citation và source tracking

- Nội dung: Hiển thị nguồn tham chiếu theo trang, cho click và highlight context.
- Kiến trúc: Source metadata gắn theo chunk, UI nhóm theo page/source.
- Công nghệ: Metadata mapping, HTML highlight trong Streamlit.
- Workflow:
	1. Retrieval trả về source documents.
	2. Hệ thống trích page/source/content.
	3. UI hiển thị nút nguồn theo trang.
	4. Khi click, đoạn liên quan được highlight theo từ khóa câu hỏi.

### Câu hỏi 6: Conversational RAG

- Nội dung: Theo dõi ngữ cảnh hội thoại và trả lời được follow-up.
- Kiến trúc: ConversationalRetrievalChain + window memory.
- Công nghệ: LangChain ConversationalRetrievalChain, ConversationBufferWindowMemory.
- Workflow:
	1. User hỏi câu đầu.
	2. Memory lưu lịch sử gần nhất.
	3. Câu follow-up được xử lý cùng ngữ cảnh hội thoại.
	4. Retrieval + generation trả kết quả liên tục theo phiên chat.

### Câu hỏi 7: Hybrid Search

- Nội dung: Kết hợp semantic retrieval và keyword retrieval.
- Kiến trúc: Vector retriever + BM25 retriever + Ensemble weighted fusion.
- Công nghệ: FAISS, BM25Retriever, EnsembleRetriever.
- Workflow:
	1. Query chạy qua cả vector và BM25.
	2. Kết quả hợp nhất theo trọng số.
	3. Trả top documents cho bước trả lời.
	4. UI hiển thị so sánh Vector vs Hybrid theo thời gian và độ phủ trang.

### Câu hỏi 8: Multi-document với metadata filtering

- Nội dung: Upload nhiều file cùng lúc, lọc truy xuất theo tài liệu.
- Kiến trúc: Multi-file ingestion + metadata-aware retrieval.
- Công nghệ: Streamlit multiple uploader, metadata filter ở retriever.
- Workflow:
	1. User upload nhiều PDF/DOCX.
	2. Mỗi chunk được gắn metadata nguồn, thời gian upload, loại file.
	3. User chọn bộ lọc theo tài liệu trong sidebar.
	4. Retrieval chỉ chạy trên tập tài liệu được chọn.

### Câu hỏi 9: Re-ranking bằng Cross-Encoder

- Nội dung: Đánh giá lại top passages sau retrieval để tăng độ liên quan.
- Kiến trúc: Two-stage retrieval gồm bi-encoder retrieval rồi cross-encoder re-rank.
- Công nghệ: sentence-transformers CrossEncoder.
- Workflow:
	1. Retriever lấy top-k ban đầu.
	2. Cross-Encoder chấm điểm query-passage.
	3. Hệ thống sắp xếp lại và giữ top cuối.
	4. UI hiển thị score, biểu đồ, và metric trước-sau re-rank (docs/pages/latency).

### Câu hỏi 10: Advanced RAG (Self-RAG)

- Nội dung: Tự đánh giá chất lượng trả lời, viết lại câu hỏi, và xử lý multi-hop.
- Kiến trúc: Lớp điều phối trong ask_question với các pha rewrite, retrieve, rerank, self-eval.
- Công nghệ: LLM prompting cho self-eval và query rewriting, logic multi-hop.
- Workflow:
	1. Nhận query gốc.
	2. Nếu mơ hồ thì rewrite thành câu độc lập.
	3. Nếu câu hỏi ghép thì tách multi-hop và xử lý từng nhánh.
	4. Tổng hợp đáp án, sinh confidence/self-eval, trả kèm nguồn và chỉ số.

### Luồng vận hành tổng quát end-to-end

1. Upload một hoặc nhiều tài liệu PDF/DOCX.
2. Loader trích xuất nội dung, chuẩn hóa metadata, split thành chunks.
3. Embedding chunks và lưu vào FAISS cache.
4. Người dùng nhập câu hỏi trong chat.
5. Hệ thống chạy retrieval theo mode Vector hoặc Hybrid, có thể kèm filter metadata.
6. Cross-Encoder re-rank các đoạn liên quan nhất.
7. Conversational chain sinh câu trả lời dựa trên context và memory hội thoại.
8. UI hiển thị answer, citation, highlight context, lịch sử chat, và các bảng metric.
