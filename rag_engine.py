import hashlib
import importlib
import time
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.llms import Ollama
from sentence_transformers import CrossEncoder
import numpy as np
import config
import logging
import re

#Phần nâng cao 7.2.5
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_conversational_components():
    """Nạp ConversationalRetrievalChain + ConversationBufferWindowMemory theo phiên bản LangChain đang cài."""
    candidates = [
        ("langchain.chains", "ConversationalRetrievalChain", "langchain.memory", "ConversationBufferWindowMemory"),
        ("langchain_classic.chains", "ConversationalRetrievalChain", "langchain_classic.memory", "ConversationBufferWindowMemory"),
    ]

    for chain_module, chain_class, memory_module, memory_class in candidates:
        try:
            chain_cls = getattr(importlib.import_module(chain_module), chain_class)
            memory_cls = getattr(importlib.import_module(memory_module), memory_class)
            return chain_cls, memory_cls
        except Exception:
            continue

    raise ImportError(
        "Không tìm thấy ConversationalRetrievalChain/ConversationBufferWindowMemory. "
        "Hãy cài gói langchain hoặc langchain-classic tương thích."
    )


ConversationalRetrievalChain, ConversationBufferWindowMemory = _load_conversational_components()


def _load_ensemble_retriever_class():
    """Nạp EnsembleRetriever tương thích nhiều biến thể package LangChain."""
    candidates = [
        ("langchain.retrievers", "EnsembleRetriever"),
        ("langchain_classic.retrievers", "EnsembleRetriever"),
    ]

    for module_name, class_name in candidates:
        try:
            return getattr(importlib.import_module(module_name), class_name)
        except Exception:
            continue

    raise ImportError(
        "Không tìm thấy EnsembleRetriever. Hãy cài bản langchain/langchain-classic tương thích."
    )


EnsembleRetriever = _load_ensemble_retriever_class()


# Prompt rewrite follow-up: chuyển câu hỏi phụ thuộc ngữ cảnh thành câu hỏi độc lập.
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Dựa trên lịch sử hội thoại và câu hỏi follow-up, hãy viết lại thành 1 câu hỏi độc lập.
- Nếu câu hỏi đã rõ ràng hoặc là chủ đề mới, KHÔNG kéo ngữ cảnh cũ vào.
- Chỉ dùng lịch sử khi câu hỏi thật sự phụ thuộc đại từ/tham chiếu mơ hồ.
- Không thêm thông tin mới.

Lịch sử hội thoại:
{chat_history}

Follow-up:
{question}

Câu hỏi độc lập:"""
)


# Prompt QA: buộc model bám ngữ cảnh retrieval và ưu tiên citation theo trang.
QA_PROMPT = PromptTemplate.from_template(
    """Bạn là trợ lý hỏi đáp tài liệu.
- Chỉ dùng thông tin trong CONTEXT để trả lời.
- Nếu không đủ thông tin, nói rõ là không tìm thấy trong tài liệu.
- Trả lời ngắn gọn, rõ ràng, cùng ngôn ngữ với câu hỏi người dùng.
- Khi phù hợp, thêm citation dạng (Page X) dựa trên CONTEXT.
- Không mở đầu bằng các cụm xác nhận cảm tính như "Đúng rồi!", "Chính xác!" khi người dùng chỉ đang đặt câu hỏi.
- Tuyệt đối không bịa nguồn.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
)


# Định dạng mỗi document thành block có [Page X] để model dễ cite.
DOCUMENT_PROMPT = PromptTemplate(
    input_variables=["page_content", "page", "source"],
    template="[Page {page}] (Source: {source})\n{page_content}",
)


def _normalize_answer_tone(answer: str) -> str:
    """Giảm các cụm mở đầu mang tính xác nhận không cần thiết để câu trả lời trung tính hơn."""
    if not answer:
        return answer

    cleaned = (answer or "").strip()
    # Chỉ xử lý cụm đầu câu để tránh làm mất nội dung giữa đoạn.
    prefixes = [
        "Đúng rồi! ", "Đúng rồi!", "Đúng vậy! ", "Đúng vậy!",
        "Chính xác! ", "Chính xác!", "Exactly! ", "Exactly!",
    ]
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].lstrip()
            break
    return cleaned


def _is_ambiguous_followup(query: str) -> bool:
    """Nhận diện câu hỏi phụ thuộc ngữ cảnh kiểu 'ý này', 'nó', 'giải thích thêm'."""
    text = (query or "").strip().lower()
    if not text:
        return False

    markers = [
        "nó là gì",
        "ý này",
        "ý đó",
        "ý trước",
        "cái này",
        "cái đó",
        "điều này",
        "điều đó",
        "giải thích thêm",
        "nói rõ hơn",
        "tiếp đi",
    ]
    return any(marker in text for marker in markers)


def _safe_page(metadata: dict) -> int:
    """Lấy số trang an toàn từ metadata để dùng cho citation."""
    try:
        return int(metadata.get("page", 1))
    except (TypeError, ValueError):
        return 1


def extract_source_pages(docs):
    """Trích xuất danh sách trang nguồn, loại trùng và sắp xếp tăng dần."""
    pages = {_safe_page(getattr(doc, "metadata", {}) or {}) for doc in docs}
    return sorted(page for page in pages if page >= 1)


def build_citation_context(docs, max_chars: int = 12000):
    """Tạo context có nhãn [Page X] để LLM biết đoạn nào đến từ trang nào."""
    blocks = []
    total_chars = 0

    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        page = _safe_page(metadata)
        block = f"[Page {page}]\n{doc.page_content}".strip()

        # Giới hạn độ dài context để tránh lỗi timeout/500 từ model server.
        candidate_size = total_chars + len(block) + 2
        if candidate_size > max_chars:
            remaining = max_chars - total_chars
            if remaining > 20:
                blocks.append(block[:remaining])
            break

        blocks.append(block)
        total_chars = candidate_size

    return "\n\n".join(blocks)


def build_source_items(docs, max_chars_per_item: int = 1200):
    """Chuẩn hóa danh sách nguồn để UI có thể click xem context gốc."""
    items = []
    seen = set()

    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        page = _safe_page(metadata)
        source = metadata.get("source", "Unknown")
        content = (doc.page_content or "").strip()
        if not content:
            continue

        # Cắt bớt chiều dài để UI nhẹ hơn nhưng vẫn đủ ngữ cảnh để kiểm tra.
        content = content[:max_chars_per_item]
        dedupe_key = (source, page, content)
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        items.append(
            {
                "source": source,
                "page": page,
                "content": content,
                "upload_at": metadata.get("upload_at", "N/A"), # <--- THÊM DÒNG NÀY
                "file_type": metadata.get("file_type", "N/A"), # <--- THÊM DÒNG NÀY
            }
        )

    return sorted(items, key=lambda item: (item["page"], item["source"]))


def _contains_cjk(text: str) -> bool:
    """Phát hiện ký tự CJK để bắt trường hợp model lạc sang tiếng Trung/Nhật/Hàn."""
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text or ""))


def _strip_cjk_characters(text: str) -> str:
    """Loại bỏ ký tự CJK ở bước cuối để đảm bảo output không còn tiếng Trung/Nhật/Hàn."""
    cleaned = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", "", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def enforce_answer_language(answer: str, lang: str, llm):
    """Nếu output bị lẫn ngôn ngữ lạ, yêu cầu model viết lại đúng ngôn ngữ đích."""
    if not answer:
        return answer

    if not _contains_cjk(answer):
        return answer

    current = answer
    for _ in range(2):
        if lang == "vi":
            rewrite_prompt = f"""Viết lại 100% bằng tiếng Việt chuẩn, tuyệt đối không dùng ký tự tiếng Trung/Nhật/Hàn.
Giữ nguyên ý chính, ngắn gọn, không thêm thông tin mới.

Nội dung: {current}
Kết quả tiếng Việt:"""
        else:
            rewrite_prompt = f"""Rewrite this in natural English only.
Do not use any Chinese/Japanese/Korean characters.
Keep the meaning unchanged and concise.

Content: {current}
English result:"""

        try:
            current = llm.invoke(rewrite_prompt)
        except Exception:
            break

        if not _contains_cjk(current):
            return current

    # Fallback an toàn: loại hẳn ký tự CJK để tránh hiển thị ngôn ngữ sai yêu cầu.
    return _strip_cjk_characters(current)


def get_embedding():
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def _normalize_weights(weights):
    """Chuẩn hóa weights để tổng bằng 1, tránh cấu hình sai làm lệch score hợp nhất."""
    if not weights or len(weights) != 2:
        return [0.5, 0.5]

    vector_w = max(0.0, float(weights[0]))
    bm25_w = max(0.0, float(weights[1]))
    total = vector_w + bm25_w
    if total == 0:
        return [0.5, 0.5]
    return [vector_w / total, bm25_w / total]


def _bm25_tokenize(text: str):
    """Tokenizer đơn giản cho BM25: lowercase + tách token chữ/số đa ngôn ngữ."""
    return re.findall(r"[A-Za-zÀ-ỹ0-9]{2,}", (text or "").lower())


def get_vector_retriever(vector_db, k: int | None = None, filter_metadata: dict | None = None):
    """Retriever semantic search dựa trên embedding + FAISS."""
    top_k = int(k or config.RETRIEVER_K)
    """Cập nhật: Thêm tham số filter_metadata."""
    search_kwargs = {
        "k": top_k,
        "fetch_k": int(getattr(config, "FETCH_K", max(top_k * 3, top_k))),
        "lambda_mult": float(getattr(config, "LAMBDA_MULT", 0.7)),
    }
    # Nếu có filter, thêm vào search_kwargs
    if filter_metadata:
        search_kwargs["filter"] = filter_metadata
    return vector_db.as_retriever(
        search_type=getattr(config, "SEARCH_TYPE", "similarity"),
        search_kwargs=search_kwargs,
    )


def get_bm25_retriever(chunks, k: int | None = None):
    """Tạo keyword retriever bằng BM25.

    BM25 hoạt động theo cơ chế xếp hạng lexical relevance (TF-IDF biến thể),
    rất tốt khi câu hỏi chứa từ khóa cụ thể cần match chính xác.
    """
    bm25 = BM25Retriever.from_documents(chunks, preprocess_func=_bm25_tokenize)
    bm25.k = int(k or config.BM25_K)
    return bm25


def get_hybrid_retriever(vector_db, chunks, vector_k: int | None = None, bm25_k: int | None = None, weights=None):
    """Kết hợp semantic search + keyword search bằng EnsembleRetriever.

    EnsembleRetriever sẽ chạy nhiều retriever song song và gộp/xếp hạng lại
    theo trọng số. Nhờ vậy, hệ thống vừa hiểu nghĩa (semantic) vừa giữ được
    độ chính xác theo từ khóa (lexical).
    """
    vector_retriever = get_vector_retriever(vector_db, k=vector_k)
    bm25_retriever = get_bm25_retriever(chunks, k=bm25_k)
    normalized_weights = _normalize_weights(weights or list(getattr(config, "HYBRID_WEIGHTS", (0.5, 0.5))))

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=normalized_weights,
    )
    return hybrid_retriever, vector_retriever, bm25_retriever


def build_retrievers(
    vector_db,
    chunks,
    vector_k: int | None = None,
    bm25_k: int | None = None,
    weights=None,
    use_hybrid: bool = True,
    filter_metadata: dict | None = None
):
    """Trả về bộ retriever đầy đủ để app có thể chuyển mode linh hoạt."""
    """Cập nhật: Truyền filter xuống các sub-retrievers."""
    
    vector = get_vector_retriever(vector_db, k=vector_k, filter_metadata=filter_metadata)
    # BM25 lọc thủ công (Simple approach)
    relevant_chunks = chunks
    if filter_metadata and "source" in filter_metadata:
        filter_val = filter_metadata["source"]
        if isinstance(filter_val, list):
            relevant_chunks = [c for c in chunks if c.metadata.get("source") in filter_val]
        else:
            relevant_chunks = [c for c in chunks if c.metadata.get("source") == filter_val]

    if use_hybrid:
        bm25 = get_bm25_retriever(relevant_chunks, k=bm25_k)
        normalized_weights = _normalize_weights(weights or list(getattr(config, "HYBRID_WEIGHTS", (0.5, 0.5))))
        
        hybrid = EnsembleRetriever(
            retrievers=[vector, bm25],
            weights=normalized_weights,
        )
        return {"vector": vector, "bm25": bm25, "hybrid": hybrid}
    return {"vector": vector, "bm25": None, "hybrid": None}

#Khởi tạo cầu nối giao tiếp với phần mềm Ollama (nơi đang chạy mô hình Qwen2.5:7b).
def get_llm():
    try:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=config.LLM_MODEL, temperature=0.7)
    except ImportError:
        return Ollama(model=config.LLM_MODEL, temperature=0.7)

def get_vector_store(
    chunks,
    identifier: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    use_cache: bool = True,
):
    """Sử dụng mã Hash để lưu và tải lại Vector Database từ ổ cứng.

    Cache key bao gồm cả chunk strategy để tránh tái sử dụng sai dữ liệu.
    """
    strategy_key = f"{chunk_size or config.CHUNK_SIZE}:{chunk_overlap or config.CHUNK_OVERLAP}"
    # Tạo hash dựa trên danh sách file + nội dung file + cấu hình chunk
    combined_info = (identifier + strategy_key).encode("utf-8")
    doc_hash = hashlib.sha256(combined_info).hexdigest()
    
    persist_dir = config.FAISS_DIR / doc_hash
    embedder = get_embedding()

    # Nếu đã có cache trên ổ cứng, Load lên ngay lập tức
    if use_cache and persist_dir.exists() and any(persist_dir.iterdir()):
        return FAISS.load_local(str(persist_dir), embedder, allow_dangerous_deserialization=True)
    
    # Nếu chưa có, tính toán Embedding và lưu xuống ổ cứng
    vector_db = FAISS.from_documents(chunks, embedder)
    if use_cache:
        vector_db.save_local(str(persist_dir))
    return vector_db


def get_conversational_chain(retriever, llm, window_size: int = 10):
    """Khởi tạo ConversationalRetrievalChain với window memory để hỗ trợ follow-up question.

    - Memory lưu Q/A gần nhất (k=window_size) để tránh phình context quá dài.
    - Chain tự động rewrite câu hỏi follow-up theo lịch sử rồi mới retrieval.
    - return_source_documents=True để giữ metadata citation cho UI.
    """
    memory = ConversationBufferWindowMemory(
        k=max(1, int(window_size)),
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        # ConversationalRetrievalChain cần list message/tuple, không phải string history.
        return_messages=True,
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={
            "prompt": QA_PROMPT,
            "document_prompt": DOCUMENT_PROMPT,
            "document_variable_name": "context",
        },
        verbose=False,
    )

def detect_language(text: str) -> str:
    """Tự động phát hiện ngôn ngữ."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        vietnamese_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
        return "vi" if any(char in text.lower() for char in vietnamese_chars) else "en"

class ReRanker:
    def __init__(self, model_name=config.RERANK_MODEL):
        # Load model vào CPU (hoặc GPU nếu có)
        self.model = CrossEncoder(model_name, max_length=512)

    # Thêm hàm này để fix lỗi 'no attribute predict'
    def predict(self, pairs):
        return self.model.predict(pairs)
    
    def rerank(self, query: str, documents: list):
        if not documents:
            return []
        
        # Chuẩn bị cặp (Query, Passages)
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Dự đoán điểm số (Scores càng cao càng liên quan)
        scores = self.model.predict(pairs)
        
        # Gắn score vào metadata để theo dõi
        for i, doc in enumerate(documents):
            doc.metadata["rerank_score"] = float(scores[i])
        
        # Sắp xếp lại theo score giảm dần
        reranked_docs = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
        
        return reranked_docs

_reranker_instance = None

def get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = ReRanker()
    return _reranker_instance

def ask_question(
    query: str,
    retriever=None,
    llm=None,
    chain=None,
    return_sources: bool = False,
    return_source_items: bool = False,
    enable_self_eval: bool = True,
    enable_query_rewrite: bool = True,
    enable_multihop: bool = True,
):
    """
    Advanced RAG: Self-eval, Query Rewriting, Multi-hop Reasoning, Confidence scoring.
    """
    lang = detect_language(query)
    source_documents = []
    answer = ""
    self_eval_score = None
    rewritten_query = query
    multihop_steps = []
    rerank_metrics = {
        "retrieval_time_ms": 0.0,
        "rerank_time_ms": 0.0,
        "total_pipeline_ms": 0.0,
        "docs_before_rerank": 0,
        "docs_after_rerank": 0,
        "pages_before_rerank": [],
        "pages_after_rerank": [],
    }

    # --- QUERY REWRITING---
    if enable_query_rewrite and _is_ambiguous_followup(query):
        try:
            # Use LLM to rewrite ambiguous query to standalone
            rewrite_prompt = f"Viết lại câu hỏi sau thành câu hỏi độc lập, rõ ràng, không phụ thuộc ngữ cảnh trước đó.\nCâu hỏi: {query}\nCâu hỏi độc lập:"
            rewritten_query = llm.invoke(rewrite_prompt).strip()
        except Exception:
            rewritten_query = query

    # --- MULTI-HOP REASONING---
    # For demo: If question contains 'and', split and answer sub-questions, then synthesize
    if enable_multihop and re.search(r"\b(and|và)\b", rewritten_query, flags=re.IGNORECASE):
        sub_questions = [
            q.strip(" ,.;:\t\n")
            for q in re.split(r"\b(?:and|và)\b", rewritten_query, flags=re.IGNORECASE)
            if q.strip()
        ]
        sub_answers = []
        for sq in sub_questions:
            sub_result = ask_question(
                sq, retriever, llm, chain, True, True, enable_self_eval=False, enable_query_rewrite=False, enable_multihop=False
            )
            if isinstance(sub_result, tuple) and len(sub_result) >= 3:
                sub_ans, sub_src, sub_items = sub_result[:3]
            else:
                sub_ans, sub_src, sub_items = str(sub_result), [], []
            sub_answers.append(sub_ans)
            multihop_steps.append({'question': sq, 'answer': sub_ans, 'sources': sub_src, 'source_items': sub_items})
        # Tổng hợp câu trả lời cuối cùng
        answer = "\n".join([f"- {a}" for a in sub_answers])
        # Hợp nhất các nguồn
        source_documents = []
        final_source_items = []
        source_pages = []
        for step in multihop_steps:
            if step['source_items']:
                final_source_items.extend(step['source_items'])
            if step['sources']:
                source_pages.extend(step['sources'])
        # Loại bỏ trùng
        source_pages = sorted({int(p) for p in source_pages if int(p) >= 1})
        # Tự đánh giá câu trả lời tổng hợp
        if enable_self_eval:
            try:
                eval_prompt = f"Đánh giá mức độ đúng/sát của câu trả lời sau với câu hỏi: '{query}'.\nCâu trả lời: {answer}\nChấm điểm từ 1 (kém) đến 5 (rất tốt), chỉ trả về số điểm và một câu nhận xét ngắn."
                self_eval_score = llm.invoke(eval_prompt)
            except Exception:
                self_eval_score = None
        if return_sources and return_source_items:
            rerank_metrics["docs_before_rerank"] = len(final_source_items)
            rerank_metrics["docs_after_rerank"] = len(final_source_items)
            rerank_metrics["pages_before_rerank"] = source_pages
            rerank_metrics["pages_after_rerank"] = source_pages
            return answer, source_pages, final_source_items, self_eval_score, rewritten_query, multihop_steps, rerank_metrics
        if return_sources:
            return answer, source_pages, self_eval_score, rewritten_query, multihop_steps
        return answer

    # --- RETRIEVAL + RE-RANKING
    if chain is not None:
        retrieval_started = time.perf_counter()
        chat_messages = []
        if getattr(chain, "memory", None) is not None and getattr(chain.memory, "chat_memory", None) is not None:
            chat_messages = getattr(chain.memory.chat_memory, "messages", []) or []
        if not chat_messages and _is_ambiguous_followup(query):
            answer = "Mình chưa có ngữ cảnh trước đó trong phiên chat này. Bạn vui lòng nêu rõ chủ thể/câu hỏi đầy đủ để mình trả lời chính xác."
            return (answer, [], [], None, rewritten_query, multihop_steps, rerank_metrics) if (return_sources and return_source_items) else answer
        result = chain.invoke({"question": rewritten_query})
        rerank_metrics["retrieval_time_ms"] = round((time.perf_counter() - retrieval_started) * 1000, 2)
        answer = result.get("answer", "")
        source_documents = result.get("source_documents", []) or []
    else:
        if retriever is None or llm is None:
            raise ValueError("Cần truyền chain hoặc cặp retriever + llm")
        retrieval_started = time.perf_counter()
        if hasattr(retriever, "search_kwargs"):
            retriever.search_kwargs["k"] = config.TOP_K_RERANK
        source_documents = retriever.invoke(rewritten_query)
        rerank_metrics["retrieval_time_ms"] = round((time.perf_counter() - retrieval_started) * 1000, 2)
        if not source_documents:
            return "Xin lỗi, mình không tìm thấy thông tin nào liên quan đến câu hỏi này trong tài liệu bạn đã cung cấp."

    rerank_metrics["docs_before_rerank"] = len(source_documents)
    rerank_metrics["pages_before_rerank"] = extract_source_pages(source_documents)

    # --- RE-RANKING ---
    final_source_items = []
    source_pages = []
    if config.USE_RERANKER and source_documents:
        rerank_started = time.perf_counter()
        reranker = get_reranker()
        passages = [doc.page_content for doc in source_documents]
        sentence_pairs = [[rewritten_query, p] for p in passages]
        scores = reranker.predict(sentence_pairs)
        for i, doc in enumerate(source_documents):
            doc.metadata["rerank_score"] = float(scores[i])
        source_documents = sorted(source_documents, key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
        source_documents = source_documents[:config.FINAL_K]
        rerank_metrics["rerank_time_ms"] = round((time.perf_counter() - rerank_started) * 1000, 2)

    rerank_metrics["docs_after_rerank"] = len(source_documents)
    rerank_metrics["pages_after_rerank"] = extract_source_pages(source_documents)

    # --- ANSWER GENERATION ---
    if chain is None:
        context = build_citation_context(source_documents, max_chars=12000)
        if lang == "vi":
            prompt = f"""Sử dụng ngữ cảnh sau đây để trả lời câu hỏi.\nNếu bạn không biết, chỉ cần nói là bạn không biết.\nTrả lời ngắn gọn (3-4 câu) BẮT BUỘC bằng tiếng Việt.\nTuyệt đối không được dùng tiếng Trung, tiếng Nhật hoặc tiếng Hàn trong câu trả lời.\nKhi phù hợp, hãy tham chiếu nguồn bằng dạng \"(Page X)\" dựa trên context.\n\nNgữ cảnh: {context}\nCâu hỏi: {rewritten_query}\nTrả lời:"""
        else:
            prompt = f"""Use the following context to answer the question.\nIf you don't know the answer, just say you don't know. Keep answer concise (3-4 sentences).\nWhen relevant, cite sources using \"(Page X)\" based on the provided context.\n\nContext: {context}\nQuestion: {rewritten_query}\nAnswer:"""
        answer = llm.invoke(prompt)

    # --- SELF-EVALUATION---
    if enable_self_eval:
        try:
            eval_prompt = f"Đánh giá mức độ đúng/sát của câu trả lời sau với câu hỏi: '{query}'.\nCâu trả lời: {answer}\nChấm điểm từ 1 (kém) đến 5 (rất tốt), chỉ trả về số điểm và một câu nhận xét ngắn."
            self_eval_score = llm.invoke(eval_prompt)
        except Exception:
            self_eval_score = None

    # --- PACKAGE RESULTS ---
    for doc in source_documents:
        m = doc.metadata
        source_pages.append(m.get("page", 1))
        final_source_items.append({
            "content": doc.page_content,
            "source": m.get("source", ""),
            "page": m.get("page", "N/A"),
            "chunk": m.get("chunk", "N/A"),
            "upload_at": m.get("upload_at", "N/A"),
            "file_type": m.get("file_type", "N/A"),
            "rerank_score": m.get("rerank_score", 0)
        })

    answer = enforce_answer_language(answer, lang, llm)
    answer = _normalize_answer_tone(answer)
    rerank_metrics["total_pipeline_ms"] = round(
        float(rerank_metrics.get("retrieval_time_ms", 0.0)) + float(rerank_metrics.get("rerank_time_ms", 0.0)),
        2,
    )

    if return_sources and return_source_items:
        return answer, source_pages, final_source_items, self_eval_score, rewritten_query, multihop_steps, rerank_metrics
    if return_sources:
        return answer, source_pages, self_eval_score, rewritten_query, multihop_steps
    return answer