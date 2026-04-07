import hashlib
import importlib
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.llms import Ollama
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


def get_vector_retriever(vector_db, k: int | None = None):
    """Retriever semantic search dựa trên embedding + FAISS."""
    top_k = int(k or config.RETRIEVER_K)
    return vector_db.as_retriever(
        search_type=getattr(config, "SEARCH_TYPE", "similarity"),
        search_kwargs={
            "k": top_k,
            "fetch_k": int(getattr(config, "FETCH_K", max(top_k * 3, top_k))),
            "lambda_mult": float(getattr(config, "LAMBDA_MULT", 0.7)),
        },
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
):
    """Trả về bộ retriever đầy đủ để app có thể chuyển mode linh hoạt."""
    if use_hybrid:
        hybrid, vector, bm25 = get_hybrid_retriever(
            vector_db,
            chunks,
            vector_k=vector_k,
            bm25_k=bm25_k,
            weights=weights,
        )
        return {
            "vector": vector,
            "bm25": bm25,
            "hybrid": hybrid,
        }

    vector = get_vector_retriever(vector_db, k=vector_k)
    return {
        "vector": vector,
        "bm25": None,
        "hybrid": None,
    }

#Khởi tạo cầu nối giao tiếp với phần mềm Ollama (nơi đang chạy mô hình Qwen2.5:7b).
def get_llm():
    try:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=config.LLM_MODEL, temperature=0.7)
    except ImportError:
        return Ollama(model=config.LLM_MODEL, temperature=0.7)

def get_vector_store(
    chunks,
    file_bytes: bytes,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    use_cache: bool = True,
):
    """Sử dụng mã Hash để lưu và tải lại Vector Database từ ổ cứng.

    Cache key bao gồm cả chunk strategy để tránh tái sử dụng sai dữ liệu.
    """
    strategy_key = f"{chunk_size or config.CHUNK_SIZE}:{chunk_overlap or config.CHUNK_OVERLAP}"
    doc_hash = hashlib.sha256(file_bytes + strategy_key.encode("utf-8")).hexdigest()
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

def ask_question(
    query: str,
    retriever=None,
    llm=None,
    chain=None,
    return_sources: bool = False,
    return_source_items: bool = False,
):
    """Tạo prompt có citation; có thể trả thêm danh sách source pages.

    Giữ tương thích ngược: mặc định vẫn trả về chuỗi answer như phiên bản cũ.
    """
    lang = detect_language(query)

    # Nhánh mới: Conversational RAG có memory + retrieval tự động.
    if chain is not None:
        # Nếu không có lịch sử mà user hỏi kiểu follow-up mơ hồ, yêu cầu làm rõ thay vì đoán.
        chat_messages = []
        if getattr(chain, "memory", None) is not None and getattr(chain.memory, "chat_memory", None) is not None:
            chat_messages = getattr(chain.memory.chat_memory, "messages", []) or []

        if not chat_messages and _is_ambiguous_followup(query):
            answer = "Mình chưa có ngữ cảnh trước đó trong phiên chat này. Bạn vui lòng nêu rõ chủ thể/câu hỏi đầy đủ để mình trả lời chính xác."
            source_pages = []
            source_items = []
            if return_sources and return_source_items:
                return answer, source_pages, source_items
            if return_sources:
                return answer, source_pages
            return answer

        result = chain.invoke({"question": query})
        answer = result.get("answer", "")
        docs = result.get("source_documents", []) or []
        logger.info(f"Retrieved {len(docs)} documents via conversational chain")
        source_pages = extract_source_pages(docs)
        source_items = build_source_items(docs)
    else:
        # Nhánh cũ giữ tương thích ngược cho các luồng chưa dùng conversational chain.
        if retriever is None or llm is None:
            raise ValueError("Cần truyền chain hoặc cặp retriever + llm")

        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} documents")
        context = build_citation_context(docs, max_chars=12000)
        source_pages = extract_source_pages(docs)
        source_items = build_source_items(docs)

        if lang == "vi":
            prompt = f"""Sử dụng ngữ cảnh sau đây để trả lời câu hỏi.
Nếu bạn không biết, chỉ cần nói là bạn không biết.
Trả lời ngắn gọn (3-4 câu) BẮT BUỘC bằng tiếng Việt.
Tuyệt đối không được dùng tiếng Trung, tiếng Nhật hoặc tiếng Hàn trong câu trả lời.
Khi phù hợp, hãy tham chiếu nguồn bằng dạng "(Page X)" dựa trên context.

Ngữ cảnh: {context}
Câu hỏi: {query}
Trả lời:"""
        else:
            prompt = f"""Use the following context to answer the question.
If you don't know the answer, just say you don't know. Keep answer concise (3-4 sentences).
When relevant, cite sources using "(Page X)" based on the provided context.

Context: {context}
Question: {query}
Answer:"""

        answer = llm.invoke(prompt)

    # Đảm bảo kết quả cuối không bị lẫn ký tự CJK sai ngôn ngữ mục tiêu.
    answer = enforce_answer_language(answer, lang, llm)
    answer = _normalize_answer_tone(answer)

    if return_sources and return_source_items:
        return answer, source_pages, source_items
    if return_sources:
        return answer, source_pages
    return answer