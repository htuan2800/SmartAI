from pathlib import Path

from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

def _load_pdf(file_path: str):
    """Đọc PDF, ưu tiên PDFPlumber, dự phòng bằng PyPDF."""
    try:
        docs = PDFPlumberLoader(file_path).load()
    except Exception:
        docs = PyPDFLoader(file_path).load()

    if not docs:
        raise ValueError("Không đọc được nội dung PDF hoặc file rỗng.")

    return docs


def _load_docx(file_path: str):
    """Đọc DOCX bằng python-docx, giữ lại đoạn văn và nội dung bảng."""
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise ImportError(
            "Thiếu thư viện python-docx. Vui lòng cài: pip install python-docx"
        ) from exc

    doc = DocxDocument(file_path)
    sections = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            sections.append(text)

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                sections.append(" | ".join(cells))

    full_text = "\n".join(sections).strip()
    if not full_text:
        raise ValueError("Không đọc được nội dung DOCX hoặc file rỗng.")

    return [Document(page_content=full_text, metadata={"source": file_path, "type": "docx"})]


def load_and_split_document(file_path: str, chunk_size: int | None = None, chunk_overlap: int | None = None):
    """Đọc tài liệu (PDF/DOCX) và chia nhỏ theo chunk strategy."""
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        docs = _load_pdf(file_path)
    elif suffix == ".docx":
        docs = _load_docx(file_path)
    else:
        raise ValueError("Định dạng file chưa được hỗ trợ. Vui lòng dùng PDF hoặc DOCX.")

    # Cắt nhỏ văn bản
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP
    )
    return text_splitter.split_documents(docs)


def load_and_split_pdf(file_path: str):
    """Giữ tương thích ngược với code cũ."""
    return load_and_split_document(file_path)