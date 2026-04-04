from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

def load_and_split_pdf(file_path: str):
    """Đọc PDF và chia nhỏ, ưu tiên PDFPlumber, dự phòng bằng PyPDF."""
    try:
        docs = PDFPlumberLoader(file_path).load()
    except Exception:
        docs = PyPDFLoader(file_path).load()
    
    if not docs:
        raise ValueError("Không đọc được nội dung PDF hoặc file rỗng.")

    # Cắt nhỏ văn bản
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    return text_splitter.split_documents(docs)