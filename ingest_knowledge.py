
import os
import sys
import glob
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.rag_service import RAGService
from pypdf import PdfReader

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def load_documents(directory: str) -> List[Document]:
    documents = []
    
    # Process Markdown and Text
    for file_path in glob.glob(os.path.join(directory, "**/*.*"), recursive=True):
        # Determine category from subfolder
        # e.g. knowledge_base/farming_practices/guide.md -> category: farming_practices
        rel_path = os.path.relpath(file_path, directory)
        category = os.path.dirname(rel_path)
        if category == "":
            category = "general"

        if file_path.endswith(('.md', '.txt')):
            print(f"Loading {file_path} (Category: {category})...")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Add simple metadata
                    doc = Document(page_content=content, metadata={
                        "source": os.path.basename(file_path),
                        "category": category
                    })
                    documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        elif file_path.endswith('.pdf'):
            print(f"Loading PDF {file_path} (Category: {category})...")
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                doc = Document(page_content=text, metadata={
                    "source": os.path.basename(file_path),
                    "category": category
                })
                documents.append(doc)
            except Exception as e:
                print(f"Error loading PDF {file_path}: {e}")

    return documents

def ingest():
    kb_dir = "knowledge_base"
    if not os.path.exists(kb_dir):
        print(f"Directory {kb_dir} not found. Creating it.")
        os.makedirs(kb_dir)
        return

    print("--- INGESTION: Loading documents... ---")
    raw_docs = load_documents(kb_dir)
    
    if not raw_docs:
        print("No documents found in knowledge_base/")
        return

    print(f"--- INGESTION: Found {len(raw_docs)} files. Splitting... ---")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = splitter.split_documents(raw_docs)
    
    print(f"--- INGESTION: Created {len(chunked_docs)} chunks. Indexing... ---")
    rag = RAGService()
    rag.add_documents(chunked_docs)
    print("--- INGESTION: Complete! ---")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    ingest()
