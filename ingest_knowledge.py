
import os
import sys
import glob
import shutil
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

def move_processed_files(source_base: str, target_base: str):
    """
    Moves files from source subdirectories to target subdirectories based on mapping.
    Mappings:
      gov_ -> government_schemes
      farm-new-source -> farming_practices
    """
    print(f"--- POST-PROCESSING: Moving files from {source_base} to {target_base}... ---")
    
    # Mapping: Source Folder Name -> Target Folder Name
    folder_mapping = {
        "gov_": "government_schemes",
        "farm-new-source": "farming_practices"
    }

    if not os.path.exists(source_base):
        print(f"Source directory {source_base} does not exist. Nothing to move.")
        return

    # Iterate over items in the source base directory
    for item in os.listdir(source_base):
        source_subdir = os.path.join(source_base, item)
        
        # We only care about directories that match our mapping
        if os.path.isdir(source_subdir) and item in folder_mapping:
            target_subdir_name = folder_mapping[item]
            target_subdir = os.path.join(target_base, target_subdir_name)
            
            # Create target subdir if it doesn't exist
            os.makedirs(target_subdir, exist_ok=True)
            
            # Move all files from source_subdir to target_subdir
            files_moved = 0
            for filename in os.listdir(source_subdir):
                file_path = os.path.join(source_subdir, filename)
                if os.path.isfile(file_path):
                    shutil.move(file_path, os.path.join(target_subdir, filename))
                    print(f"Moved: {item}/{filename} -> {target_subdir_name}/{filename}")
                    files_moved += 1
            
            # Remove the source subdirectory if it's empty
            if not os.listdir(source_subdir):
                os.rmdir(source_subdir)
                print(f"Removed empty directory: {source_subdir}")
    
    # Clean up the main source base if empty
    if not os.listdir(source_base):
        os.rmdir(source_base)
        print(f"Removed empty source base: {source_base}")

def ingest():
    inject_dir = "inject_new_sources"
    kb_dir = "knowledge_base"
    
    if not os.path.exists(inject_dir):
        print(f"Directory {inject_dir} not found. Nothing to ingest.")
        return

    print(f"--- INGESTION: data from '{inject_dir}'... ---")
    
    # Load documents ONLY from the injection directory
    raw_docs = load_documents(inject_dir)
    
    if not raw_docs:
        print(f"No documents found in {inject_dir}")
        return

    print(f"--- INGESTION: Found {len(raw_docs)} files. Splitting... ---")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = splitter.split_documents(raw_docs)
    
    print(f"--- INGESTION: Created {len(chunked_docs)} chunks. Indexing... ---")
    rag = RAGService()
    rag.add_documents(chunked_docs)
    print("--- INGESTION: Indexing Complete! ---")
    
    # Move files after successful ingestion
    move_processed_files(inject_dir, kb_dir)
    print("--- PROCESS COMPLETE ---")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    ingest()
