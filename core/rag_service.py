
import os
import pickle
import hashlib
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from core.config import settings

class RAGService:
    def __init__(self, persistence_dir: str = "./knowledge_base_index"):
        self.persistence_dir = persistence_dir
        self.embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key, model="text-embedding-3-small")
        self.vector_store = None
        self.bm25_retriever = None
        self.documents = []  # Full corpus persistence
        
        self.load_index()

    def load_index(self):
        """Load FAISS, BM25 indices, and document corpus from disk if they exist."""
        if os.path.exists(self.persistence_dir):
            print(f"--- RAG: Loading existing index from {self.persistence_dir} ---")
            
            # Load Corpus
            docs_path = os.path.join(self.persistence_dir, "documents.pkl")
            if os.path.exists(docs_path):
                try:
                    with open(docs_path, "rb") as f:
                        self.documents = pickle.load(f)
                    print(f"--- RAG: Loaded {len(self.documents)} documents from persistence ---")
                except Exception as e:
                    print(f"--- RAG: Error loading documents: {e} ---")
                    self.documents = []

            # Load FAISS
            try:
                self.vector_store = FAISS.load_local(
                    folder_path=self.persistence_dir, 
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True # We trust our own local files
                )
            except Exception as e:
                print(f"--- RAG: Error loading FAISS: {e} ---")
                self.vector_store = None
            
            # Load BM25 (only if enabled)
            if settings.enable_bm25:
                bm25_path = os.path.join(self.persistence_dir, "bm25.pkl")
                if os.path.exists(bm25_path):
                    try:
                        with open(bm25_path, "rb") as f:
                            self.bm25_retriever = pickle.load(f)
                    except Exception as e:
                        print(f"--- RAG: Error loading BM25: {e} ---")
                        # Attempt rebuild from documents if load fails
                        if self.documents:
                            print("--- RAG: Rebuilding BM25 from loaded documents ---")
                            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
                        else:
                            self.bm25_retriever = None
            else:
                 self.bm25_retriever = None
        else:
            print("--- RAG: No existing index found. Starting fresh. ---")

    def save_index(self):
        """Save indices and corpus to disk."""
        if not os.path.exists(self.persistence_dir):
            os.makedirs(self.persistence_dir)
            
        # Save Corpus
        if self.documents:
            with open(os.path.join(self.persistence_dir, "documents.pkl"), "wb") as f:
                 pickle.dump(self.documents, f)
            
        if self.vector_store:
            self.vector_store.save_local(self.persistence_dir)
            
        if self.bm25_retriever and settings.enable_bm25:
            with open(os.path.join(self.persistence_dir, "bm25.pkl"), "wb") as f:
                pickle.dump(self.bm25_retriever, f)
        
        print(f"--- RAG: Index and corpus saved to {self.persistence_dir} ---")

    def _generate_chunk_id(self, content: str) -> str:
        """Generate a stable hash ID for a document chunk."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def add_documents(self, documents: List[Document]):
        """Add new documents to the indices with stable IDs and full corpus persistence."""
        if not documents:
            return
            
        print(f"--- RAG: Adding {len(documents)} documents to index ---")
        
        # Assign stable IDs and deduplicate against existing corpus logic if needed
        # For now, we append. Ideally, we should check if ID exists.
        # Simple deduplication based on ID:
        existing_ids = {doc.metadata.get("chunk_id") for doc in self.documents if doc.metadata.get("chunk_id")}
        
        new_docs_to_add = []
        for doc in documents:
            # Generate and assign ID if not present
            if "chunk_id" not in doc.metadata:
                doc.metadata["chunk_id"] = self._generate_chunk_id(doc.page_content)
            
            if doc.metadata["chunk_id"] not in existing_ids:
                new_docs_to_add.append(doc)
                existing_ids.add(doc.metadata["chunk_id"])
        
        if not new_docs_to_add:
            print("--- RAG: No new documents to add (duplicates skipped) ---")
            return

        print(f"--- RAG: Actually adding {len(new_docs_to_add)} unique documents ---")

        # Update persistent corpus
        self.documents.extend(new_docs_to_add)

        # Update FAISS
        if self.vector_store:
            self.vector_store.add_documents(new_docs_to_add)
        else:
            self.vector_store = FAISS.from_documents(new_docs_to_add, self.embeddings)
            
        # Rebuild BM25 if enabled
        if settings.enable_bm25:
            # To ensure consistency, we rebuild BM25 on the FULL corpus
            # This is fast enough for <100k docs.
            print("--- RAG: Rebuilding BM25 index on full corpus ---")
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        
        self.save_index()

    def hybrid_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform hybrid search using RRF with stable ID matching."""
        if not self.vector_store:
             print("--- RAG: Vector index not ready, returning empty ---")
             return []

        # 1. Semantic Search
        # We fetch more candidates to allow RRF to do its job better
        semantic_k = k * 3 if settings.enable_bm25 else k
        semantic_docs = self.vector_store.similarity_search(query, k=semantic_k)
        
        if not settings.enable_bm25 or not self.bm25_retriever:
            return semantic_docs[:k]

        # 2. Keyword Search
        keyword_k = k * 3
        keyword_docs = self.bm25_retriever.invoke(query)
        keyword_docs = keyword_docs[:keyword_k]
        
        # 3. RRF Fusion
        combined_docs = self._rrf_merge(semantic_docs, keyword_docs, k=k)
        return combined_docs

    def _rrf_merge(self, list1: List[Document], list2: List[Document], k: int = 4, c: int = 60) -> List[Document]:
        """Reciprocal Rank Fusion using stable chunk_ids."""
        from collections import defaultdict
        
        scores = defaultdict(float)
        
        # Map IDs to docs to return the correct objects
        id_to_doc = {}
        
        def process_list(doc_list):
            for rank, doc in enumerate(doc_list):
                # Use chunk_id if available, else page_content (fallback for old index compat)
                doc_id = doc.metadata.get("chunk_id", doc.page_content)
                id_to_doc[doc_id] = doc
                scores[doc_id] += 1 / (rank + c)
        
        process_list(list1)
        process_list(list2)
        
        # Sort by score
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        merged_docs = []
        for doc_id, score in sorted_ids[:k]:
            if doc_id in id_to_doc:
                merged_docs.append(id_to_doc[doc_id])
                
        return merged_docs
