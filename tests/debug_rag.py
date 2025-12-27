import os
import sys
# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from core.rag_service import RAGService
from core.config import settings

print(f"OpenAI API Key present: {bool(settings.openai_api_key)}")
if settings.openai_api_key:
    print(f"Key starts with: {settings.openai_api_key[:5]}...")

print("Initializing RAGService...")
try:
    rag = RAGService()
    print("RAGService initialized.")
    
    print(f"Vector Store: {rag.vector_store}")
    print(f"BM25 Retriever: {rag.bm25_retriever}")
    
    if rag.vector_store and rag.bm25_retriever:
        print("Testing search...")
        results = rag.hybrid_search("tomato", k=2)
        print(f"Search results: {len(results)}")
        for r in results:
            print(f" - {r.metadata.get('source')}: {r.page_content[:50]}...")
    else:
        print("INDICES FALIED TO LOAD")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
