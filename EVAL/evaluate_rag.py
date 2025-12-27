import json
import time
import os
from typing import List, Dict
import numpy as np
from core.rag_service import RAGService
from rich.console import Console
from rich.table import Table

console = Console()

def load_ground_truth(filepath: str) -> List[Dict]:
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_metrics(retrieved_docs: List[str], relevant_docs: List[str], k: int):
    # retrieved_docs are filenames/sources
    # relevant_docs are filenames/sources
    
    # Precision@K
    # Intersection of retrieved and relevant / k
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    intersection = retrieved_set.intersection(relevant_set)
    precision = len(intersection) / k if k > 0 else 0
    
    # Recall@K
    # Intersection / total relevant
    recall = len(intersection) / len(relevant_set) if len(relevant_set) > 0 else 0
    
    # MRR
    # 1 / rank of first relevant doc
    mrr = 0
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_set:
            mrr = 1 / (i + 1)
            break
            
    return precision, recall, mrr

def evaluate(ground_truth_file: str = "ground_truth.json", k: int = 5):
    console.print(f"[bold blue]Starting RAG Evaluation (k={k})...[/bold blue]")
    
    try:
        ground_truth = load_ground_truth(ground_truth_file)
    except FileNotFoundError:
        console.print(f"[bold red]Error: {ground_truth_file} not found![/bold red]")
        return

    rag = RAGService()
    
    total_precision = 0
    total_recall = 0
    total_mrr = 0
    total_latency = 0
    
    results = []
    
    table = Table(title="Evaluation Details")
    table.add_column("Query", style="cyan", no_wrap=False)
    table.add_column("Retrieved (Top 1)", style="magenta")
    table.add_column("P@K", justify="right")
    table.add_column("R@K", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Latency (s)", justify="right")

    for item in ground_truth:
        query = item['query']
        relevant_docs = item['relevant_docs']
        
        start_time = time.time()
        # Get retrieved documents
        # RAGService returns Document objects with metadata 'source'
        retrieved_doc_objects = rag.hybrid_search(query, k=k)
        latency = time.time() - start_time
        
        # Extract source filenames from metadata
        retrieved_sources = [doc.metadata.get('source', '') for doc in retrieved_doc_objects]
        
        precision, recall, mrr = calculate_metrics(retrieved_sources, relevant_docs, k)
        
        total_precision += precision
        total_recall += recall
        total_mrr += mrr
        total_latency += latency
        
        top_result = retrieved_sources[0] if retrieved_sources else "None"
        
        table.add_row(
            query, 
            top_result, 
            f"{precision:.2f}", 
            f"{recall:.2f}", 
            f"{mrr:.2f}", 
            f"{latency:.3f}"
        )
        
        results.append({
            "query": query,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "mrr": mrr,
                "latency": latency
            }
        })

    console.print(table)
    
    num_queries = len(ground_truth)
    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries
    avg_mrr = total_mrr / num_queries
    avg_latency = total_latency / num_queries
    
    console.print("\n[bold green]Summary Results:[/bold green]")
    console.print(f"Mean Precision@{k}: {avg_precision:.4f}")
    console.print(f"Mean Recall@{k}:    {avg_recall:.4f}")
    console.print(f"Mean MRR:          {avg_mrr:.4f}")
    console.print(f"Avg Latency:       {avg_latency:.4f}s")
    
    # Save report
    report_content = f"""# RAG Evaluation Report

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**K**: {k}
**Number of Queries**: {num_queries}

## Summary Metrics
- **Mean Precision@{k}**: {avg_precision:.4f}
- **Mean Recall@{k}**: {avg_recall:.4f}
- **Mean MRR**: {avg_mrr:.4f}
- **Average Latency**: {avg_latency:.4f} seconds

## Detailed Results

| Query | Precision@{k} | Recall@{k} | MRR | Latency (s) |
|-------|--------------|-----------|-----|-------------|
"""
    for res in results:
        q = res['query']
        m = res['metrics']
        report_content += f"| {q} | {m['precision']:.4f} | {m['recall']:.4f} | {m['mrr']:.4f} | {m['latency']:.4f} |\n"
        
    with open("rag_evaluation_report.md", "w") as f:
        f.write(report_content)
    
    print("\nReport saved to rag_evaluation_report.md")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    evaluate(k=1)
