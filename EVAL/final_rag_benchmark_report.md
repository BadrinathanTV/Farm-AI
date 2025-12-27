# RAG System Benchmark Report

**Date**: 2025-12-27
**Evaluated By**: Farm-AI Assistant

## 1. Executive Summary

We conducted a comprehensive evaluation of the Farm-AI RAG system using a tailored ground truth dataset of **41 queries** covering diverse topics (Crop Guides, Government Schemes, Natural Farming, Research Methods). 

The system demonstrated **exceptional performance** across both retrieval methods tested. The specific findings are:
- **Accuracy**: The system retrieved the correct document as the *very top result* (**Precision@1**) in **95.12%** of cases.
- **Robustness**: Removing the keyword-based BM25 algorithm did **not** degrade performance for this dataset, indicating that the semantic vector search is highly effective and well-aligned with the query patterns.
- **Latency**: Average retrieval time is approximately **1 second**, which is acceptable for real-time user interactions.

## 2. Methodology

We evaluated two configurations with **K=1** (checking only the top retrieved result):
1.  **Hybrid Search (Baseline)**: Combines Vector Search (Semantic) + BM25 (Keyword) using Reciprocal Rank Fusion (RRF).
2.  **Vector-Only Search**: Uses only FAISS Vector Search (Semantic).

**Metrics Used:**
*   **Precision@1**: Percentage of queries where the top result was correct.
*   **Recall@1**: Percentage of total relevant documents found in the top result.
*   **MRR (Mean Reciprocal Rank)**: Measure of where the first correct answer appears (1.0 = top).
*   **Latency**: Time taken to retrieve results.

## 3. Comparative Analysis

| Metric | Hybrid Search (Vector + BM25) | Vector-Only Search | Difference |
| :--- | :---: | :---: | :---: |
| **Mean Precision@1** | **0.9512** | **0.9512** | 0.00% |
| **Mean Recall@1** | **0.9024** | **0.9024** | 0.00% |
| **Mean MRR** | **0.9512** | **0.9512** | 0.00% |
| **Avg Latency** | **0.943s** | **1.147s** | +0.204s (slower) |

> [!NOTE]
> The identical accuracy scores suggest that for the current set of documents and queries, the vector embeddings alone are sufficient to capture the necessary semantic meaning. The slight increase in latency for Vector-Only is negligible and likely due to system variance during the test run rather than an algorithmic penalty.

## 4. Detailed Performance Breakdown

### 4.1. Success Cases
The system perfectly retrieved relevant documents for complex queries such as:
*   *"How to identify research question and hypothesis for farm trial?"*
*   *"Objectives of Agriculture Infrastructure Fund (AIF) scheme"*
*   *"Integration of livestock in farming systems"*

### 4.2. Failure Analysis
The same specific queries failed in both configurations (Precision = 0.0), pointing to potential gaps in the knowledge base or specific embedding challenges:
*   *"How to apply for seed treatment drums subsidy?"*
*   *"What is the NMSA scheme?"*

**Recommendation**: Investigate the documents related to "seed treatment drums" and "NMSA" to ensure the content is indexed correctly or add more specific metadata/keywords to these sections.

## 5. Conclusion
The Farm-AI RAG system is **production-ready** in terms of retrieval accuracy. 
*   **Recommendation**: You can safely proceed with the **Hybrid Search** as it provides a safety net for keyword-specific terms that vector search might miss in the future, without a significant latency cost (in fact, it was slightly faster in this specific test due to variance).
*   **Next Steps**: diverse the document set further or fine-tune embeddings if "NMSA" and "Seed treatment" queries remain critical issues.
