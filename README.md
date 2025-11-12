QLEGAL – Legal Document Retrieval System

QLEGAL is a legal search engine built to help users find relevant sections from Indian Supreme Court judgments more efficiently. The system processes over 27,000 PDF judgments, extracts and cleans the text, divides long documents into manageable sections, and makes them searchable.

The retrieval pipeline begins with BM25, which provides fast keyword-based matching to identify the most relevant sections for a given legal query. These initial results are then passed to a fine-tuned LegalBERT model, which re-ranks them based on semantic relevance to improve contextual matching. A custom dataset of 235 query–document pairs was created through AI-assisted generation and manual verification to support training and evaluation.

The full workflow includes PDF extraction, preprocessing, indexing, retrieval with BM25, and re-ranking with LegalBERT, followed by evaluation using metrics such as Precision@5, Recall@5, MRR and nDCG. While BM25 performed strongly, the re-ranking model’s effectiveness was limited by dataset size and domain mismatch, highlighting areas for future improvement.

More details, including the trained models and dataset used, can be found here:


Trained models: https://www.kaggle.com/models/santos44/ir_qlegal/


Dataset: https://www.kaggle.com/datasets/santos44/ir2025-qlegal/data
