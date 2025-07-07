for query in queries:
    hnswlib_precision, hnswlib_recall = compute_precision_recall(hnswlib_results[query], relevant_documents[query])
    faiss_precision, faiss_recall = compute_precision_recall(faiss_results[query], relevant_documents[query])
    print(f"HNSWlib Results for '{query}': Precision: {hnswlib_precision}, Recall: {hnswlib_recall}")
    print(f"FAISS Results for '{query}': Precision: {faiss_precision}, Recall: {faiss_recall}")
