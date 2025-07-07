def evaluate_performance_for_queries(queries, index_hnswlib, index_faiss, model, pdf_subset):
    results_hnswlib = []
    results_faiss = []
    
    for query in queries:
        query_vector = model.encode([query])
        
        # Search using HNSWlib
        similar_docs_hnswlib = search_hnswlib_index(query_vector, index_hnswlib, pdf_subset)
        precision_hnswlib, recall_hnswlib = calculate_precision_recall(similar_docs_hnswlib)
        results_hnswlib.append({'query': query, 'precision': precision_hnswlib, 'recall': recall_hnswlib})
        
        # Search using FAISS
        similar_docs_faiss = search_faiss_index(query_vector, index_faiss, pdf_subset)
        precision_faiss, recall_faiss = calculate_precision_recall(similar_docs_faiss)
        results_faiss.append({'query': query, 'precision': precision_faiss, 'recall': recall_faiss})
    
    return results_hnswlib, results_faiss
