def evaluate_performance(ground_truth, retrieved_docs):
    retrieved_ids = retrieved_docs.index.tolist()
    relevant_ids = ground_truth
    intersection = set(retrieved_ids).intersection(set(relevant_ids))
    precision = len(intersection) / len(retrieved_ids) if len(retrieved_ids) > 0 else 0
    recall = len(intersection) / len(relevant_ids) if len(relevant_ids) > 0 else 0
    return precision, recall

# Evaluate HNSWlib-based search
hnswlib_results = []
for query, relevant_ids in ground_truth.items():
    similar_docs_hnswlib = search_similar_documents_hnswlib(query, index_hnswlib, pdf_subset)
    precision, recall = evaluate_performance(relevant_ids, similar_docs_hnswlib)
    hnswlib_results.append({'query': query, 'precision': precision, 'recall': recall})
