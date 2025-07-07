def evaluate_retrieval_results(retrieved_documents_list, relevant_documents, query_time):
    precision_scores = []
    recall_scores = []
    for retrieved_documents in retrieved_documents_list:
        retrieved_ids = set(retrieved_documents['id'])
        relevant_ids = set(relevant_documents['id'])
        true_positives = len(retrieved_ids.intersection(relevant_ids))
        precision = true_positives / len(retrieved_ids) if len(retrieved_ids) > 0 else 0
        recall = true_positives / len(relevant_ids) if len(relevant_ids) > 0 else 0
        precision_scores.append(precision)
        recall_scores.append(recall)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    return avg_precision, avg_recall, query_time
