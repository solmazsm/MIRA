def evaluate_performance(ground_truth, retrieved_docs):
    retrieved_ids = retrieved_docs.index.tolist()
    relevant_ids = ground_truth
    intersection = set(retrieved_ids).intersection(set(relevant_ids))
    precision = len(intersection) / len(retrieved_ids) if len(retrieved_ids) > 0 else 0
    recall = len(intersection) / len(relevant_ids) if len(relevant_ids) > 0 else 0
    return precision, recall
