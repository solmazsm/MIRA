#vector_search_evaluation
def calculate_recall_at_k(relevant_indices, retrieved_indices, k=10):
    relevant_set = set(relevant_indices)
    retrieved_set = set(retrieved_indices[:k])
    intersection = relevant_set.intersection(retrieved_set)
    recall_at_k = len(intersection) / len(relevant_set) if relevant_set else 0
    return recall_at_k


def train_and_evaluate_best_params(pdf_embeddings):
    # Train the index
    index = NearestNeighbors(n_neighbors=100, metric='cosine')
    index.fit(pdf_embeddings)
    
    # Simulate query for each document in pdf
    precision_scores = []
    recall_at_1_scores = []
    recall_at_10_scores = []
    recall_at_100_scores = []
    query_times = []
    for query_embedding in pdf_embeddings:
        start_time = time.time()
        distances, indices = index.kneighbors([query_embedding])
        end_time = time.time()
        
        
        nearest_indices = indices[0]
        relevant_indices = nearest_indices
        retrieved_indices = nearest_indices.tolist()
        precision = len(set(relevant_indices).intersection(set(retrieved_indices))) / len(retrieved_indices)
        recall_at_1 = calculate_recall_at_k(relevant_indices, retrieved_indices, k=1)
        recall_at_10 = calculate_recall_at_k(relevant_indices, retrieved_indices, k=10)
        recall_at_100 = calculate_recall_at_k(relevant_indices, retrieved_indices, k=100)
        
     
        precision_scores.append(precision)
        recall_at_1_scores.append(recall_at_1)
        recall_at_10_scores.append(recall_at_10)
        recall_at_100_scores.append(recall_at_100)
        query_times.append(end_time - start_time)
    
    
    mean_recall_at_1 = np.mean(recall_at_1_scores)
    mean_recall_at_10 = np.mean(recall_at_10_scores)
    mean_recall_at_100 = np.mean(recall_at_100_scores)
    mean_query_time = np.mean(query_times)
    
    return {
        'precision': mean_precision, 
        'recall@1': mean_recall_at_1, 
        'recall@10': mean_recall_at_10, 
        'recall@100': mean_recall_at_100, 
        'query_time': mean_query_time
    }

