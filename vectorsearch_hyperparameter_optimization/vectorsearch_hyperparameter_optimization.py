
model = SentenceTransformer("all-MiniLM-L6-v2")

pdf_embeddings = model.encode(pdf_subset.title.values.tolist())

def calculate_recall_at_k(relevant_indices, retrieved_indices, k=10):
    relevant_set = set(relevant_indices)
    retrieved_set = set(retrieved_indices[:k])
    intersection = relevant_set.intersection(retrieved_set)
    recall_at_k = len(intersection) / len(relevant_set)
    return recall_at_k


def train_and_evaluate(pdf_embeddings, pretrained_model, index_dimension, similarity_threshold):
    # Train the index
    index = NearestNeighbors(n_neighbors=10, metric='cosine')  # Set n_neighbors to 10
    index.fit(pdf_embeddings)
    
   
    precision_scores = []
    recall_at_10_scores = []  # Change to recall_at_10_scores
    query_times = []
    for query_embedding in pdf_embeddings:
        start_time = time.time()
        distances, indices = index.kneighbors([query_embedding])
        end_time = time.time()
        
        # Calculate precision and recall@10
        # Calculate precision and recall@10
        nearest_indices = indices[0]
        relevant_indices = nearest_indices  # Use nearest_indices directly
        retrieved_indices = nearest_indices.tolist()  # Convert to list
        precision = precision_score(relevant_indices, retrieved_indices, average='weighted', zero_division=0)


        recall_at_10 = calculate_recall_at_k(relevant_indices, retrieved_indices)
        
     
        precision_scores.append(precision)
        recall_at_10_scores.append(recall_at_10)  
        query_times.append(end_time - start_time)
    
    mean_precision = np.mean(precision_scores)
    mean_recall_at_10 = np.mean(recall_at_10_scores)  # Change to mean_recall_at_10
    mean_query_time = np.mean(query_times)
    
    return {'precision': mean_precision, 'recall@10': mean_recall_at_10, 'query_time': mean_query_time}


param_grid = {
    'pretrained_model': ["all-MiniLM-L6-v2"],
    'index_dimension': [256, 512, 1024, 2048],  
    'similarity_threshold': [0.6, 0.7, 0.8, 0.9, 0.95]  
}


param_combinations = list(ParameterGrid(param_grid))


results_pdf = []
for params in param_combinations:
    result = train_and_evaluate(pdf_embeddings, **params)
    results_pdf.append({**params, **result})


