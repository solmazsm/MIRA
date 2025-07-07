

pdf_to_index = pdf_subset.set_index(["id"], drop=False)
id_index = np.array(pdf_to_index.id.values).flatten().astype("int")
content_encoded_normalized = faiss_title_embedding.copy()
faiss.normalize_L2(content_encoded_normalized)

index_content = faiss.IndexIDMap(faiss.IndexFlatIP(len(faiss_title_embedding[0])))
index_content.add_with_ids(content_encoded_normalized, id_index)

def train_and_evaluate(pretrained_model, index_dimension, similarity_threshold):
  

    
    print(f"Training and evaluating with parameters: "
          f"pretrained_model={pretrained_model}, "
          f"index_dimension={index_dimension}, "
          f"similarity_threshold={similarity_threshold}")
    print(f"Precision: {precision}, Recall: {recall}, Query Time: {query_time} seconds")

    return precision, recall, query_time


pretrained_models = ["all-MiniLM-L6-v2", "roberta-base", "bert-base-uncased"]
index_dimensions = [256, 512, 1024]
similarity_thresholds = [0.7, 0.8, 0.9]

results = []


for model in pretrained_models:
    for dimension in index_dimensions:
        for threshold in similarity_thresholds:
            # Train and evaluate the retrieval system with the current parameters
            precision, recall, query_time = train_and_evaluate(model, dimension, threshold)

            # Store the evaluation results
            results.append({
                "pretrained_model": model,
                "index_dimension": dimension,
                "similarity_threshold": threshold,
                "precision": precision,
                "recall": recall,
                "query_time": query_time
            })


