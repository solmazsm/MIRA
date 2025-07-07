
pdf_to_index = pdf_subset.set_index(["id"], drop=False)
id_index = np.array(pdf_to_index.id.values).flatten().astype("int")
content_encoded_normalized = faiss_title_embedding.copy()
faiss.normalize_L2(content_encoded_normalized)

index_content = faiss.IndexIDMap(faiss.IndexFlatIP(len(faiss_title_embedding[0])))
index_content.add_with_ids(content_encoded_normalized, id_index)


def multi_vector_search(queries, index, pdf_to_index, k=3):
    results = []
    for query_text in queries:
        # Encode the query using the model
        query_vector = model.encode([query_text])
        faiss.normalize_L2(query_vector)

        
        top_k = index.search(query_vector, k)

        # Retrieve and process results
        ids = top_k[1][0].tolist()
        similarities = top_k[0][0].tolist()
        docs = pdf_to_index.loc[ids]
        docs["similarities"] = similarities
        results.append(docs)
    return results


def train_and_evaluate(pretrained_model, index_dimension, similarity_threshold):


    queries = ["animal", "space", "science"]
    search_results = multi_vector_search(queries, index_content, pdf_to_index)


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

