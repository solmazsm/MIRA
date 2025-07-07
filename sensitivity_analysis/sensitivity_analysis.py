def sensitivity_analysis(index_dimensions, similarity_thresholds, pretrained_models):
    results = []
    for model in pretrained_models:
        for dimension in index_dimensions:
            for threshold in similarity_thresholds:
                start_time = time.time()  # Record start time
                retrieved_documents = multi_vector_search(['SCIENCE', 'TECHNOLOGY', 'HEALTH'], index_content, pdf_to_index)
                relevant_topics = ['SCIENCE', 'TECHNOLOGY', 'HEALTH']
                relevant_documents = pdf[pdf['topic'].isin(relevant_topics)]
                precision, recall, query_time = evaluate_retrieval_results(retrieved_documents, relevant_documents, 0)
                end_time = time.time()  # Record end time
                query_time = end_time - start_time  # Calculate query time
                results.append({
                    "pretrained_model": model,
                    "index_dimension": dimension,
                    "similarity_threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "query_time": query_time
                })
    return results
