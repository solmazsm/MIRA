def search_similar_documents_hnswlib(query_text, index, pdf_subset, k=3):
    query_vector = model.encode([query_text])
    labels, distances = index.knn_query(query_vector, k=k)
    similar_documents = pdf_subset.iloc[labels[0]].reset_index(drop=True)
    similar_documents['distance'] = distances[0]
    return similar_documents
