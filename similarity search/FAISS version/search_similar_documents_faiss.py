def search_similar_documents_faiss(query_text, index, pdf_subset, k=3):
    query_vector = model.encode([query_text])[0]
    _, labels = index.search(query_vector.reshape(1, -1), k)
    similar_documents = pdf_subset.iloc[labels[0]].reset_index(drop=True)
    return similar_documents
