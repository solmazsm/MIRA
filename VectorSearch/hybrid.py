#FAISS index
faiss_index = faiss.IndexFlatIP(len(faiss_title_embedding[0]))
faiss_index.add(faiss_title_embedding)

# HNSWlib index
dim = len(hnswlib_title_embedding[0])
num_elements = len(hnswlib_title_embedding)
hnsw_index = hnswlib.Index(space='cosine', dim=dim)
hnsw_index.init_index(max_elements=num_elements, ef_construction=200, M=16)
hnsw_index.add_items(hnswlib_title_embedding)

# VectorSearch using both FAISS and HNSWlib
def vector_search(query_vector, index, k=3):
    if index == 'FAISS':
        D, I = faiss_index.search(query_vector, k)
    elif index == 'HNSWlib':
        labels, _ = hnsw_index.knn_query(query_vector, k)
        I = labels.tolist()
    else:
        raise ValueError("Invalid index type. Choose 'FAISS' or 'HNSWlib'.")
    return I

# train and evaluate the retrieval system
def train_and_evaluate(pretrained_model, index_type, index_dimension, similarity_threshold):
    results = []
    for query_text in pdf_subset['title'].head(3):  # Considering the first 3 titles for illustration
        query_vector = model.encode([query_text])
        start_time = time.time()
        retrieved_documents = vector_search(query_vector, index_type)
        end_time = time.time()
        query_time = end_time - start_time
