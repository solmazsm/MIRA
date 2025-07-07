

cache_dir = r"C:\Users\Soli1\.cache"
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)
faiss_title_embedding = model.encode(pdf_subset.title.values.tolist())
hnswlib_title_embedding = faiss_title_embedding  # Using the same embeddings for HNSWlib


faiss_index = faiss.IndexFlatIP(len(faiss_title_embedding[0]))
faiss_index.add(faiss_title_embedding)
dim = len(hnswlib_title_embedding[0])
num_elements = len(hnswlib_title_embedding)
hnsw_index = hnswlib.Index(space='cosine', dim=dim)
hnsw_index.init_index(max_elements=num_elements, ef_construction=200, M=16)
hnsw_index.add_items(hnswlib_title_embedding)

param_grid = {
    'pretrained_model': ["all-MiniLM-L6-v2", "roberta-base", "bert-base-uncased"],
    'index_type': ['FAISS', 'HNSWlib'],
    'index_dimension': [256, 512, 1024],
    'similarity_threshold': [0.7, 0.8, 0.9]
}

param_combinations = list(ParameterGrid(param_grid))

results = []
for params in param_combinations:
    result = train_and_evaluate(**params)
    results.append(params | result)  # Combine dictionaries

