index = hnswlib.Index(space='cosine', dim=len(title_embeddings[0]))
index.init_index(max_elements=len(title_embeddings), ef_construction=200, M=16)
index.add_items(title_embeddings)
