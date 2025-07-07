

model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)
faiss_title_embedding = model.encode(pdf_subset.title.values.tolist())


pdf_to_index = pdf_subset.set_index(["id"], drop=False)
id_index = np.array(pdf_to_index.id.values).flatten().astype("int")
content_encoded_normalized = faiss_title_embedding.copy()
faiss.normalize_L2(content_encoded_normalized)

index_content = faiss.IndexIDMap(faiss.IndexFlatIP(len(faiss_title_embedding[0])))
index_content.add_with_ids(content_encoded_normalized, id_index)

param_grid = {
    'pretrained_model': ["all-MiniLM-L6-v2", "roberta-base", "bert-base-uncased"],
    'index_dimension': [256, 512, 1024],
    'similarity_threshold': [0.7, 0.8, 0.9]
}
