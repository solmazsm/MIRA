
faiss_title_embedding = model.encode(pdf_subset.title.values.tolist())


pdf_to_index = pdf_subset.set_index(["id"], drop=False)
id_index = np.array(pdf_to_index.index.values).flatten().astype("int")

content_encoded_normalized = faiss_title_embedding.copy()
faiss.normalize_L2(content_encoded_normalized)

index_content = faiss.IndexIDMap(faiss.IndexFlatIP(len(faiss_title_embedding[0])))
index_content.add_with_ids(content_encoded_normalized, id_index)


def multi_vector_search(queries, index, pdf_to_index, k=3):
    results = []
    for query_text in queries:
        
        query_vector = model.encode([query_text])
        faiss.normalize_L2(query_vector)

      
        top_k = index.search(query_vector, k)

        
        ids = top_k[1][0].tolist()
        similarities = top_k[0][0].tolist()
        docs = pdf_to_index.loc[ids]
        docs["similarities"] = similarities
        results.append(docs)
    return results


