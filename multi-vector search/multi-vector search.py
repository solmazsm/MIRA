def search_content(queries, pdf_to_index, k=3):
    all_results = []
    for query in queries:
        query_vector = model.encode([query])
        faiss.normalize_L2(query_vector)

        # each query
        top_k = index_content.search(query_vector, k)
        ids = top_k[1][0].tolist()
        similarities = top_k[0][0].astype(float).tolist()

        # Retrieve 
        results = pdf_to_index.loc[ids]
        results["similarities"] = similarities
        all_results.append(results)


    combined_results = pd.concat(all_results)

    combined_results["similarities"] = pd.to_numeric(combined_results["similarities"])

   
    aggregated_results = combined_results.groupby(combined_results.index).mean().sort_values(by="similarities", ascending=False).head(k)

    return aggregated_results
