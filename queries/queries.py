queries = ["animal", "space", "science"]
search_results = multi_vector_search(queries, index_content, pdf_to_index)
for idx, query_result in enumerate(search_results):
    print(f"Results for query '{queries[idx]}':")
    print(query_result)
    print("\n")
