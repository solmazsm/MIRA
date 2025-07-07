  # Encode 
    title_embeddings = encode_document_titles(pdf_subset, model)
    
    # HNSWlib index
    index_hnswlib = create_hnswlib_index(title_embeddings)
    
    #FAISS index
    index_faiss = create_faiss_index(title_embeddings)
    
    #queries
    evaluation_queries = ["science", "technology", "health", "environment", "business", "politics", "sports", "entertainment"]
    
   
    results_hnswlib, results_faiss = evaluate_performance_for_queries(evaluation_queries, index_hnswlib, index_faiss, model, pdf_subset)
    
    
if __name__ == "__main__":
    main()
