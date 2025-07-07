def faiss_benchmark(data_array):
    dimension = data_array.shape[1]  # Dimension of the vectors
    nlist = 100  # Number of clusters
    nprobe = 10  # Number of clusters to explore at query time

    # Initialize FAISS index
    index = faiss.IndexFlatL2(dimension)
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    index.nprobe = nprobe

    # Train the index
    index.train(data_array)

    # Add vectors to the index
    index.add(data_array)

    query_vector = np.random.rand(1, dimension).astype(np.float32)

    
    k = 5  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_vector, k)

    return distances, indices
