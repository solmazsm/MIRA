
index_ivfpq.nprobe = nprobe
index_hnsw.hnsw.efSearch = efSearch

start = time.time()

multi_query_results = []
for q in xq:
    # Simulate multi-vector query (here, using just one vector per query for simplicity)
    D_coarse, I_coarse = index_ivfpq.search(np.expand_dims(q, axis=0), k)

   
    D_fine, I_fine = index_hnsw.search(np.expand_dims(q, axis=0), k)
    multi_query_results.append(I_fine)

aggregated_result = aggregate_multi_query(multi_query_results)

end = time.time()
print(f"Hybrid search completed in {end - start:.2f} seconds")

