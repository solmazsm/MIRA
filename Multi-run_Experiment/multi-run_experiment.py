

# ---------------------- Multi-run Experiment ----------------------
all_metrics = []

for nprobe in nprobe_list:
    for efSearch in efSearch_list:
        index_ivfpq.nprobe = nprobe
        index_hnsw.hnsw.efSearch = efSearch

        # ---- Per-Query Search ----
        start_mem = get_memory_usage_gb()
        start_time = time.time()
        multi_query_results = []
        for q in xq:
            _, I_coarse = index_ivfpq.search(np.expand_dims(q, axis=0), k)
            _, I_fine = index_hnsw.search(np.expand_dims(q, axis=0), k)
            multi_query_results.append(I_fine[0])
        end_time = time.time()
        end_mem = get_memory_usage_gb()
        mem_footprint = end_mem - start_mem
        qps = xq.shape[0] / (end_time - start_time)
        recall = compute_recall(multi_query_results, groundtruth, k)

     
        start_time_batch = time.time()
        _, I_coarse_batch = index_ivfpq.search(xq, k)
        _, I_fine_batch = index_hnsw.search(xq, k)
        end_time_batch = time.time()
        qps_batch = xq.shape[0] / (end_time_batch - start_time_batch)
        recall_batch = compute_recall(I_fine_batch, groundtruth, k)

        # ---- Multi-Vector Search ----
        start_time_pq = time.time()
        index_ivfpq.nprobe = nprobe
        _, I_pq = index_ivfpq.search(xq, k)
        end_time_pq = time.time()
        qps_pq = xq.shape[0] / (end_time_pq - start_time_pq)
        recall_pq = compute_recall(I_pq, groundtruth, k)

       
        start_time_adaptive = time.time()
        adaptive_results = []
        adaptive_retries = 0
        for q in xq:
            index_ivfpq.nprobe = 1
            _, D_fast, I_fast = index_ivfpq.search(np.expand_dims(q, axis=0), k)
            if D_fast[0][0] > 100.0:
                adaptive_retries += 1
                index_ivfpq.nprobe = 5
                _, D_slow, I_slow = index_ivfpq.search(np.expand_dims(q, axis=0), k)
                adaptive_results.append(I_slow[0])
            else:
                adaptive_results.append(I_fast[0])
        end_time_adaptive = time.time()
        qps_adaptive = xq.shape[0] / (end_time_adaptive - start_time_adaptive)
        recall_adaptive = compute_recall(adaptive_results, groundtruth, k)

        metrics = {
            "Scale": f"{xb.shape[0]//1_000_000}M",
            "nprobe": nprobe,
            "efSearch": efSearch,
            "Mem (GiB)": round(mem_footprint, 3),
            "QPS (Per-Query)": round(qps, 2),
            "Recall (Per-Query)": round(recall, 4),
            "Time (s) (Per-Query)": round(end_time - start_time, 2),
            "QPS (Batch)": round(qps_batch, 2),
            "Recall (Batch)": round(recall_batch, 4),
            "Time (s) (Batch)": round(end_time_batch - start_time_batch, 2),
            "QPS (PQ-MultiVector)": round(qps_pq, 2),
            "Recall (PQ-MultiVector)": round(recall_pq, 4),
            "Time (s) (PQ-MultiVector)": round(end_time_pq - start_time_pq, 2),
            "QPS (Adaptive)": round(qps_adaptive, 2),
            "Recall (Adaptive)": round(recall_adaptive, 4),
            "Time (s) (Adaptive)": round(end_time_adaptive - start_time_adaptive, 2),
            "Adaptive Retries": adaptive_retries
        }

        all_metrics.append(metrics)

