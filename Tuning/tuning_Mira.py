# ---- Adaptive Tuning Search ----
start_time_adaptive = time.time()
adaptive_results = []
adaptive_retries = 0

for q in xq:
    # Start with fast settings
    index_ivfpq.nprobe = 1
    _, D_fast, I_fast = index_ivfpq.search(np.expand_dims(q, axis=0), k)

    # If nearest distance too large, retry with higher settings
    if D_fast[0][0] > 100.0:  # Threshold can be adjusted
        adaptive_retries += 1
        index_ivfpq.nprobe = 5
        _, D_slow, I_slow = index_ivfpq.search(np.expand_dims(q, axis=0), k)
        adaptive_results.append(I_slow[0])
    else:
        adaptive_results.append(I_fast[0])

end_time_adaptive = time.time()
qps_adaptive = xq.shape[0] / (end_time_adaptive - start_time_adaptive)
recall_adaptive = compute_recall(adaptive_results, groundtruth, k)


metrics.update({
    "QPS (Adaptive)": round(qps_adaptive, 2),
    "Recall (Adaptive)": round(recall_adaptive, 4),
    "Time (s) (Adaptive)": round(end_time_adaptive - start_time_adaptive, 2),
    "Adaptive Retries": adaptive_retries
})
