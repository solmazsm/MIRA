# Get query times
query_times = results_df['query_time'].values


sorted_query_times = np.sort(query_times)

#cumulative distribution function
ecdf = np.arange(1, len(sorted_query_times) + 1) / len(sorted_query_times)
