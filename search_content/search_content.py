def search_content(queries, model, index_content, pdf_to_index, k=3):
   
    ids = results['ids'][0]
    similarities = results['distances'][0]
    topics = [metadata['topic'] for metadata in results['metadatas'][0]]
    documents = results['documents'][0]

   
    df = pd.DataFrame({'id': ids, 'similarities': similarities, 'topic': topics, 'document': documents})

   
    df["similarities"] = pd.to_numeric(df["similarities"], errors='coerce')

   
    aggregated_results = df.groupby("id").mean().sort_values(by="similarities", ascending=False).head(k)

    return aggregated_results
