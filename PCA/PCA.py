
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([example.texts[0] for example in faiss_train_examples])


unique_titles = []
unique_embeddings = []
unique_topics = []


for title, embedding in zip(pdf_subset['title'], embeddings):
    topic = pdf_subset[pdf_subset['title'] == title]['topic'].values[0]
    if title not in unique_titles:
        unique_titles.append(title)
        unique_embeddings.append(embedding)
        unique_topics.append(topic)


unique_embeddings = np.array(unique_embeddings)


pca = PCA(n_components=2) 
embeddings_2d = pca.fit_transform(unique_embeddings)


unique_topics_set = list(set(unique_topics))
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_topics_set)))


topic_color_map = {topic: color for topic, color in zip(unique_topics_set, colors)}


plt.figure(figsize=(10, 8))
for i, topic in enumerate(unique_topics):
    color = topic_color_map[topic]
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=color, label=topic, alpha=0.5)


handles = []
for topic in unique_topics_set:
    color = topic_color_map[topic]
    handles.append(plt.Line2D([], [], marker='o', color=color, linestyle='None', markersize=5, label=topic))
plt.legend(handles=handles, title='Topics', loc='upper left', fontsize='small')
