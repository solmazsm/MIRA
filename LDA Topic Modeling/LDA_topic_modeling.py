# title: efficient_retrieval
# Tokenize 
vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2, lowercase=True)
doc_term_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()


dictionary = corpora.Dictionary([doc.split() for doc in documents])
corpus = [dictionary.doc2bow(doc.split()) for doc in documents]

# Perform LDA topic modeling
lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=5, passes=15)


topic_names = ["Politics", "Technology", "Health", "Finance", "Environment"]
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")


topic_distribution = [lda_model.get_document_topics(doc) for doc in corpus]

# Extract topic probabilities 
topic_probabilities = np.zeros((len(topic_distribution), lda_model.num_topics))
for i, dist in enumerate(topic_distribution):
    for topic_idx, prob in dist:
        topic_probabilities[i, topic_idx] = prob
