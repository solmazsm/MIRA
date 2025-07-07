## 📚 Dataset

The VectorSearch dataset is a collection of news articles and image feature vectors indexed using semantic vector embeddings for efficient search and retrieval. It includes articles from diverse news sources and visual feature datasets, supporting large-scale text and image retrieval experiments.

---

| **Dataset**                   | **Description**                                                                                         | **Link**                                                                                  |
|------------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **All the News**             | 2,688,878 articles from 27 U.S. publications (2016–2020); expanded from original 2017 dataset (~100k)   | [All the News](https://components.one/datasets/all-the-news-2-news-articles-dataset)     |
| **NewsCatcher**              | 108k+ articles across 8 topics: business, entertainment, health, nation, science, sports, tech, world   | [NewsCatcher](https://www.newscatcherapi.com/)                                           |
| **News Dataset (API)**       | Subset of NewsCatcher, organized by topic                                                              | [NewsCatcher](https://www.newscatcherapi.com/)                                           |
| **Glove1.2M**                | Pretrained word embeddings (1.2M vectors) for large-scale retrieval                                     | [Glove1.2M](https://github.com/solmazsm/gqr?tab=readme-ov-file)                          |
| **Deep1M**                   | 1 million dense vectors for benchmarking similarity search                                              | [Deep1M](https://github.com/solmazsm/gqr?tab=readme-ov-file)                             |
| **SIFT10M / SIFT1M**         | Visual feature vectors from image datasets                                                              | [SIFT Dataset](http://corpus-texmex.irisa.fr/)                                           |
               

In our system, the input file (e.g., `audio.data`) is a binary file containing vectors in 32-bit floating-point (`float32`) format. Compared to text-based formats, binary files enable significantly faster read times and lower storage overhead.

We extended this support to handle multi-vector inputs per document, enabling retrieval over sets of semantic embeddings instead of single vectors. Example input files such as [`audio.data`](https://github.com/RSIA-LIESMARS-WHU/LSHBOX-sample-data) are stored in the [`float32` data type](https://github.com/RSIA-LIESMARS-WHU/LSHBOX?tab=readme-ov-file), commonly used for efficient vector storage.

