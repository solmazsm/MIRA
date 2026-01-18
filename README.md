
## MIRA: A Self-Optimizing System for Scalable and Efficient Multi-Vector Retrieval


[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C++-supported-blue.svg)](https://en.cppreference.com/)


---

#### This folder contains all the necessary implementations required to replicate the studies.

### Documentation

https://solmazsm.github.io/VectorSearch/

The VectorSearch (VS) system includes comprehensive documentation for setup, usage, and evaluation. It provides:

- **Getting Started tutorials** for both Python and C++ implementations  
- **Installation instructions** and environment setup guides  
- **Step-by-step search examples** using multi-vector queries  
- **API reference** for system modules and dataset loaders  
- **Performance benchmarks** and comparison studies on real datasets

## Dataset

The VectorSearch dataset is a collection of news articles and image feature vectors indexed using semantic vector embeddings for efficient search and retrieval. It includes articles from diverse news sources and visual feature datasets, supporting large-scale text and image retrieval experiments.

---

| **Dataset**                   | **Description**                                                                                         | **Link**                                                                                  |
|------------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **All the News**             | 2,688,878 articles from 27 U.S. publications (2016–2020); expanded from original 2017 dataset (~100k)   | [All the News](https://components.one/datasets/all-the-news-2-news-articles-dataset)     |
| **NewsCatcher**              | 108k+ articles across 8 topics: business, entertainment, health, nation, science, sports, tech, world   | [NewsCatcher](https://www.newscatcherapi.com/)                                           |
| **News Dataset (API)**       | Subset of NewsCatcher, organized by topic                                                              | [NewsCatcher](https://www.newscatcherapi.com/)                                           |
| **Glove1.2M**                | Pretrained word embeddings (1.2M vectors) for large-scale retrieval                                     | [Glove1.2M](http://corpus-texmex.irisa.fr/)                          |
| **Deep1M**                   | 1 million dense vectors for benchmarking similarity search                                              | [Deep1M](http://corpus-texmex.irisa.fr/)                            |
| **SIFT10M / SIFT1M**         | Visual feature vectors from image datasets                                                              | [SIFT Dataset](http://corpus-texmex.irisa.fr/)                                           |
                              

In our system, the input file (e.g., `audio.data`) is a binary file containing vectors in 32-bit floating-point (`float32`) format. Compared to text-based formats, binary files enable significantly faster read times and lower storage overhead, rendering them particularly effective for scalable and memory-efficient retrieval in high-dimensional vector spaces.

We extended this support to handle multi-vector inputs per document, enabling retrieval over sets of semantic embeddings instead of single vectors. Example input files such as [`audio.data`](https://github.com/RSIA-LIESMARS-WHU/LSHBOX-sample-data) are stored in the [`float32` data type](https://github.com/RSIA-LIESMARS-WHU/LSHBOX?tab=readme-ov-file), commonly used for efficient vector storage.


---
### Download Instructions

Use the `prepare_dataset.py` script to download and preprocess datasets:

```bash
python scripts/prepare_dataset.py --dataset all_the_news
python scripts/prepare_dataset.py --dataset newscatcher
```
## Evaluation Metrics

- **Query Throughput (QPS)**  
- **Memory Usage**  
- **Recall@10**
- **Recall@100**  
- **Mean Precision**  
- **Query Time**

---

##  Key Features

- **Multi-vector document embedding** using transformer-based models (e.g., MiniLM, BERT, RoBERTa)
- **Hybrid indexing** with (coarse retrieval) and (fine reranking)
- **Dynamic hyperparameter tuning** using grid search
- **Scalable evaluation** on benchmark datasets: News, Glove1.2M, Deep1M, and SIFT10M
- **Caching support** for efficient inference and reduced computation
## Experimental Setup

VectorSearch was tested in both local and cloud environments to validate scalability and performance.

- **Local Environment:**  
  Intel Core i5-11400H CPU @ 2.70GHz  
  16 GB DDR4 RAM  
  NVIDIA RTX 3050 GPU (4 GB VRAM)  
  Ubuntu 22.04.3, Python 3.10

- **Cloud Environment (Azure):**  
  Standard_E64ds_v4 Virtual Machine  
  64 vCPUs, 504 GiB RAM  
  Intel Xeon Platinum 8272CL  
  Ubuntu 22.04, Python 3.10

All embeddings were computed using [SentenceTransformer](https://www.sbert.net/) (v2.2.2).
VectorSearch introduces a novel hybrid indexing system that coarse retrieval via quantization-based filtering with lightweight graph refinement, enabling scalable, memory-efficient, and dynamic multi-vector retrieval across large embedding spaces.

Retrieval experiments were performed using 10,000 queries, and all reported metrics represent the average of five independent runs.

Scripts for local and cloud deployment are included for reproducibility.

### Requirements

- C++17 compiler (`g++`, `clang++`, MSVC)
- `make` or `nmake` (Linux/macOS or Windows)


### Compilation (Linux/macOS)

```bash
make
```
```bash
./VectorSearch deep1m
```


