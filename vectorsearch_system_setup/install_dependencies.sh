# === System Dependencies ===
sudo apt-get update && sudo apt-get install -y \
  build-essential cmake g++ make libomp-dev \
  python3 python3-pip python3-venv python3-dev

# === Python Packages ===
pip3 install --upgrade pip
pip3 install numpy faiss-cpu sentence-transformers

# === Clone and Run VectorSearch ===
git clone https://github.com/solmazsm/VectorSearch.git
cd VectorSearch

# === Run VectorSearch with a sample query ===
python3 multi_vector_search/run_vectorsearch.py \
  --query_file queries/sample.txt \
  --index_dir index/

# === Evaluate Retrieval Performance ===
python3 evaluate_performance_for_queries/eval_precision.py \
  --groundtruth_file data/groundtruth.json \
  --retrieved_file results/output.json
