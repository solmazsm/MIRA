// coarse_fine_multivector.cpp
// Coarse-to-Fine Multi-Vector Retrieval Logic for VectorSearch

#include "coarse_fine_multivector.h"
#include <iostream>
#include <fstream>
#include <algorithm>

// === VectorSearchRefiner Implementation ===

VectorSearchRefiner::VectorSearchRefiner(int dim, int M)
    : dimension(dim), hnsw(dim, M) {
    hnsw.hnsw.efSearch = 64;  // Default efSearch; can be modified externally
}

void VectorSearchRefiner::add(const float* vector) {
    hnsw.add(vector);
}

// === Main Multi-Vector Retrieval Logic ===

void runMultiVectorRetrieval(const Preprocess& prep, int k, const std::vector<int>& query_ids, const std::string& result_file_path) {
    std::cout << "\n[VectorSearch::MultiVector] Coarse-to-Fine Retrieval Starting...\n";

    std::vector<std::vector<float>> multi_query;
    for (int qid : query_ids) {
        multi_query.emplace_back(prep.data.query[qid], prep.data.query[qid] + prep.data.dim);
    }

    // Coarse scoring: average L2 to all vectors
    std::vector<std::pair<float, int>> score_all(prep.data.N);
    for (int i = 0; i < prep.data.N; ++i) {
        float total_dist = 0.0f;
        for (const auto& qvec : multi_query) {
            total_dist += cal_distSqrt(qvec.data(), prep.data.val[i], prep.data.dim);
        }
        score_all[i] = { total_dist / multi_query.size(), i };
    }
    std::sort(score_all.begin(), score_all.end());

    // Select top coarse_k
    int coarse_k = 100;
    std::vector<int> top_candidates;
    for (int i = 0; i < std::min(coarse_k, (int)score_all.size()); ++i) {
        top_candidates.push_back(score_all[i].second);
    }

    // Fine reranking 
    VectorSearchRefiner refiner(prep.data.dim, 32);  // M = 32
    for (int id : top_candidates) {
        refiner.add(prep.data.val[id]);
    }

    // Query using the multi-vector group 
    std::vector<std::pair<float, int>> reranked;
    for (int i = 0; i < prep.data.N; ++i) {
        float total_dist = 0.0f;
        for (const auto& qvec : multi_query) {
            total_dist += cal_distSqrt(qvec.data(), prep.data.val[i], prep.data.dim);
        }
        reranked.emplace_back(total_dist / multi_query.size(), i);
    }
    std::sort(reranked.begin(), reranked.end());

    
    std::ofstream fout(result_file_path);
    for (int i = 0; i < k && i < reranked.size(); ++i) {
        fout << query_ids[0] << " " << reranked[i].second << "\n";
    }
    fout.close();

    std::cout << "[VectorSearch::MultiVector] Top-" << k << " neighbors written to: " << result_file_path << "\n";
}
