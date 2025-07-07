

#include "coarse_fine_multivector.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

void runMultiVectorRetrieval(const Preprocess& prep, int k, const std::vector<int>& query_ids, const std::string& result_file_path) {
    std::cout << "\n[VectorSearch: Coarse-Fine Multi-Vector Retrieval] Starting...\n";

    std::vector<std::vector<float>> multi_query;
    for (int qid : query_ids) {
        multi_query.emplace_back(prep.data.query[qid], prep.data.query[qid] + prep.data.dim);
    }

    std::vector<std::pair<float, int>> coarse_scores(prep.data.N);
    for (int i = 0; i < prep.data.N; ++i) {
        float total_dist = 0.0f;
        for (const auto& qvec : multi_query) {
            total_dist += cal_distSqrt(qvec.data(), prep.data.val[i], prep.data.dim);
        }
        coarse_scores[i] = { total_dist / multi_query.size(), i };
    }

    std::sort(coarse_scores.begin(), coarse_scores.end());

    std::vector<int> coarse_top_k;
    int coarse_k = 100;
    for (int i = 0; i < std::min(coarse_k, (int)coarse_scores.size()); ++i) {
        coarse_top_k.push_back(coarse_scores[i].second);
    }

    VectorSearchCoarseIndexer coarse_indexer(prep.data.dim);
    VectorSearchRefiner refiner(prep.data.dim, 32);  // M = 32
    refiner.hnsw.efSearch = 64;

    for (int id : coarse_top_k) {
        refiner.add(prep.data.val[id]);  // add selected coarse candidates
    }

    std::vector<std::pair<float, int>> refined_scores;
    for (int i = 0; i < prep.data.N; ++i) {
        float total_dist = 0.0f;
        for (const auto& qvec : multi_query) {
            total_dist += cal_distSqrt(qvec.data(), prep.data.val[i], prep.data.dim);
        }
        refined_scores.push_back({ total_dist / multi_query.size(), i });
    }

    std::sort(refined_scores.begin(), refined_scores.end());

    std::ofstream fout(result_file_path);
    for (int i = 0; i < k; ++i) {
        fout << query_ids[0] << " " << refined_scores[i].second << "\n";
    }
    fout.close();

    std::cout << "Top-" << k << " neighbors written to: " << result_file_path << std::endl;
}
