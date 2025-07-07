
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <string>
#include "alg.h" 

// Computes average recall@10 using a result file and ground truth
float calculate_average_recall_at_10(const std::string& result_file, const Preprocess& prep, const std::vector<int>& query_ids) {
    std::ifstream fin(result_file);
    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open result file: " << result_file << std::endl;
        return -1.0f;
    }

    std::unordered_map<int, std::vector<int>> retrieved_neighbors;
    int query_id, neighbor_id;
    while (fin >> query_id >> neighbor_id) {
        retrieved_neighbors[query_id].push_back(neighbor_id);
    }
    fin.close();

    float total_recall = 0.0f;
    for (auto qid : query_ids) {
        auto& retrieved = retrieved_neighbors[qid];
        auto ground_truth = prep.benchmark.indice[qid];
        int correct = 0;
        int top_k = std::min(10, (int)retrieved.size());
        for (int i = 0; i < top_k; ++i) {
            int retrieved_neighbor = retrieved[i];
            for (int j = 0; j < 100; ++j) {
                if (retrieved_neighbor == ground_truth[j]) {
                    ++correct;
                    break;
                }
            }
        }
        total_recall += static_cast<float>(correct) / top_k;
    }
    return total_recall / query_ids.size();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_name>" << std::endl;
        return 1;
    }

    std::string datasetName = argv[1];
    std::string data_fold = "E:/Dataset_for_c/";

    Preprocess prep(data_fold + datasetName + ".data", data_fold + "ANN/" + datasetName + ".bench_graph");

    std::cout << "\nRunning Multi-Vector Retrieval (Aggregated)...\n";
    std::vector<int> query_ids = {0, 1, 2, 3, 4};
    std::vector<std::vector<float>> multi_query;

    for (int qid : query_ids) {
        multi_query.emplace_back(prep.data.query[qid], prep.data.query[qid] + prep.data.dim);
    }

    std::vector<std::pair<float, int>> score_all(prep.data.N);
    for (int i = 0; i < prep.data.N; ++i) {
        float total_dist = 0.0f;
        for (const auto& qvec : multi_query) {
            total_dist += cal_distSqrt(qvec.data(), prep.data.val[i], prep.data.dim);
        }
        score_all[i] = { total_dist / multi_query.size(), i };
    }

    std::sort(score_all.begin(), score_all.end());

    int k = 10;
    std::cout << "Top-" << k << " Neighbors (Multi-Vector Aggregation):\n";
    std::ofstream fout("result.txt");
    for (int i = 0; i < k; ++i) {
        std::cout << i << ": ID = " << score_all[i].second << ", AvgDist = " << score_all[i].first << "\n";
        fout << query_ids[0] << " " << score_all[i].second << "\n";  // Write for the first query only
    }
    fout.close();

    float avg_recall_at_10 = calculate_average_recall_at_10("result.txt", prep, query_ids);
    std::cout << "=========================================\n";
    std::cout << "Average Recall@10 over multi-vector queries: " << avg_recall_at_10 << std::endl;
    std::cout << "=========================================\n";

    return 0;
}
