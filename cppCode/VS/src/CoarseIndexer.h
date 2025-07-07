// CoarseIndexer.h // FineReranker.h
#pragma once
#include <vector>
#include <cassert>

#include <queue>
#include <cmath>
#include <limits>

class CoarseIndexer {
public:
    explicit CoarseIndexer(int dim) : dim_(dim) {}

    void add(const float* vec) {
        assert(vec);
        base_vectors_.emplace_back(vec, vec + dim_);
    }

    const std::vector<std::vector<float>>& getBaseVectors() const {
        return base_vectors_;
    }

private:
    int dim_;
    std::vector<std::vector<float>> base_vectors_;
};


class FineReranker {
public:
    FineReranker(int dim, int M) : dim_(dim), M_(M), efSearch(64) {}

    void add(const float* vec) {
        rerank_data_.emplace_back(vec, vec + dim_);
    }

    // Brute-force rerank for simplicity (replace with HNSW later)
    std::pair<float, int> search(const std::vector<float>& query) {
        float best_dist = std::numeric_limits<float>::max();
        int best_id = -1;

        for (size_t i = 0; i < rerank_data_.size(); ++i) {
            float dist = l2_distance(query.data(), rerank_data_[i].data());
            if (dist < best_dist) {
                best_dist = dist;
                best_id = i;
            }
        }
        return {best_dist, best_id};
    }

    int efSearch;

private:
    float l2_distance(const float* a, const float* b) {
        float sum = 0.0f;
        for (int i = 0; i < dim_; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    int dim_;
    int M_;
    std::vector<std::vector<float>> rerank_data_;
};
