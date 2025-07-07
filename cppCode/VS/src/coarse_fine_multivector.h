#pragma once
#include "alg.h"
#include <vector>
#include <string>

void runMultiVectorRetrieval(const Preprocess& prep, int k, const std::vector<int>& query_ids, const std::string& result_file_path);
