#pragma once
#include <vector>

void compute_bound_no_simd(const std::vector<double>& x, std::vector<double>& y);
void compute_bound_simd(const std::vector<double>& x, std::vector<double>& y);
