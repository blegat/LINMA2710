#pragma once
#include <vector>

void memory_bound_no_simd(const std::vector<double>& x, std::vector<double>& y);
void memory_bound_simd(const std::vector<double>& x, std::vector<double>& y);
