#include <vector>
#include <cstddef>

void memory_bound_simd(const std::vector<double>& x,
                       std::vector<double>& y)
{
    const std::size_t N = x.size();
#pragma clang loop vectorize(enable)
    for (std::size_t i = 0; i < N; ++i)
        y[i] = x[i] * x[i];
}
