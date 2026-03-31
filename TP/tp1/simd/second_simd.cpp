#include <vector>
#include <cstddef>

void compute_bound_simd(const std::vector<double>& x,
                        std::vector<double>& y)
{
    const std::size_t N = x.size();
#pragma clang loop vectorize(enable)
    for (std::size_t i = 0; i < N; ++i) {
        double val = x[i];
        for (int k = 0; k < 50; ++k)
            val = val * val;
        y[i] = val;
    }
}
