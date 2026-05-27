#include <vector>
#include <cstddef>

void compute_bound_no_simd(const std::vector<double>& x,
                           std::vector<double>& y)
{
    const std::size_t N = x.size();
#pragma clang loop vectorize(disable)
    //TODO
}
