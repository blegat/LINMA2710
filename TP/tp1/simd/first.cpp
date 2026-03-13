#include <iostream>
#include <vector>
#include <chrono>
#include "first.hpp"

int main() {
    const std::size_t N = 20'000'000;
    std::vector<double> x(N), y1(N), y2(N);

    for (std::size_t i = 0; i < N; ++i)
        x[i] = 0.001 * i + 1.0;

    auto t1 = std::chrono::high_resolution_clock::now();
    memory_bound_no_simd(x, y1);
    auto t2 = std::chrono::high_resolution_clock::now();

    memory_bound_simd(x, y2);
    auto t3 = std::chrono::high_resolution_clock::now();

    auto mem_no_simd = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto mem_simd    = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

    std::cout << y1[123] << " " << y2[123] << "\n\n";
    std::cout << "=== Memory-bound loop ===\n";
    std::cout << "No SIMD   : " << mem_no_simd << " ms\n";
    std::cout << "With SIMD : " << mem_simd    << " ms\n";
}
