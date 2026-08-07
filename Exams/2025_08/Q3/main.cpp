#include "sparse_matrix_csc.hpp"
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

// Function to generate a random sparse matrix
std::vector<std::vector<float>> generateRandomSparseMatrix(int rows, int cols, double sparsity) {
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols, 0));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::uniform_real_distribution<float> value_dis(-10.0f, 10.0f);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (dis(gen) > sparsity) {
                matrix[i][j] = value_dis(gen);
            }
        }
    }
    
    return matrix;
}

// Function to generate a random vector
std::vector<float> generateRandomVector(int size) {
    std::vector<float> vec(size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    
    for (int i = 0; i < size; ++i) {
        vec[i] = dis(gen);
    }
    
    return vec;
}

// Function to verify matrix-vector multiplication
bool verifyMultiplication(const std::vector<std::vector<float>>& dense_matrix, 
                         const std::vector<float>& x, 
                         const std::vector<float>& y_csc) {
    int rows = dense_matrix.size();
    int cols = dense_matrix[0].size();
    
    std::vector<float> y_dense(rows, 0);
    
    // Compute dense matrix-vector multiplication
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            y_dense[i] += dense_matrix[i][j] * x[j];
        }
    }
    
    // Compare results
    const float tolerance = 1e-4f;
    for (int i = 0; i < rows; ++i) {
        float diff = std::fabs(y_dense[i] - y_csc[i]);
        float rel_tolerance = std::fabs(y_dense[i]) * 1e-5f + 1e-5f; // Relative tolerance with minimum
        if (diff > tolerance && diff > rel_tolerance) {
            std::cout << "Mismatch at index " << i << ": " 
                      << y_dense[i] << " vs " << y_csc[i] 
                      << " (diff: " << diff << ", tolerance: " << tolerance << ", rel_tolerance: " << rel_tolerance << ")" << std::endl;
            return false;
        }
    }
    
    return true;
}

// Function to measure performance
void benchmarkMultiplication(const SparseMatrixCSC& matrix, 
                           const std::vector<float>& x, 
                           int num_iterations) {
    std::vector<float> y(matrix.getNumRows());
    
    // Warm up
    for (int i = 0; i < 10; ++i) {
        matrix.multiply(x, y);
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        matrix.multiply(x, y);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time = duration.count() / static_cast<double>(num_iterations);
    
    std::cout << "Average time per multiplication: " << std::fixed << std::setprecision(3) 
              << avg_time << " microseconds" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) 
              << (1000000.0 / avg_time) << " multiplications/second" << std::endl;
}

int main() {
    // Set number of OpenMP threads
    SparseMatrixCSC::setNumThreads(4);
    std::cout << "Using " << SparseMatrixCSC::getNumThreads() << " OpenMP threads\n\n";
    
    // Test parameters
    const int rows = 1000;
    const int cols = 1000;
    const double sparsity = 0.95; // 95% sparse (5% non-zeros)
    const int num_iterations = 1000;
    
    std::cout << "Generating random sparse matrix (" << rows << "x" << cols 
              << ") with " << (1.0 - sparsity) * 100 << "% non-zeros...\n";
    
    // Generate test data
    auto dense_matrix = generateRandomSparseMatrix(rows, cols, sparsity);
    auto x = generateRandomVector(cols);
    
    // Create CSC matrix
    SparseMatrixCSC csc_matrix(dense_matrix);
    
    std::cout << "CSC matrix created with " << csc_matrix.getNNZ() << " non-zeros\n";
    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2)
              << (100.0 * csc_matrix.getNNZ() / (rows * cols)) << "%\n\n";
    
    // Test matrix-vector multiplication
    std::cout << "Testing matrix-vector multiplication...\n";
    auto y = csc_matrix.multiply(x);
    
    // Verify result
    if (verifyMultiplication(dense_matrix, x, y)) {
        std::cout << "✓ Matrix-vector multiplication verified successfully!\n\n";
    } else {
        std::cout << "✗ Matrix-vector multiplication verification failed!\n\n";
        return 1;
    }
    
    // Performance benchmark
    std::cout << "Running performance benchmark (" << num_iterations << " iterations)...\n";
    benchmarkMultiplication(csc_matrix, x, num_iterations);
    
    // Test with different thread counts
    std::cout << "\nTesting with different thread counts:\n";
    std::vector<int> thread_counts = {1, 2, 4, 8};
    
    for (int threads : thread_counts) {
        SparseMatrixCSC::setNumThreads(threads);
        std::cout << "\n" << threads << " thread(s): ";
        benchmarkMultiplication(csc_matrix, x, 100);
    }
    
    // Example with a small matrix for demonstration
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "Small example matrix:\n";
    
    std::vector<std::vector<float>> small_matrix = {
        {1.0f, 0.0f, 3.0f},
        {0.0f, 2.0f, 0.0f},
        {4.0f, 0.0f, 5.0f}
    };
    
    SparseMatrixCSC small_csc(small_matrix);
    small_csc.print();
    
    std::vector<float> small_x = {1.0f, 2.0f, 3.0f};
    auto small_y = small_csc.multiply(small_x);
    
    std::cout << "\nInput vector x: ";
    for (const auto& val : small_x) {
        std::cout << val << " ";
    }
    std::cout << "\nResult vector y: ";
    for (const auto& val : small_y) {
        std::cout << val << " ";
    }
    std::cout << "\n";
    
    // Verify small example
    std::cout << "\nVerification:\n";
    auto dense_result = small_csc.toDense();
    for (int i = 0; i < 3; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < 3; ++j) {
            std::cout << dense_result[i][j] << " ";
        }
        std::cout << "| " << small_y[i] << "\n";
    }
    
    return 0;
} 
