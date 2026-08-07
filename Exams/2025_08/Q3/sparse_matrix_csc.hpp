#ifndef SPARSE_MATRIX_CSC_HPP
#define SPARSE_MATRIX_CSC_HPP

#include <vector>
#include <iostream>
#include <omp.h>

class SparseMatrixCSC {
private:
    std::vector<float> nzval;    // Non-zero nzval
    std::vector<size_t> rowval; // Row indices of non-zero nzval
    std::vector<size_t> colptr;  // Column pointers (start index of each column)
    size_t m;
    size_t n;
public:
    std::vector<float> multiply(const std::vector<float>& x) const {
        if (x.size() != n)
            throw std::invalid_argument("Size mismatch");
        std::vector<float> y(m, 0);
        #pragma omp parallel for
        for (size_t j = 0; j < n; ++j) {
            for (size_t idx = colptr[j]; idx < colptr[j + 1]; ++idx) {
                size_t i = rowval[idx];
                float val = nzval[idx]; //#pragma omp atomic
                y[i] += val * x[j];
            }
        }
        return y;
    }

    // Constructor
    SparseMatrixCSC(size_t rows, size_t cols) : m(rows), n(cols) {
        colptr.resize(cols + 1, 0);
    }

    // Constructor from dense matrix
    SparseMatrixCSC(const std::vector<std::vector<float>>& dense_matrix) {
        m = dense_matrix.size();
        n = dense_matrix[0].size();
        
        // Count non-zeros
        size_t nnz = 0;
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                if (dense_matrix[i][j] != 0) {
                    nnz++;
                }
            }
        }
        
        nzval.resize(nnz);
        rowval.resize(nnz);
        colptr.resize(n + 1, 0);
        
        // Fill CSC format
        size_t idx = 0;
        for (size_t j = 0; j < n; ++j) {
            colptr[j] = idx;
            for (size_t i = 0; i < m; ++i) {
                if (dense_matrix[i][j] != 0) {
                    nzval[idx] = dense_matrix[i][j];
                    rowval[idx] = i;
                    idx++;
                }
            }
        }
        colptr[n] = nnz;
    }

    // Get matrix dimensions
    size_t getNumRows() const { return m; }
    size_t getNumCols() const { return n; }
    size_t getNNZ() const { return nzval.size(); }

    // Matrix-vector multiplication with pre-allocated result vector
    void multiply(const std::vector<float>& x, std::vector<float>& y) const {
        if (x.size() != n) {
            throw std::invalid_argument("Vector size does not match matrix columns");
        }
        if (y.size() != m) {
            y.resize(m);
        }
        
        // Initialize result vector to zero
        #pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
            y[i] = 0;
        }
        
        // Perform matrix-vector multiplication
        #pragma omp parallel for
        for (size_t j = 0; j < n; ++j) {
            for (size_t idx = colptr[j]; idx < colptr[j + 1]; ++idx) {
                size_t i = rowval[idx];
                float val = nzval[idx];
                #pragma omp atomic
                y[i] += val * x[j];
            }
        }
    }

    // Print matrix (for debugging)
    void print() const {
        std::cout << "CSC Matrix (" << m << "x" << n << ") with " << nzval.size() << " non-zeros\n";
        std::cout << "nzval: ";
        for (const auto& val : nzval) {
            std::cout << val << " ";
        }
        std::cout << "\nRow indices: ";
        for (const auto& idx : rowval) {
            std::cout << idx << " ";
        }
        std::cout << "\nColumn pointers: ";
        for (const auto& ptr : colptr) {
            std::cout << ptr << " ";
        }
        std::cout << "\n";
    }

    // Convert to dense matrix (for verification)
    std::vector<std::vector<float>> toDense() const {
        std::vector<std::vector<float>> dense(m, std::vector<float>(n, 0));
        
        for (size_t j = 0; j < n; ++j) {
            for (size_t idx = colptr[j]; idx < colptr[j + 1]; ++idx) {
                size_t i = rowval[idx];
                dense[i][j] = nzval[idx];
            }
        }
        
        return dense;
    }

    // Set number of OpenMP threads
    static void setNumThreads(int num_threads) {
        omp_set_num_threads(num_threads);
    }

    // Get current number of OpenMP threads
    static int getNumThreads() {
        return omp_get_max_threads();
    }
};

#endif // SPARSE_MATRIX_CSC_HPP 
