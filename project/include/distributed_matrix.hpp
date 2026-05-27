#ifndef DISTRIBUTED_MATRIX_H
#define DISTRIBUTED_MATRIX_H

#include "matrix.hpp"
#include <mpi.h>
#include <vector>
#include <functional>

// Represent a *global* matrix of size `globalRows x globalCols` by
// storing a *local* matrix on each process that represents the part of the matrix
// from column `startCol` (included, 0-based index) to column `startCol + localCols` (excluded, 0-based index).
class DistributedMatrix
{
private:
    int globalRows;    // Total number of rows
    int globalCols;    // Total number of columns
    int localCols;     // Number of columns in this process
    int startCol;      // Starting column index for this process
    int numProcesses;  // Total number of MPI processes
    int rank;          // Rank of this process
    Matrix localData;  // Local portion of the matrix

public:
    // --- Constructors & Assignment ---
    //      Assumes that MPI is already initialized
    //      This constructor is called in parallel by all processes
    //      Extract the columns that should be handled by this process in localData
    DistributedMatrix(const Matrix& matrix, int numProcesses);
    DistributedMatrix(const DistributedMatrix& other);
    DistributedMatrix& operator=(const DistributedMatrix& other) = default;

    // --- Common API (shared with Matrix and MatrixCL) ---

    int numRows() const;
    int numCols() const;

    void fill(double value);

    DistributedMatrix operator+(const DistributedMatrix& other) const;
    DistributedMatrix operator-(const DistributedMatrix& other) const;
    DistributedMatrix operator*(double scalar) const;

    // Note: returns a regular Matrix (requires gathering all data)
    Matrix transpose() const;

    // this = this - scalar * other
    void sub_mul(double scalar, const DistributedMatrix& other);

    // --- Distributed-specific operations ---

    double get(int i, int j) const;
    void set(int i, int j, double value);

    // Column index conversions
    int globalColIndex(int localColIndex) const;
    int localColIndex(int globalColIndex) const;
    int ownerProcess(int globalColIndex) const;

    const Matrix& getLocalData() const;

    // Apply a function element-wise (no communication needed)
    DistributedMatrix apply(const std::function<double(double)> &func) const;

    // Apply a binary function to two distributed matrices with the same column partitioning
    static DistributedMatrix applyBinary(
        const DistributedMatrix& a,
        const DistributedMatrix& b,
        const std::function<double(double, double)> &func);

    // Matrix * DistributedMatrix multiplication
    friend DistributedMatrix multiply(const Matrix& left, const DistributedMatrix& right);

    // DistributedMatrix * DistributedMatrix^T (returns a regular Matrix)
    //      Assumes the same column partitioning for both inputs
    Matrix multiplyTransposed(const DistributedMatrix& other) const;

    // Sum of all elements across all processes
    double sum() const;

    // Gather into a complete matrix on all processes (for testing/debugging)
    Matrix gather() const;

    ~DistributedMatrix() = default;
};

// Matrix * DistributedMatrix multiplication (left matrix already on all processes)
DistributedMatrix multiply(const Matrix& left, const DistributedMatrix& right);

// Broadcast a matrix from one process to all others
void sync_matrix(Matrix *matrix, int rank, int src);

#endif // DISTRIBUTED_MATRIX_H
