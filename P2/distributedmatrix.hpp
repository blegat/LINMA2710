#ifndef DISTRIBUTED_MATRIX_H
#define DISTRIBUTED_MATRIX_H

#include "abstractmatrix.hpp"
#include "matrix.hpp"
#include <mpi.h>
#include <vector>
#include <functional>

// Represent a *global* matrix of size `globalRows x globalCols` by
// storying a *local* matrix on each process that represents the part of the matrix
// from column `startCol` (included, 0-based index) to column `startCol + localCols` (excluded, 0-based index).
class DistributedMatrix : public AbstractMatrix
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
    // Constructor taking a Matrix in input and returning a DistributedMatrix
    //      Assumes that MPI is already initialized
    //      This constructor is called in parallel by all processes
    //      Extract the columns that should be handled by this process in localData
    DistributedMatrix(const Matrix& matrix, int numProcesses);
    
    // Copy constructor
    DistributedMatrix(const DistributedMatrix& other);

    // Assignement operator (not necessary to implement)
    DistributedMatrix& operator=(const DistributedMatrix& other) = default;
    
    // Implementation of AbstractMatrix interface
    int numRows() const override;
    int numCols() const override;
    double get(int i, int j) const override;
    void set(int i, int j, double value) override;
    
    // Get the global column index (in the full distributed matrix) from a local index (in the localData matrix)
    int globalColIndex(int localColIndex) const;
    
    // Get the local column index (in localData) from a global index (in the full distributed matrix) (or -1 if not local)
    int localColIndex(int globalColIndex) const;
    
    // Get the process rank that owns a particular global column
    int ownerProcess(int globalColIndex) const;
    
    // Get the local data matrix
    const Matrix& getLocalData() const;
    
    // Apply a function element-wise on the local data, returning the result as a new DistributedMatrix with the same partitioning of the columns across processes
    DistributedMatrix apply(const std::function<double(double)> &func) const;
    
    // Apply a binary function to two distributed matrices with the same columns' partitioning across processes (and keeps this partioning for the result)
    static DistributedMatrix applyBinary(
        const DistributedMatrix& a,
        const DistributedMatrix& b,
        const std::function<double(double, double)> &func);
    
    // Matrix multiplication: Matrix * DistributedMatrix (friend declaration) (no implementation here)
    friend DistributedMatrix multiply(const Matrix& left, const DistributedMatrix& right);
    
    // Matrix multiplication: DistributedMatrix * DistributedMatrix^T (returns a regular Matrix)
    //      Can assume the same columns' partitioning across processes for the inputs
    Matrix multiplyTransposed(const DistributedMatrix& other) const;

    // Return the sum of all the elements of the global matrix
    double sum() const;
    
    // Gather the distributed matrix into a complete matrix on all processes
    // This is operation is helpful for testing and debugging but should be avoided on large scale computation
    Matrix gather() const;
    
    // Destructor
    ~DistributedMatrix() = default;
};

// Function for matrix * distributedmatrix multiplication
//      Assumes that the left matrix is already on all processes (no need to broadcast it)
//      Returns a DistributedMatrix with the same columns' partioning as the input right DistributedMatrix
DistributedMatrix multiply(const Matrix& left, const DistributedMatrix& right);

// Synchronize the value of all processes so that after the call of this function,
// the value of the matrix on all process is the value before this call that the matrix had on the process for which `rank == src`.
void sync_matrix(Matrix *matrix, int rank, int src);

#endif // DISTRIBUTED_MATRIX_H
