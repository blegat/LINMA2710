#include "distributed_matrix.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

// The matrix is split by columns across MPI processes.
// Each process stores a local Matrix with a subset of columns.
// Columns are distributed as evenly as possible.

DistributedMatrix::DistributedMatrix(const Matrix& matrix, int numProcs)
    : globalRows(matrix.numRows()),
      globalCols(matrix.numCols()),
      localCols(0),
      startCol(0),
      numProcesses(numProcs),
      rank(0),
      localData(matrix.numRows(), 1)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // TODO
}

DistributedMatrix::DistributedMatrix(const DistributedMatrix& other)
    : globalRows(other.globalRows),
      globalCols(other.globalCols),
      localCols(other.localCols),
      startCol(other.startCol),
      numProcesses(other.numProcesses),
      rank(other.rank),
      localData(other.localData)
{
}

int DistributedMatrix::numRows() const { return globalRows; }
int DistributedMatrix::numCols() const { return globalCols; }
const Matrix& DistributedMatrix::getLocalData() const { return localData; }

double DistributedMatrix::get(int i, int j) const
{
    // TODO
    return 0.0;
}

void DistributedMatrix::set(int i, int j, double value)
{
    // TODO
}

int DistributedMatrix::globalColIndex(int localColIdx) const
{
    // TODO
    return -1;
}

int DistributedMatrix::localColIndex(int globalColIdx) const
{
    // TODO
    return -1;
}

int DistributedMatrix::ownerProcess(int globalColIdx) const
{
    // TODO
    return -1;
}

void DistributedMatrix::fill(double value)
{
    // TODO
}

DistributedMatrix DistributedMatrix::operator+(const DistributedMatrix& other) const
{
    // TODO
    return DistributedMatrix(*this);
}

DistributedMatrix DistributedMatrix::operator-(const DistributedMatrix& other) const
{
    // TODO
    return DistributedMatrix(*this);
}

DistributedMatrix DistributedMatrix::operator*(double scalar) const
{
    // TODO
    return DistributedMatrix(*this);
}

Matrix DistributedMatrix::transpose() const
{
    // TODO
    return Matrix(globalCols, globalRows);
}

void DistributedMatrix::sub_mul(double scalar, const DistributedMatrix& other)
{
    // TODO
}

DistributedMatrix DistributedMatrix::apply(const std::function<double(double)>& func) const
{
    // TODO
    return DistributedMatrix(*this);
}

DistributedMatrix DistributedMatrix::applyBinary(
    const DistributedMatrix& a,
    const DistributedMatrix& b,
    const std::function<double(double, double)>& func)
{
    // TODO
    return DistributedMatrix(a);
}

DistributedMatrix multiply(const Matrix& left, const DistributedMatrix& right)
{
    // TODO
    return DistributedMatrix(right);
}

Matrix DistributedMatrix::multiplyTransposed(const DistributedMatrix& other) const
{
    // TODO
    return Matrix(globalRows, other.globalRows);
}

double DistributedMatrix::sum() const
{
    // TODO
    return 0.0;
}

Matrix DistributedMatrix::gather() const
{
    // TODO
    return Matrix(globalRows, globalCols);
}

void sync_matrix(Matrix *matrix, int rank, int src)
{
    // TODO
}
