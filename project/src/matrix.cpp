#include "matrix.hpp"
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

Matrix::Matrix(int rows, int cols)
    : rows(0), cols(0)
{
    // TODO
}

Matrix::Matrix(const Matrix &other)
    : rows(0), cols(0)
{
    // TODO
}

int Matrix::numRows() const
{
    return 0; // TODO
}

int Matrix::numCols() const
{
    return 0; // TODO
}

double Matrix::get(int i, int j) const
{
    return 0.0; // TODO
}

void Matrix::set(int i, int j, double value)
{
    // TODO
}

void Matrix::fill(double value)
{
    // TODO
}

Matrix Matrix::operator+(const Matrix &other) const
{
    Matrix result(0, 0);
    // TODO
    return result;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    Matrix result(0, 0);
    // TODO
    return result;
}

Matrix Matrix::operator*(const Matrix &other) const
{
    Matrix result(0, 0);
    // TODO
    return result;
}

Matrix Matrix::operator*(double scalar) const
{
    Matrix result(0, 0);
    // TODO
    return result;
}

Matrix Matrix::transpose() const
{
    Matrix result(0, 0);
    // TODO
    return result;
}

Matrix Matrix::apply(const std::function<double(double)> &func) const
{
    Matrix result(0, 0);
    // TODO
    return result;
}

void Matrix::sub_mul(double scalar, const Matrix &other)
{
    // TODO
}
