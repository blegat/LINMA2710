#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <functional>

#include "abstractmatrix.hpp"

class Matrix : public AbstractMatrix
{
private:
    int rows, cols;
    std::vector<double> data;

public:
    // Constructors
    Matrix(int rows, int cols);
    Matrix(const Matrix &other);

    // Basic access
    double get(int i, int j) const;
    void set(int i, int j, double value);

    // Getters for the number of rows and columns
    int numRows() const;
    int numCols() const;

    // Fill the entire matrix with a single value.
    void fill(double value);

    // Elementary operations
    Matrix operator+(const Matrix &other) const; // Addition
    Matrix operator*(const Matrix &other) const; // Matrix multiplication
    
    // Transpose: returns a new Matrix that is the transpose.
    Matrix transpose() const;

    // Apply a function element–wise.
    Matrix apply(const std::function<double(double)> &func) const;

    // Subtract the product of a scalar and a given matrix: "this = this - scalar * other"
    void sub_mul(double scalar, const Matrix &other);

    // Add copy assignment operator (For students: you can ignore this)
    Matrix &operator=(const Matrix &other)
    {
        if (this != &other)
        {
            rows = other.rows;
            cols = other.cols;
            data = other.data;
        }
        return *this;
    }
};

#endif // MATRIX_H
