#ifndef ABSTRACT_MATRIX_H
#define ABSTRACT_MATRIX_H

#include <functional>

class AbstractMatrix 
{
public:
    // Pure virtual methods
    virtual int numRows() const = 0;
    virtual int numCols() const = 0;
    
    // Virtual methods that can be overridden
    virtual double get(int i, int j) const = 0;
    virtual void set(int i, int j, double value) = 0;
    
    // Virtual destructor
    virtual ~AbstractMatrix() = default;
};

#endif // ABSTRACT_MATRIX_H