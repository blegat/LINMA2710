#include <cassert>
#include <cmath>
#include <iostream>

#include "matrix.hpp"

bool approxEqual(double a, double b, double epsilon = 1e-6)
{
    return std::fabs(a - b) < epsilon;
}

bool matricesEqual(const Matrix &a, const Matrix &b, double epsilon = 1e-6)
{
    if (a.numRows() != b.numRows() || a.numCols() != b.numCols())
        return false;
    for (int i = 0; i < a.numRows(); ++i)
        for (int j = 0; j < a.numCols(); ++j)
            if (!approxEqual(a.get(i, j), b.get(i, j), epsilon))
                return false;
    return true;
}

void testConstructorsAndAccessors()
{
    Matrix m(2, 3);
    m.fill(1.5);

    for (int i = 0; i < m.numRows(); ++i)
        for (int j = 0; j < m.numCols(); ++j)
            assert(approxEqual(m.get(i, j), 1.5));

    m.set(0, 0, 3.0);
    assert(approxEqual(m.get(0, 0), 3.0));

    Matrix copy(m);
    assert(copy.numRows() == m.numRows() && copy.numCols() == m.numCols());
    assert(matricesEqual(m, copy));

    Matrix assigned(1, 1);
    assigned = m;
    assert(matricesEqual(m, assigned));

    std::cout << "testConstructorsAndAccessors passed." << std::endl;
}

void testAdditionSubtraction()
{
    Matrix a(2, 2);
    a.set(0, 0, 1); a.set(0, 1, 2);
    a.set(1, 0, 3); a.set(1, 1, 4);

    Matrix b(2, 2);
    b.set(0, 0, 5); b.set(0, 1, 6);
    b.set(1, 0, 7); b.set(1, 1, 8);

    Matrix sum = a + b;
    assert(approxEqual(sum.get(0, 0), 6));
    assert(approxEqual(sum.get(0, 1), 8));
    assert(approxEqual(sum.get(1, 0), 10));
    assert(approxEqual(sum.get(1, 1), 12));

    Matrix diff = b - a;
    assert(approxEqual(diff.get(0, 0), 4));
    assert(approxEqual(diff.get(0, 1), 4));
    assert(approxEqual(diff.get(1, 0), 4));
    assert(approxEqual(diff.get(1, 1), 4));

    std::cout << "testAdditionSubtraction passed." << std::endl;
}

void testScalarAndSquareMultiplication()
{
    Matrix a(2, 2);
    a.set(0, 0, 1); a.set(0, 1, 2);
    a.set(1, 0, 3); a.set(1, 1, 4);

    Matrix scalarMul = a * 2;
    assert(approxEqual(scalarMul.get(0, 0), 2));
    assert(approxEqual(scalarMul.get(0, 1), 4));
    assert(approxEqual(scalarMul.get(1, 0), 6));
    assert(approxEqual(scalarMul.get(1, 1), 8));

    Matrix b(2, 2);
    b.set(0, 0, 5); b.set(0, 1, 6);
    b.set(1, 0, 7); b.set(1, 1, 8);

    Matrix prod = a * b;
    assert(approxEqual(prod.get(0, 0), 19));
    assert(approxEqual(prod.get(0, 1), 22));
    assert(approxEqual(prod.get(1, 0), 43));
    assert(approxEqual(prod.get(1, 1), 50));

    std::cout << "testScalarAndSquareMultiplication passed." << std::endl;
}

void testRectangularMultiplication()
{
    Matrix A(3, 2);
    A.set(0, 0, 1); A.set(0, 1, 2);
    A.set(1, 0, 3); A.set(1, 1, 4);
    A.set(2, 0, 5); A.set(2, 1, 6);

    Matrix B(2, 4);
    B.set(0, 0, 7);  B.set(0, 1, 8);  B.set(0, 2, 9);  B.set(0, 3, 10);
    B.set(1, 0, 11); B.set(1, 1, 12); B.set(1, 2, 13); B.set(1, 3, 14);

    Matrix C = A * B;
    assert(C.numRows() == 3 && C.numCols() == 4);

    double expected[3][4] = {
        {29, 32, 35, 38},
        {65, 72, 79, 86},
        {101, 112, 123, 134}};

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            assert(approxEqual(C.get(i, j), expected[i][j]));

    std::cout << "testRectangularMultiplication passed." << std::endl;
}

void testTranspose()
{
    Matrix a(2, 2);
    a.set(0, 0, 1); a.set(0, 1, 2);
    a.set(1, 0, 3); a.set(1, 1, 4);

    Matrix t = a.transpose();
    assert(approxEqual(t.get(0, 0), 1));
    assert(approxEqual(t.get(0, 1), 3));
    assert(approxEqual(t.get(1, 0), 2));
    assert(approxEqual(t.get(1, 1), 4));

    std::cout << "testTranspose passed." << std::endl;
}

void testApply()
{
    Matrix a(2, 2);
    a.set(0, 0, 1); a.set(0, 1, 2);
    a.set(1, 0, 3); a.set(1, 1, 4);

    Matrix squared = a.apply([](double x) { return x * x; });
    assert(approxEqual(squared.get(0, 0), 1));
    assert(approxEqual(squared.get(0, 1), 4));
    assert(approxEqual(squared.get(1, 0), 9));
    assert(approxEqual(squared.get(1, 1), 16));

    std::cout << "testApply passed." << std::endl;
}

void testSubMul()
{
    Matrix a(2, 2);
    a.set(0, 0, -1); a.set(0, 1, 2);
    a.set(1, 0, 3);  a.set(1, 1, 5);

    Matrix c(2, 2);
    c.set(0, 0, 5); c.set(0, 1, 6);
    c.set(1, 0, 7); c.set(1, 1, 8);

    a.sub_mul(1, c);
    assert(approxEqual(a.get(0, 0), -6));
    assert(approxEqual(a.get(0, 1), -4));
    assert(approxEqual(a.get(1, 0), -4));
    assert(approxEqual(a.get(1, 1), -3));

    std::cout << "testSubMul passed." << std::endl;
}

int main()
{
    testConstructorsAndAccessors();
    testAdditionSubtraction();
    testScalarAndSquareMultiplication();
    testRectangularMultiplication();
    testTranspose();
    testApply();
    testSubMul();

    std::cout << "All matrix tests passed." << std::endl;
    return 0;
}
