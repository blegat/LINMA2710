#include <cassert>
#include <cmath>
#include <iostream>

#include "matrix.hpp"
#include "mlp_sgd.cpp"
// A helper function to compare floating–point values.
bool almostEqual(double a, double b, double epsilon = 1e-6)
{
    return std::fabs(a - b) < epsilon;
}

void test_mlp_training()
{
    Dataset data;

    // Simple dataset: XOR problem
    data.X.push_back(Node(2, 1));
    data.X.back().set(0, 0, 0.0);
    data.X.back().set(1, 0, 0.0);

    data.Y.push_back(Node(1, 1));
    data.Y.back().set(0, 0, 0.0);

    data.X.push_back(Node(2, 1));
    data.X.back().set(0, 0, 0.0);
    data.X.back().set(1, 0, 1.0);

    data.Y.push_back(Node(1, 1));
    data.Y.back().set(0, 0, 1.0);

    data.X.push_back(Node(2, 1));
    data.X.back().set(0, 0, 1.0);
    data.X.back().set(1, 0, 0.0);

    data.Y.push_back(Node(1, 1));
    data.Y.back().set(0, 0, 1.0);

    data.X.push_back(Node(2, 1));
    data.X.back().set(0, 0, 1.0);
    data.X.back().set(1, 0, 1.0);

    data.Y.push_back(Node(1, 1));
    data.Y.back().set(0, 0, 0.);

    MLP model(2, 128, 1, 1.);
    model.train(data, 1000);

    // Evaluate the model
    for (size_t i = 0; i < data.X.size(); ++i)
    {
        Node *output = model.forward(data.X[i]);
        std::cout << "Input: " << data.X[i].get(0, 0) << ", " << data.X[i].get(1, 0);
        std::cout << " | Predicted: " << output->get(0, 0);
        std::cout << " | Target: " << data.Y[i].get(0, 0) << "\n";
    }

    std::cout << "MLP training test completed.\n";
}

int main()
{
    // --------------------------------------------------
    // Test 1: Constructors, fill, get, and set
    // --------------------------------------------------
    {
        // Create a 2x3 matrix and fill it with 1.5.
        Matrix m(2, 3);
        m.fill(1.5);

        // Check that every element is 1.5.
        for (int i = 0; i < m.numRows(); ++i)
        {
            for (int j = 0; j < m.numCols(); ++j)
            {
                assert(almostEqual(m.get(i, j), 1.5));
            }
        }

        // Modify an element.
        m.set(0, 0, 3.0);
        assert(almostEqual(m.get(0, 0), 3.0));

        // Test the copy constructor.
        Matrix copy(m);
        assert(copy.numRows() == m.numRows() && copy.numCols() == m.numCols());
        for (int i = 0; i < m.numRows(); ++i)
        {
            for (int j = 0; j < m.numCols(); ++j)
            {
                assert(almostEqual(m.get(i, j), copy.get(i, j)));
            }
        }
    }

    // --------------------------------------------------
    // Test 2: Addition and Subtraction Operators
    // --------------------------------------------------
    {
        // Define two 2x2 matrices.
        Matrix a(2, 2);
        a.set(0, 0, 1);
        a.set(0, 1, 2);
        a.set(1, 0, 3);
        a.set(1, 1, 4);

        Matrix b(2, 2);
        b.set(0, 0, 5);
        b.set(0, 1, 6);
        b.set(1, 0, 7);
        b.set(1, 1, 8);

        // Test addition: a + b should yield [ [6,8], [10,12] ]
        Matrix sum = a + b;
        assert(almostEqual(sum.get(0, 0), 6));
        assert(almostEqual(sum.get(0, 1), 8));
        assert(almostEqual(sum.get(1, 0), 10));
        assert(almostEqual(sum.get(1, 1), 12));

        // Test subtraction: b - a should yield [ [4,4], [4,4] ]
        Matrix diff = b - a;
        assert(almostEqual(diff.get(0, 0), 4));
        assert(almostEqual(diff.get(0, 1), 4));
        assert(almostEqual(diff.get(1, 0), 4));
        assert(almostEqual(diff.get(1, 1), 4));
    }

    // --------------------------------------------------
    // Test 3: Scalar Multiplication and Matrix Multiplication
    // --------------------------------------------------
    {
        // Reuse matrix 'a' from before: [ [1,2], [3,4] ]
        Matrix a(2, 2);
        a.set(0, 0, 1);
        a.set(0, 1, 2);
        a.set(1, 0, 3);
        a.set(1, 1, 4);

        // Scalar multiplication: a * 2 should yield [ [2,4], [6,8] ]
        Matrix scalarMul = a * 2;
        assert(almostEqual(scalarMul.get(0, 0), 2));
        assert(almostEqual(scalarMul.get(0, 1), 4));
        assert(almostEqual(scalarMul.get(1, 0), 6));
        assert(almostEqual(scalarMul.get(1, 1), 8));

        // Matrix multiplication: Multiply a by another 2x2 matrix b.
        Matrix b(2, 2);
        b.set(0, 0, 5);
        b.set(0, 1, 6);
        b.set(1, 0, 7);
        b.set(1, 1, 8);

        // Expected product:
        // [ [1*5 + 2*7, 1*6 + 2*8],
        //   [3*5 + 4*7, 3*6 + 4*8] ]
        // = [ [19, 22],
        //     [43, 50] ]
        Matrix prod = a * b;
        assert(almostEqual(prod.get(0, 0), 19));
        assert(almostEqual(prod.get(0, 1), 22));
        assert(almostEqual(prod.get(1, 0), 43));
        assert(almostEqual(prod.get(1, 1), 50));
    }

    {
        // Create a 3x2 matrix A.
        // A = [ [1, 2],
        //       [3, 4],
        //       [5, 6] ]
        Matrix A(3, 2);
        A.set(0, 0, 1);
        A.set(0, 1, 2);
        A.set(1, 0, 3);
        A.set(1, 1, 4);
        A.set(2, 0, 5);
        A.set(2, 1, 6);

        // Create a 2x4 matrix B.
        // B = [ [ 7,  8,  9, 10],
        //       [11, 12, 13, 14] ]
        Matrix B(2, 4);
        B.set(0, 0, 7);
        B.set(0, 1, 8);
        B.set(0, 2, 9);
        B.set(0, 3, 10);
        B.set(1, 0, 11);
        B.set(1, 1, 12);
        B.set(1, 2, 13);
        B.set(1, 3, 14);

        // Multiply A and B: C = A * B
        // Expected result C (3x4) is computed as:
        // Row 0: [1*7 + 2*11, 1*8 + 2*12, 1*9 + 2*13, 1*10 + 2*14] = [29, 32, 35, 38]
        // Row 1: [3*7 + 4*11, 3*8 + 4*12, 3*9 + 4*13, 3*10 + 4*14] = [65, 72, 79, 86]
        // Row 2: [5*7 + 6*11, 5*8 + 6*12, 5*9 + 6*13, 5*10 + 6*14] = [101, 112, 123, 134]
        Matrix C = A * B;

        // Expected values.
        double expected[3][4] = {
            {29, 32, 35, 38},
            {65, 72, 79, 86},
            {101, 112, 123, 134}};

        // Verify dimensions.
        assert(C.numRows() == 3 && C.numCols() == 4);

        // Check each element of the product.
        for (int i = 0; i < C.numRows(); ++i)
        {
            for (int j = 0; j < C.numCols(); ++j)
            {
                if (!almostEqual(C.get(i, j), expected[i][j]))
                {
                    std::cerr << "Mismatch at (" << i << "," << j << "): "
                              << "expected " << expected[i][j]
                              << ", got " << C.get(i, j) << "\n";
                    assert(false);
                }
            }
        }
    }

    // --------------------------------------------------
    // Test 4: Transpose
    // --------------------------------------------------
    {
        // a = [ [1,2], [3,4] ]
        Matrix a(2, 2);
        a.set(0, 0, 1);
        a.set(0, 1, 2);
        a.set(1, 0, 3);
        a.set(1, 1, 4);

        // Transpose should yield [ [1,3], [2,4] ]
        Matrix t = a.transpose();
        assert(almostEqual(t.get(0, 0), 1));
        assert(almostEqual(t.get(0, 1), 3));
        assert(almostEqual(t.get(1, 0), 2));
        assert(almostEqual(t.get(1, 1), 4));
    }

    // --------------------------------------------------
    // Test 5: Apply (element–wise function)
    // --------------------------------------------------
    {
        // a = [ [1,2], [3,4] ]
        Matrix a(2, 2);
        a.set(0, 0, 1);
        a.set(0, 1, 2);
        a.set(1, 0, 3);
        a.set(1, 1, 4);

        // Apply a function to square each element.
        Matrix squared = a.apply([](double x)
                                 { return x * x; });
        // Expected squared: [ [1,4], [9,16] ]
        assert(almostEqual(squared.get(0, 0), 1));
        assert(almostEqual(squared.get(0, 1), 4));
        assert(almostEqual(squared.get(1, 0), 9));
        assert(almostEqual(squared.get(1, 1), 16));
    }

    // --------------------------------------------------
    // Test 6: sub_mul (Subtract scalar then multiply with another matrix)
    // --------------------------------------------------
    {
        // For this test, assume that sub_mul subtracts a scalar from each element of
        // the current matrix and then multiplies (element–wise) by the corresponding element of another matrix.
        // Let a = [ [1,2], [3,4] ] and c = [ [5,6], [7,8] ].
        Matrix a(2, 2);
        a.set(0, 0, -1);
        a.set(0, 1, 2);
        a.set(1, 0, 3);
        a.set(1, 1, 5);

        Matrix c(2, 2);
        c.set(0, 0, 5);
        c.set(0, 1, 6);
        c.set(1, 0, 7);
        c.set(1, 1, 8);

        // After performing a.sub_mul(1, c), we expect each element of a to be:
        // new_a[i][j] = (a[i][j] - 1) * c[i][j]
        // Calculation:
        // [ [-1-1*5, 2-1*6 ],
        //   [3-1*7, 5-1*8 ] ]
        // = [ [-6, -4],
        //     [-4, -3] ]
        a.sub_mul(1, c);
        assert(almostEqual(a.get(0, 0), -6));
        assert(almostEqual(a.get(0, 1), -4));
        assert(almostEqual(a.get(1, 0), -4));
        assert(almostEqual(a.get(1, 1), -3));
    }

    std::cout << "Matrix tests passed." << std::endl;

    test_mlp_training();
    clear_nodes();

    return 0;
}
