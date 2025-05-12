#include "distributedmatrix.hpp"
#include "matrix.hpp"
#include "mlp_sgd_distributed.cpp"
#include <mpi.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <functional>

// Helper function to test if two doubles are approximately equal
bool approxEqual(double a, double b, double epsilon = 1e-10) {
    return std::abs(a - b) < epsilon;
}

// Helper function to test if two matrices are approximately equal
bool matricesEqual(const Matrix& a, const Matrix& b, double epsilon = 1e-10) {
    if (a.numRows() != b.numRows() || a.numCols() != b.numCols()) {
        return false;
    }
    
    for (int i = 0; i < a.numRows(); i++) {
        for (int j = 0; j < a.numCols(); j++) {
            if (!approxEqual(a.get(i, j), b.get(i, j), epsilon)) {
                return false;
            }
        }
    }
    
    return true;
}

// Test constructor and basic properties
void testConstructorAndBasics() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Create a test matrix
    Matrix testMatrix(3, 4);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            testMatrix.set(i, j, i * 10 + j);
        }
    }
    
    // Create distributed matrix
    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    DistributedMatrix distMatrix(testMatrix, numProcs);
    
    // Check dimensions
    assert(distMatrix.numRows() == 3);
    assert(distMatrix.numCols() == 4);
    
    // Gather and check equality with original
    Matrix gathered = distMatrix.gather();
    assert(matricesEqual(gathered, testMatrix));
    
    if (rank == 0) {
        std::cout << "Constructor and basic properties test passed!" << std::endl;
    }
}

// Test column distribution
void testColumnDistribution() {
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    
    // Create a test matrix with more columns to better test distribution
    int cols = numProcs * 2 + 1;  // Ensure some remainder for testing
    Matrix testMatrix(3, cols);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < cols; j++) {
            testMatrix.set(i, j, i * 100 + j);
        }
    }
    
    // Create distributed matrix
    DistributedMatrix distMatrix(testMatrix, numProcs);
    
    // Check local data dimensions
    const Matrix& localData = distMatrix.getLocalData();
    
    // Calculate expected column distribution
    int baseCols = cols / numProcs;
    int remainder = cols % numProcs;
    int expectedLocalCols = baseCols + (rank < remainder ? 1 : 0);
    
    assert(localData.numRows() == 3);
    assert(localData.numCols() == expectedLocalCols);
    
    // Check global/local column index conversion
    for (int j = 0; j < localData.numCols(); j++) {
        int globalJ = distMatrix.globalColIndex(j);
        assert(distMatrix.localColIndex(globalJ) == j);
        assert(distMatrix.ownerProcess(globalJ) == rank);
    }
    
    if (rank == 0) {
        std::cout << "Column distribution test passed!" << std::endl;
    }
}

// Test apply function
void testApply() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Create a test matrix
    Matrix testMatrix(2, 5);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 5; j++) {
            testMatrix.set(i, j, i + j);
        }
    }
    
    // Create distributed matrix
    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    DistributedMatrix distMatrix(testMatrix, numProcs);
    
    // Apply a function to square each element
    auto squareFunc = [](double x) { return x * x; };
    DistributedMatrix squaredMatrix = distMatrix.apply(squareFunc);
    
    // Create expected result
    Matrix expectedMatrix = testMatrix.apply(squareFunc);
    
    // Gather and check
    Matrix gathered = squaredMatrix.gather();
    assert(matricesEqual(gathered, expectedMatrix));
    
    if (rank == 0) {
        std::cout << "Apply function test passed!" << std::endl;
    }
}

// Test applyBinary function
void testApplyBinary() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Create test matrices
    Matrix testMatrix1(3, 4);
    Matrix testMatrix2(3, 4);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            testMatrix1.set(i, j, i + j);
            testMatrix2.set(i, j, i * j);
        }
    }
    
    // Create distributed matrices
    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    DistributedMatrix distMatrix1(testMatrix1, numProcs);
    DistributedMatrix distMatrix2(testMatrix2, numProcs);
    
    // Apply binary function (add)
    auto addFunc = [](double a, double b) { return a + b; };
    DistributedMatrix resultMatrix = DistributedMatrix::applyBinary(distMatrix1, distMatrix2, addFunc);
    
    // Create expected result
    Matrix expectedMatrix(3, 4);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            expectedMatrix.set(i, j, testMatrix1.get(i, j) + testMatrix2.get(i, j));
        }
    }
    
    // Gather and check
    Matrix gathered = resultMatrix.gather();
    assert(matricesEqual(gathered, expectedMatrix));
    
    if (rank == 0) {
        std::cout << "ApplyBinary function test passed!" << std::endl;
    }
}

// Test matrix multiplication (Matrix * DistributedMatrix)
void testMultiply() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Create test matrices
    Matrix leftMatrix(2, 3);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            leftMatrix.set(i, j, i * 3 + j + 1);  // 1-based values for better testing
        }
    }
    
    Matrix rightMatrixFull(3, 4);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            rightMatrixFull.set(i, j, i * 4 + j + 1);  // 1-based values
        }
    }
    
    // Create distributed matrix for right operand
    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    DistributedMatrix rightMatrix(rightMatrixFull, numProcs);
    
    // Multiply
    DistributedMatrix resultMatrix = multiply(leftMatrix, rightMatrix);
    
    // Compute expected result
    Matrix expectedMatrix = leftMatrix * rightMatrixFull;
    
    // Gather and check
    Matrix gathered = resultMatrix.gather();
    assert(matricesEqual(gathered, expectedMatrix, 1e-8));  // Use larger epsilon for multiplication
    
    if (rank == 0) {
        std::cout << "Matrix multiplication test passed!" << std::endl;
    }
}

// Test multiplyTransposed
void testMultiplyTransposed() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Create test matrices
    Matrix matrix1Full(3, 5);
    Matrix matrix2Full(4, 5);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 5; j++) {
            matrix1Full.set(i, j, i * 5 + j + 1);
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 5; j++) {
            matrix2Full.set(i, j, i * 5 + j + 2);
        }
    }
    
    // Create distributed matrices
    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    DistributedMatrix matrix1(matrix1Full, numProcs);
    DistributedMatrix matrix2(matrix2Full, numProcs);
    
    // Compute A * B^T
    Matrix result = matrix1.multiplyTransposed(matrix2);
    
    // Compute expected result
    Matrix expectedMatrix = matrix1Full * matrix2Full.transpose();
    
    // Check
    assert(matricesEqual(result, expectedMatrix, 1e-8));
    
    if (rank == 0) {
        std::cout << "MultiplyTransposed test passed!" << std::endl;
    }
}

void testSum() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Create test matrix
    Matrix matrixFull(3, 5);
    double total = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 5; j++) {
            matrixFull.set(i, j, i * 5 + j + 1);
            total += i * 5 + j + 1;
        }
    }
    
    // Create a distributed matrix
    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    DistributedMatrix matrix(matrixFull, numProcs);
    
    // Compute the sum
    double result = matrix.sum();
    
    // Check
    assert(approxEqual(result, total, 1e-8));
    
    if (rank == 0) {
        std::cout << "Sum test passed!" << std::endl;
    }
}

// Test gather function
void testGather() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Create a test matrix
    Matrix testMatrix(4, 6);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 6; j++) {
            testMatrix.set(i, j, i * 10 + j);
        }
    }
    
    // Create distributed matrix
    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    DistributedMatrix distMatrix(testMatrix, numProcs);
    
    // Gather and check equality with original
    Matrix gathered = distMatrix.gather();
    assert(matricesEqual(gathered, testMatrix));
    
    if (rank == 0) {
        std::cout << "Gather function test passed!" << std::endl;
    }
}

// Test the get and set functions (may throw exceptions)
void testGetAndSet() {
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    
    // Skip this test if only one process
    if (numProcs == 1) {
        if (rank == 0) {
            std::cout << "Get/Set test skipped (requires multiple processes)" << std::endl;
        }
        return;
    }
    
    // Create a test matrix
    Matrix testMatrix(2, numProcs);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < numProcs; j++) {
            testMatrix.set(i, j, i * numProcs + j);
        }
    }
    
    // Create distributed matrix - each process gets exactly one column
    DistributedMatrix distMatrix(testMatrix, numProcs);
    
    // Test local access - should work
    bool localAccessWorks = true;
    try {
        double val = distMatrix.get(1, rank);
        assert(approxEqual(val, 1 * numProcs + rank));
        
        distMatrix.set(1, rank, 99.0);
        val = distMatrix.get(1, rank);
        assert(approxEqual(val, 99.0));
    } catch (std::exception& e) {
        localAccessWorks = false;
    }
    assert(localAccessWorks);
    
    // Test remote access - should throw
    if (numProcs > 1) {
        bool remoteAccessFails = false;
        int remoteRank = (rank + 1) % numProcs;
        try {
            (void)distMatrix.get(1, remoteRank);
        } catch (std::exception& e) {
            remoteAccessFails = true;
        }
        assert(remoteAccessFails);
        
        remoteAccessFails = false;
        try {
            distMatrix.set(1, remoteRank, 100.0);
        } catch (std::exception& e) {
            remoteAccessFails = true;
        }
        assert(remoteAccessFails);
    }
    
    if (rank == 0) {
        std::cout << "Get/Set function test passed!" << std::endl;
    }
}

// Test copy constructor
void testCopyConstructor() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Create a test matrix
    Matrix testMatrix(3, 5);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 5; j++) {
            testMatrix.set(i, j, i * 5 + j);
        }
    }
    
    // Create distributed matrix
    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    DistributedMatrix original(testMatrix, numProcs);
    
    // Create a copy
    DistributedMatrix copy(original);
    
    // Check basic properties
    assert(copy.numRows() == original.numRows());
    assert(copy.numCols() == original.numCols());
    
    // Check local data
    const Matrix& originalLocal = original.getLocalData();
    const Matrix& copyLocal = copy.getLocalData();
    assert(matricesEqual(originalLocal, copyLocal));
    
    // Modify copy and check independence
    auto doubleFunc = [](double x) { return 2 * x; };
    DistributedMatrix modifiedCopy = copy.apply(doubleFunc);
    
    Matrix originalGathered = original.gather();
    Matrix modifiedGathered = modifiedCopy.gather();
    assert(!matricesEqual(originalGathered, modifiedGathered));
    
    if (rank == 0) {
        std::cout << "Copy constructor test passed!" << std::endl;
    }
}

void test_distributed_mlp_training()
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Print info about the MPI environment
    if (rank == 0) {
        std::cout << "Running with " << size << " MPI processes." << std::endl;
    }

    // Create a simple XOR dataset
    // Create data
    Matrix X(3, 4);
    Matrix Y(1, 4);

    // 0
    X.set(0, 0, 0.0);
    X.set(1, 0, 0.0);
    X.set(2, 0, 1.0);
    Y.set(0, 0, 0.0);
    
    // 1
    X.set(0, 1, 0.0);
    X.set(1, 1, 1.0);
    X.set(2, 1, 1.0);
    Y.set(0, 1, 1.0);
    
    // 2
    X.set(0, 2, 1.0);
    X.set(1, 2, 0.0);
    X.set(2, 2, 1.0);
    Y.set(0, 2, 1.0);
    
    // 3
    X.set(0, 3, 1.0);
    X.set(1, 3, 1.0);
    X.set(2, 3, 1.0);
    Y.set(0, 3, 0.0);
    
    // Distribute the data
    Dataset data = Dataset(DistributedMatrix(X, size), DistributedMatrix(Y, size));
    
    // Create and train the model
    MLP model(3, 128, 1, 0.1);
    
    if (rank == 0) {
        std::cout << "Training distributed MLP for XOR problem..." << std::endl;
    }
    
    model.train(data, 5000);
    
    if (rank==0) {
        std::cout << "Distributed MLP training test completed." << std::endl;
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(&argc, &argv);
    }
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        std::cout << "Starting DistributedMatrix tests..." << std::endl;
    }
    
    try {
        // Run tests
        testConstructorAndBasics();
        testColumnDistribution();
        testApply();
        testApplyBinary();
        testMultiply();
        testMultiplyTransposed();
        testSum();
        testGather();
        testGetAndSet();
        testCopyConstructor();
        test_distributed_mlp_training();
        
        if (rank == 0) {
            std::cout << "All tests passed successfully!" << std::endl;
        }
    } 
    catch (std::exception& e) {
        if (rank == 0) {
            std::cerr << "Test failed with exception: " << e.what() << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Finalize MPI if we initialized it
    // int finalized;
    // MPI_Finalized(&finalized);
    // if (!finalized && initialized) {
    //     MPI_Finalize();
    // }

    MPI_Finalize();
    return 0;
}