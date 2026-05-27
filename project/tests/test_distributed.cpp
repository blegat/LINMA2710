#include "distributed_matrix.hpp"
#include "matrix.hpp"
#include <mpi.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <functional>

bool approxEqual(double a, double b, double epsilon = 1e-10) {
    return std::abs(a - b) < epsilon;
}

bool matricesEqual(const Matrix& a, const Matrix& b, double epsilon = 1e-10) {
    if (a.numRows() != b.numRows() || a.numCols() != b.numCols())
        return false;
    for (int i = 0; i < a.numRows(); i++)
        for (int j = 0; j < a.numCols(); j++)
            if (!approxEqual(a.get(i, j), b.get(i, j), epsilon))
                return false;
    return true;
}

void testConstructorAndBasics() {
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    Matrix testMatrix(3, 4);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            testMatrix.set(i, j, i * 10 + j);

    DistributedMatrix distMatrix(testMatrix, numProcs);

    assert(distMatrix.numRows() == 3);
    assert(distMatrix.numCols() == 4);

    Matrix gathered = distMatrix.gather();
    assert(matricesEqual(gathered, testMatrix));

    if (rank == 0)
        std::cout << "testConstructorAndBasics passed." << std::endl;
}

void testColumnDistribution() {
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int cols = numProcs * 2 + 1;
    Matrix testMatrix(3, cols);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < cols; j++)
            testMatrix.set(i, j, i * 100 + j);

    DistributedMatrix distMatrix(testMatrix, numProcs);
    const Matrix& localData = distMatrix.getLocalData();

    int baseCols = cols / numProcs;
    int remainder = cols % numProcs;
    int expectedLocalCols = baseCols + (rank < remainder ? 1 : 0);

    assert(localData.numRows() == 3);
    assert(localData.numCols() == expectedLocalCols);

    for (int j = 0; j < localData.numCols(); j++) {
        int globalJ = distMatrix.globalColIndex(j);
        assert(distMatrix.localColIndex(globalJ) == j);
        assert(distMatrix.ownerProcess(globalJ) == rank);
    }

    if (rank == 0)
        std::cout << "testColumnDistribution passed." << std::endl;
}

void testApply() {
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    Matrix testMatrix(2, 5);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 5; j++)
            testMatrix.set(i, j, i + j);

    DistributedMatrix distMatrix(testMatrix, numProcs);

    auto squareFunc = [](double x) { return x * x; };
    DistributedMatrix squaredMatrix = distMatrix.apply(squareFunc);

    Matrix gathered = squaredMatrix.gather();
    assert(matricesEqual(gathered, testMatrix.apply(squareFunc)));

    if (rank == 0)
        std::cout << "testApply passed." << std::endl;
}

void testApplyBinary() {
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    Matrix testMatrix1(3, 4);
    Matrix testMatrix2(3, 4);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            testMatrix1.set(i, j, i + j);
            testMatrix2.set(i, j, i * j);
        }
    }

    DistributedMatrix distMatrix1(testMatrix1, numProcs);
    DistributedMatrix distMatrix2(testMatrix2, numProcs);

    auto addFunc = [](double a, double b) { return a + b; };
    DistributedMatrix resultMatrix = DistributedMatrix::applyBinary(distMatrix1, distMatrix2, addFunc);

    Matrix expected = testMatrix1 + testMatrix2;
    Matrix gathered = resultMatrix.gather();
    assert(matricesEqual(gathered, expected));

    if (rank == 0)
        std::cout << "testApplyBinary passed." << std::endl;
}

void testMultiply() {
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    Matrix leftMatrix(2, 3);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            leftMatrix.set(i, j, i * 3 + j + 1);

    Matrix rightMatrixFull(3, 4);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            rightMatrixFull.set(i, j, i * 4 + j + 1);

    DistributedMatrix rightMatrix(rightMatrixFull, numProcs);
    DistributedMatrix resultMatrix = multiply(leftMatrix, rightMatrix);

    Matrix gathered = resultMatrix.gather();
    assert(matricesEqual(gathered, leftMatrix * rightMatrixFull, 1e-8));

    if (rank == 0)
        std::cout << "testMultiply passed." << std::endl;
}

void testMultiplyTransposed() {
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    Matrix matrix1Full(3, 5);
    Matrix matrix2Full(4, 5);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 5; j++)
            matrix1Full.set(i, j, i * 5 + j + 1);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 5; j++)
            matrix2Full.set(i, j, i * 5 + j + 2);

    DistributedMatrix matrix1(matrix1Full, numProcs);
    DistributedMatrix matrix2(matrix2Full, numProcs);

    Matrix result = matrix1.multiplyTransposed(matrix2);
    Matrix expected = matrix1Full * matrix2Full.transpose();
    assert(matricesEqual(result, expected, 1e-8));

    if (rank == 0)
        std::cout << "testMultiplyTransposed passed." << std::endl;
}

void testSum() {
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    Matrix matrixFull(3, 5);
    double total = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 5; j++) {
            double val = i * 5 + j + 1;
            matrixFull.set(i, j, val);
            total += val;
        }
    }

    DistributedMatrix matrix(matrixFull, numProcs);
    assert(approxEqual(matrix.sum(), total, 1e-8));

    if (rank == 0)
        std::cout << "testSum passed." << std::endl;
}

void testGather() {
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    Matrix testMatrix(4, 6);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 6; j++)
            testMatrix.set(i, j, i * 10 + j);

    DistributedMatrix distMatrix(testMatrix, numProcs);
    assert(matricesEqual(distMatrix.gather(), testMatrix));

    if (rank == 0)
        std::cout << "testGather passed." << std::endl;
}

void testGetAndSet() {
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if (numProcs == 1) {
        if (rank == 0)
            std::cout << "testGetAndSet skipped (requires multiple processes)." << std::endl;
        return;
    }

    Matrix testMatrix(2, numProcs);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < numProcs; j++)
            testMatrix.set(i, j, i * numProcs + j);

    DistributedMatrix distMatrix(testMatrix, numProcs);

    double val = distMatrix.get(1, rank);
    assert(approxEqual(val, 1 * numProcs + rank));

    distMatrix.set(1, rank, 99.0);
    assert(approxEqual(distMatrix.get(1, rank), 99.0));

    if (numProcs > 1) {
        int remoteRank = (rank + 1) % numProcs;

        bool threw = false;
        try { (void)distMatrix.get(1, remoteRank); }
        catch (std::exception&) { threw = true; }
        assert(threw);

        threw = false;
        try { distMatrix.set(1, remoteRank, 100.0); }
        catch (std::exception&) { threw = true; }
        assert(threw);
    }

    if (rank == 0)
        std::cout << "testGetAndSet passed." << std::endl;
}

void testCopyConstructor() {
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    Matrix testMatrix(3, 5);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 5; j++)
            testMatrix.set(i, j, i * 5 + j);

    DistributedMatrix original(testMatrix, numProcs);
    DistributedMatrix copy(original);

    assert(copy.numRows() == original.numRows());
    assert(copy.numCols() == original.numCols());
    assert(matricesEqual(original.getLocalData(), copy.getLocalData()));

    DistributedMatrix modified = copy.apply([](double x) { return 2 * x; });
    assert(!matricesEqual(original.gather(), modified.gather()));

    if (rank == 0)
        std::cout << "testCopyConstructor passed." << std::endl;
}

void testCommonOperations() {
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    Matrix testMatrix1(3, 4);
    Matrix testMatrix2(3, 4);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            testMatrix1.set(i, j, i * 4 + j + 1);
            testMatrix2.set(i, j, (i * 4 + j + 1) * 2);
        }
    }

    DistributedMatrix dist1(testMatrix1, numProcs);
    DistributedMatrix dist2(testMatrix2, numProcs);

    DistributedMatrix filled(testMatrix1, numProcs);
    filled.fill(3.14);
    Matrix filledGathered = filled.gather();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            assert(approxEqual(filledGathered.get(i, j), 3.14));

    assert(matricesEqual((dist1 + dist2).gather(), testMatrix1 + testMatrix2));
    assert(matricesEqual((dist2 - dist1).gather(), testMatrix2 - testMatrix1));
    assert(matricesEqual((dist1 * 3.0).gather(), testMatrix1 * 3.0));
    assert(matricesEqual(dist1.transpose(), testMatrix1.transpose()));

    DistributedMatrix subMulTest(testMatrix1, numProcs);
    subMulTest.sub_mul(2.0, dist2);
    Matrix expectedSubMul(testMatrix1);
    expectedSubMul.sub_mul(2.0, testMatrix2);
    assert(matricesEqual(subMulTest.gather(), expectedSubMul));

    if (rank == 0)
        std::cout << "testCommonOperations passed." << std::endl;
}

int main(int argc, char** argv) {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized)
        MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
        std::cout << "Starting DistributedMatrix tests..." << std::endl;

    try {
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
        testCommonOperations();

        if (rank == 0)
            std::cout << "All distributed matrix tests passed." << std::endl;
    }
    catch (std::exception& e) {
        if (rank == 0)
            std::cerr << "Test failed: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
