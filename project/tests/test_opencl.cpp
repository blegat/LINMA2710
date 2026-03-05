#include "matrix_opencl.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

bool verifyMatrix(const MatrixCL& mat, const std::vector<float>& expected, float epsilon = 1e-5f) {
    if (static_cast<size_t>(mat.numRows() * mat.numCols()) != expected.size())
        return false;
    std::vector<float> actual = mat.copyToHost();
    for (size_t i = 0; i < actual.size(); ++i)
        if (!approxEqual(actual[i], expected[i], epsilon))
            return false;
    return true;
}

cl::Context context;
cl::CommandQueue queue;

void setupOpenCL() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    assert(!platforms.empty());

    cl::Platform platform = platforms.front();
    std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty())
        platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
    assert(!devices.empty());

    cl::Device device = devices.front();
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    context = cl::Context(device);
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    MatrixCL::initializeKernels(context, {device});

    std::cout << "setupOpenCL passed." << std::endl;
}

void testFill() {
    MatrixCL mat(2, 3, context, queue);
    mat.fill(5.5f);
    assert(verifyMatrix(mat, {5.5f, 5.5f, 5.5f, 5.5f, 5.5f, 5.5f}));

    std::cout << "testFill passed." << std::endl;
}

void testCopyConstructorAndAssignment() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    MatrixCL original(2, 3, context, queue, &data);

    MatrixCL copy(original);
    assert(verifyMatrix(copy, data));

    MatrixCL assigned(1, 1, context, queue);
    assigned = original;
    assert(verifyMatrix(assigned, data));

    std::cout << "testCopyConstructorAndAssignment passed." << std::endl;
}

void testAddition() {
    std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> dataB = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    MatrixCL matA(2, 3, context, queue, &dataA);
    MatrixCL matB(2, 3, context, queue, &dataB);

    MatrixCL result = matA + matB;
    assert(verifyMatrix(result, {8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f}));

    std::cout << "testAddition passed." << std::endl;
}

void testSubtraction() {
    std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> dataB = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    MatrixCL matA(2, 3, context, queue, &dataA);
    MatrixCL matB(2, 3, context, queue, &dataB);

    MatrixCL result = matA - matB;
    assert(verifyMatrix(result, {-6.0f, -6.0f, -6.0f, -6.0f, -6.0f, -6.0f}));

    std::cout << "testSubtraction passed." << std::endl;
}

void testScalarMultiplication() {
    std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    MatrixCL matA(2, 3, context, queue, &dataA);

    MatrixCL result = matA * 3.0f;
    assert(verifyMatrix(result, {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f}));

    std::cout << "testScalarMultiplication passed." << std::endl;
}

void testTranspose() {
    std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    MatrixCL matA(2, 3, context, queue, &dataA);

    MatrixCL result = matA.transpose();
    assert(result.numRows() == 3 && result.numCols() == 2);
    assert(verifyMatrix(result, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}));

    std::cout << "testTranspose passed." << std::endl;
}

void testMatrixMultiplication() {
    std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> dataC = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    MatrixCL matA(2, 3, context, queue, &dataA);
    MatrixCL matC(3, 3, context, queue, &dataC);

    MatrixCL result = matA * matC;
    assert(verifyMatrix(result, {30.0f, 36.0f, 42.0f, 66.0f, 81.0f, 96.0f}));

    std::cout << "testMatrixMultiplication passed." << std::endl;
}

void testSubMul() {
    std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> dataB = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    MatrixCL matA(2, 3, context, queue, &dataA);
    MatrixCL matB(2, 3, context, queue, &dataB);

    matA.sub_mul(2.0f, matB);
    assert(verifyMatrix(matA, {-13.0f, -14.0f, -15.0f, -16.0f, -17.0f, -18.0f}));

    std::cout << "testSubMul passed." << std::endl;
}

int main() {
    try {
        setupOpenCL();
        testFill();
        testCopyConstructorAndAssignment();
        testAddition();
        testSubtraction();
        testScalarMultiplication();
        testTranspose();
        testMatrixMultiplication();
        testSubMul();
    } catch (const cl::BuildError& err) {
        std::cerr << "OpenCL Build Error: " << err.what() << " (" << err.err() << ")" << std::endl;
        for (const auto& pair : err.getBuildLog())
            std::cerr << "Build Log (" << pair.first.getInfo<CL_DEVICE_NAME>() << "):\n" << pair.second << std::endl;
        return 1;
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error: " << err.what() << " (" << err.err() << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "All OpenCL matrix tests passed." << std::endl;
    return 0;
}
