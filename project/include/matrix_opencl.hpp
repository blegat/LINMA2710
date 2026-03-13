#ifndef MATRIX_OPENCL_HPP
#define MATRIX_OPENCL_HPP

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_MAKE_VERSION(major, minor, patch) (((major) << 16) | ((minor) << 8) | (patch))
#include "CL/opencl.hpp"

#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

// --- Kernel Cache Structure ---
// Holds pre-compiled OpenCL kernels for reuse.
struct KernelCache {
    cl::Kernel kernel_fill;
    cl::Kernel kernel_add;
    cl::Kernel kernel_sub_mul;
    cl::Kernel kernel_transpose;
    cl::Kernel kernel_matrix_mul;

    bool initialized = false;

    void compileKernels(cl::Context context, const std::vector<cl::Device>& devices);
};

// --- MatrixCL Class ---
class MatrixCL
{
private:
    int rows_, cols_;
    cl::Context context_;
    cl::CommandQueue queue_;
    cl::Buffer buffer_;

    static std::shared_ptr<KernelCache> kernels_;

    size_t buffer_size_bytes() const;

public:
    // --- Initialization ---
    // Must be called once *after* OpenCL context/device setup, *before* any MatrixCL ops.
    static void initializeKernels(cl::Context context, const std::vector<cl::Device>& devices);

    // --- Constructors & Assignment ---
    MatrixCL(int rows, int cols, cl::Context context, cl::CommandQueue queue, const std::vector<float>* initial_data = nullptr);
    MatrixCL(const MatrixCL& other);  // Device-to-device copy
    ~MatrixCL() = default;
    MatrixCL& operator=(const MatrixCL& other);

    // --- Common API (shared with Matrix and DistributedMatrix) ---
    //     All operations are performed on the GPU via OpenCL kernels.

    int numRows() const;
    int numCols() const;

    void fill(float value);

    MatrixCL operator+(const MatrixCL& other) const;
    MatrixCL operator-(const MatrixCL& other) const;
    MatrixCL operator*(const MatrixCL& other) const; // Matrix multiplication
    MatrixCL operator*(float scalar) const;           // Scalar multiplication

    MatrixCL transpose() const;

    // this = this - scalar * other
    void sub_mul(float scalar, const MatrixCL& other);

    // --- OpenCL-specific operations ---

    cl::Context getContext() const;
    cl::CommandQueue getQueue() const;
    const cl::Buffer& getBuffer() const;

    // Copy data from device buffer back to host
    std::vector<float> copyToHost() const;
};

#endif // MATRIX_OPENCL_HPP
