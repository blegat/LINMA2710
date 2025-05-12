// P3/matrix_opencl.hpp
#ifndef MATRIX_OPENCL_HPP
#define MATRIX_OPENCL_HPP

// Define CL_HPP_ENABLE_EXCEPTIONS and CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include "CL/opencl.hpp" // Use OpenCL C++ bindings

#include <vector>
#include <string>
#include <stdexcept>
#include <memory> 
#include <mutex>

// --- Forward Declarations ---
class MatrixCL;

// --- Kernel Cache Structure ---
// Holds pre-compiled OpenCL kernels for reuse.
struct KernelCache {
    cl::Kernel kernel_fill;
    cl::Kernel kernel_add;
    cl::Kernel kernel_sub_mul;
    cl::Kernel kernel_transpose;
    cl::Kernel kernel_matrix_mul;
    cl::Kernel kernel_sigmoid;
    cl::Kernel kernel_sigmoid_backward;
    cl::Kernel kernel_bce_elementwise;
    cl::Kernel kernel_bce_backward;

    // Flag to indicate if kernels have been compiled
    bool initialized = false;

    // Function to compile all kernels
    void compileKernels(cl::Context context, const std::vector<cl::Device>& devices);
};


// --- MatrixCL Class ---
class MatrixCL
{
private:
    int rows_, cols_;
    cl::Context context_;      // Reference to the OpenCL context
    cl::CommandQueue queue_;   // Reference to the command queue
    cl::Buffer buffer_;       // OpenCL buffer on the device

    // --- Static Kernel Cache ---
    // Holds compiled kernels, shared by all MatrixCL instances.
    static std::shared_ptr<KernelCache> kernels_;

    size_t buffer_size_bytes() const; // Helper for buffer size, returns rows*cols*sizeof(float)


public:
    // --- Initialization ---
    // Must be called once *after* OpenCL context/device setup, *before* any MatrixCL ops.
    // This ensures the kernel cache is ready.
    static void initializeKernels(cl::Context context, const std::vector<cl::Device>& devices);

    // --- Constructors ---
    // Creates a matrix initialized with zero elements or optional initial data
    MatrixCL(int rows, int cols, cl::Context context, cl::CommandQueue queue, const std::vector<float>* initial_data = nullptr);
    
    // Copy constructor (performs device-to-device copy)
    MatrixCL(const MatrixCL& other);

    // Destructor (cl::Buffer manages its own release via RAII)
    ~MatrixCL() = default; // RAII handles buffer release

    // Copy assignment operator
    MatrixCL& operator=(const MatrixCL& other);

    // Getters
    int numRows() const;
    int numCols() const;
    cl::Context getContext() const;
    cl::CommandQueue getQueue() const;
    const cl::Buffer& getBuffer() const; // Read-only access to buffer

    // Copy data from device buffer back to host in an std::vector
    std::vector<float> copyToHost() const;

    // --- Operations (Must be implemented with OpenCL Kernels) ---
    // Fill the entire matrix with a single value
    void fill(float value);

    // Addition: C = A + B
    MatrixCL operator+(const MatrixCL& other) const;
    
    // Matrix multiplication: C = A * B
    MatrixCL operator*(const MatrixCL& other) const;

    // Transpose: returns a new Matrix that is the transpose (B = A^T)
    MatrixCL transpose() const;

    // Subtract the product of a scalar and a given matrix: "this = this - scalar * other"
    // Performs the operation in-place on 'this' matrix's buffer.
    void sub_mul(float scalar, const MatrixCL& other);

    // Applies sigmoid element-wise: Returns a matrix containing sigmoid(this)
    MatrixCL sigmoid() const;
    // Calculates gradient for sigmoid and adds it to 'this' matrix (gradient accumulator).
    void sigmoid_backward(const MatrixCL& input_values, const MatrixCL& output_gradient);

    // Calculates Binary Cross-Entropy Loss between the entries of 'this' matrix and the target matrix element-wise. Returns a MatrixCL containing the losses.
    MatrixCL binary_cross_entropy(const MatrixCL& targets) const;
    // Calculates the gradient of BCE w.r.t predictions and adds it to 'this' matrix. Note: divides the gradient by the number of elements.
    void binary_cross_entropy_backward(const MatrixCL& predictions, const MatrixCL& targets);
};


#endif // MATRIX_OPENCL_HPP
