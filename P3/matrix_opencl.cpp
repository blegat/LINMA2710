// P3/matrix_opencl.cpp
#include "matrix_opencl.hpp"
#include <iostream> // For error reporting during kernel build
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream> // For building kernel source string
#include <memory> 
#include <mutex>  

// ---------------------------------------------------------------------------
// Static Member Definitions
// ---------------------------------------------------------------------------
std::shared_ptr<KernelCache> MatrixCL::kernels_ = nullptr;

// ---------------------------------------------------------------------------
// Helper Function: Load and Build OpenCL Program (Used only during init)
// ---------------------------------------------------------------------------
cl::Program loadAndBuildProgram(cl::Context context,
                                const std::vector<cl::Device>& devices,
                                const std::string& sourceCode,
                                const std::string& kernel_name_for_error)
{
    cl::Program program(context, sourceCode);
    try {
        program.build(devices);
    } catch (const cl::BuildError& err) {
        std::cerr << "OpenCL Build Error for kernel source '" << kernel_name_for_error << "':\n"
                  << err.what() << "(" << err.err() << ")" << std::endl;
        for (const auto& pair : err.getBuildLog()) {
            std::cerr << "Device " << pair.first.getInfo<CL_DEVICE_NAME>() << ":" << std::endl;
            std::cerr << pair.second << std::endl;
        }
        throw;
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error during program build for '" << kernel_name_for_error << "': "
                  << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    }
    return program;
}

// ---------------------------------------------------------------------------
// OpenCL Kernel Source Code Strings
// ---------------------------------------------------------------------------
const std::string kernel_source_fill = R"(
    __kernel void fill(__global float* matrix, float value, int rows, int cols) {
        TODO (provided function signatures can be changed)
    }
)";
const std::string kernel_source_add = R"(
    __kernel void add(__global const float* A, __global const float* B, __global float* C, int rows, int cols) {
        TODO (provided function signatures can be changed)
    }
)";
const std::string kernel_source_sub_mul = R"(
    __kernel void sub_mul(__global float* A, __global const float* B, float scalar, int rows, int cols) {
        TODO (provided function signatures can be changed)
    }
)";
const std::string kernel_source_transpose = R"(
    __kernel void transpose(__global const float* A, __global float* B, int A_rows, int A_cols) {
        TODO (provided function signatures can be changed)
    }
)";
const std::string kernel_source_matrix_mul = R"(
    __kernel void matrix_mul(__global const float* A, __global const float* B, __global float* C, int A_rows, int A_cols, int B_cols) {
        TODO (provided function signatures can be changed)
    }
)";
const std::string kernel_source_sigmoid = R"(
    __kernel void sigmoid(__global const float* input, __global float* output, int rows, int cols) {
        TODO (provided function signatures can be changed)
    }
)";
const std::string kernel_source_sigmoid_backward = R"(
    __kernel void sigmoid_backward(__global float* grad_acc, __global const float* input, __global const float* out_grad, int rows, int cols) {
        TODO (provided function signatures can be changed)
    }
)";
const std::string kernel_source_bce_elementwise = R"(
     __kernel void bce_elementwise(__global const float* predictions, __global const float* targets, __global float* elementwise_loss, int rows, int cols, float epsilon) {
        TODO (provided function signatures can be changed)
    }
)";
const std::string kernel_source_bce_backward = R"(
    __kernel void bce_backward(__global float* grad_acc, __global const float* predictions, __global const float* targets, int rows, int cols, float epsilon, float inv_num_elements) {
        int idx = get_global_id(0); int total_elements = rows * cols;
        if (idx < total_elements) {
            float pred = predictions[idx]; float targ = targets[idx];
            float denominator1 = max(pred + epsilon, epsilon); // Avoid exactly zero denominator
            float denominator2 = max(1.0f - pred + epsilon, epsilon);
            float bce_grad = -(targ / denominator1 - (1.0f - targ) / denominator2);
            grad_acc[idx] += inv_num_elements * bce_grad;
        }
    }
)";

// ---------------------------------------------------------------------------
// KernelCache Implementation
// ---------------------------------------------------------------------------
void KernelCache::compileKernels(cl::Context context, const std::vector<cl::Device>& devices) {
    if (initialized) return; // Already compiled

    std::cout << "Compiling OpenCL kernels..." << std::endl;
    try {
        cl::Program prog_fill = loadAndBuildProgram(context, devices, kernel_source_fill, "fill");
        kernel_fill = cl::Kernel(prog_fill, "fill");

        cl::Program prog_add = loadAndBuildProgram(context, devices, kernel_source_add, "add");
        kernel_add = cl::Kernel(prog_add, "add");

        cl::Program prog_sub_mul = loadAndBuildProgram(context, devices, kernel_source_sub_mul, "sub_mul");
        kernel_sub_mul = cl::Kernel(prog_sub_mul, "sub_mul");

        cl::Program prog_transpose = loadAndBuildProgram(context, devices, kernel_source_transpose, "transpose");
        kernel_transpose = cl::Kernel(prog_transpose, "transpose");

        cl::Program prog_matrix_mul = loadAndBuildProgram(context, devices, kernel_source_matrix_mul, "matrix_mul");
        kernel_matrix_mul = cl::Kernel(prog_matrix_mul, "matrix_mul");

        cl::Program prog_sigmoid = loadAndBuildProgram(context, devices, kernel_source_sigmoid, "sigmoid");
        kernel_sigmoid = cl::Kernel(prog_sigmoid, "sigmoid");

        cl::Program prog_sigmoid_bw = loadAndBuildProgram(context, devices, kernel_source_sigmoid_backward, "sigmoid_backward");
        kernel_sigmoid_backward = cl::Kernel(prog_sigmoid_bw, "sigmoid_backward");

        cl::Program prog_bce_ew = loadAndBuildProgram(context, devices, kernel_source_bce_elementwise, "bce_elementwise");
        kernel_bce_elementwise = cl::Kernel(prog_bce_ew, "bce_elementwise");

        cl::Program prog_bce_bw = loadAndBuildProgram(context, devices, kernel_source_bce_backward, "bce_backward");
        kernel_bce_backward = cl::Kernel(prog_bce_bw, "bce_backward");

        initialized = true;
        std::cout << "OpenCL kernels compiled successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to compile one or more OpenCL kernels. Aborting." << std::endl;
        throw; // Re-throw to potentially stop the program
    }
}

// ---------------------------------------------------------------------------
// MatrixCL Static Methods Implementation
// ---------------------------------------------------------------------------

// Ensures kernel cache is initialized exactly once.
void MatrixCL::initializeKernels(cl::Context context, const std::vector<cl::Device>& devices) {
    try {
        // Only initialize if not already done
        if (!kernels_ || !kernels_->initialized) {
            std::cout << "Creating and compiling kernels directly..." << std::endl;
            kernels_ = std::make_shared<KernelCache>();
            kernels_->compileKernels(context, devices);
        }
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in direct kernel initialization: " 
                  << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Exception in direct kernel initialization: " << e.what() << std::endl;
        throw;
    }
}


// ---------------------------------------------------------------------------
// MatrixCL Class Implementation
// ---------------------------------------------------------------------------

size_t MatrixCL::buffer_size_bytes() const {
    return static_cast<size_t>(rows_) * cols_ * sizeof(float);
}







/* TODO */







void MatrixCL::binary_cross_entropy_backward(const MatrixCL& predictions, const MatrixCL& targets) {
     if (rows_ != predictions.numRows() || cols_ != predictions.numCols() ||
        rows_ != targets.numRows() || cols_ != targets.numCols()) {
        throw std::invalid_argument("Matrix dimensions must match for binary_cross_entropy_backward.");
    }
    if (context_() != predictions.getContext()() || queue_() != predictions.getQueue()() ||
        context_() != targets.getContext()() || queue_() != targets.getQueue()()) {
         throw std::runtime_error("Cannot perform BCE backward update on matrices from different OpenCL contexts or queues.");
    }

    size_t num_elements = static_cast<size_t>(rows_) * cols_;
     if (num_elements == 0) return;

    const float epsilon = 1e-8f;
    const float inv_num_elements = 1.0f / static_cast<float>(num_elements);

    try {
        cl::Kernel kernel = kernels_->kernel_bce_backward; // Use cached kernel

        kernel.setArg(0, this->buffer_);            // gradient_accumulator (read-write)
        kernel.setArg(1, predictions.getBuffer());  // predictions (read-only)
        kernel.setArg(2, targets.getBuffer());      // targets (read-only)
        kernel.setArg(3, rows_);
        kernel.setArg(4, cols_);
        kernel.setArg(5, epsilon);
        kernel.setArg(6, inv_num_elements);

        size_t global_work_size = num_elements;
        queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);

    } catch (const cl::Error& err) {
        throw std::runtime_error("OpenCL error during binary_cross_entropy_backward: " + std::string(err.what()) + " (" + std::to_string(err.err()) + ")");
    } catch (const std::runtime_error& err) {
         throw std::runtime_error("Error during binary_cross_entropy_backward: " + std::string(err.what()));
    }
}