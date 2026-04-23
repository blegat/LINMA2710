#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_MAKE_VERSION(major, minor, patch) (((major) << 16) | ((minor) << 8) | (patch))
#include "CL/opencl.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>

// ============================================================
// OpenCL Kernel Sources
// ============================================================

// A simple kernel that fills a buffer with a constant value.
// This one is provided as an example.
const std::string kernel_source_fill = R"(
    __kernel void fill(__global float* y, float value, int N) {
        int i = get_global_id(0);
        if (i < N) {
            y[i] = value;
        }
    }
)";

// TODO 1: Complete this kernel.
// Each work-item should compute y[i] = x[i] * x[i].
const std::string kernel_source_square = R"(
    __kernel void square(__global const float* x,
                         __global float* y,
                         int N) {
        // TODO 1
    }
)";

// TODO 3: Complete this kernel.
// Each work-item should apply f(v) = v*v for n_iter iterations,
// starting from x[i], and store the final result in y[i].
const std::string kernel_source_square_iter = R"(
    __kernel void square_iter(__global const float* x,
                              __global float* y,
                              int N,
                              int n_iter) {
        // TODO 3
    }
)";

// Kernel for in-place squaring (used in Part 4).
// Each call squares every element in place.
const std::string kernel_source_square_inplace = R"(
    __kernel void square_inplace(__global float* x, int N) {
        int i = get_global_id(0);
        if (i < N) {
            x[i] = x[i] * x[i];
        }
    }
)";

// ============================================================
// CPU reference implementations
// ============================================================

void cpu_square(const std::vector<float>& x, std::vector<float>& y) {
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = x[i] * x[i];
}

void cpu_square_iter(const std::vector<float>& x, std::vector<float>& y, int n_iter) {
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        for (int k = 0; k < n_iter; ++k)
            v = v * v;
        y[i] = v;
    }
}

// ============================================================
// Utility
// ============================================================

cl::Program buildProgram(cl::Context& context, const std::vector<cl::Device>& devices,
                         const std::string& source, const std::string& name) {
    cl::Program program(context, source);
    try {
        program.build(devices);
    } catch (const cl::BuildError& err) {
        std::cerr << "Build error for kernel '" << name << "':" << std::endl;
        for (const auto& pair : err.getBuildLog())
            std::cerr << pair.second << std::endl;
        throw;
    }
    return program;
}

bool verify(const std::vector<float>& gpu, const std::vector<float>& cpu, float tol = 1e-5f) {
    for (size_t i = 0; i < gpu.size(); ++i) {
        if (std::abs(gpu[i] - cpu[i]) > tol * std::max(1.0f, std::abs(cpu[i]))) {
            std::cerr << "Mismatch at index " << i
                      << ": GPU=" << gpu[i] << " CPU=" << cpu[i] << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    const int N = 20'000'000;
    const int N_ITER = 50;

    // OpenCL setup
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platform found." << std::endl;
        return 1;
    }

    cl::Platform platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty())
        platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
    if (devices.empty()) {
        std::cerr << "No OpenCL device found." << std::endl;
        return 1;
    }

    cl::Device device = devices.front();
    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Build kernels
    cl::Kernel kernel_fill(buildProgram(context, {device}, kernel_source_fill, "fill"), "fill");
    cl::Kernel kernel_square(buildProgram(context, {device}, kernel_source_square, "square"), "square");
    cl::Kernel kernel_square_iter(buildProgram(context, {device}, kernel_source_square_iter, "square_iter"), "square_iter");
    cl::Kernel kernel_square_inplace(buildProgram(context, {device}, kernel_source_square_inplace, "square_inplace"), "square_inplace");

    // Prepare host data
    std::vector<float> h_x(N), h_y_gpu(N), h_y_cpu(N);
    for (int i = 0; i < N; ++i)
        h_x[i] = 1.0f - 0.001f * (i % 1000);  // values in (-1, 1] so iterates don't explode

    // Create device buffers
    cl::Buffer d_x(context, CL_MEM_READ_ONLY, N * sizeof(float));
    cl::Buffer d_y(context, CL_MEM_WRITE_ONLY, N * sizeof(float));

    // ================================================================
    // Part 2: Single squaring  y[i] = x[i]^2
    // ================================================================
    std::cout << "\n=== Part 2: Single squaring ===" << std::endl;

    // Transfer data to device
    auto t_transfer_start = std::chrono::high_resolution_clock::now();
    queue.enqueueWriteBuffer(d_x, CL_TRUE, 0, N * sizeof(float), h_x.data());
    auto t_transfer_end = std::chrono::high_resolution_clock::now();
    double transfer_time = std::chrono::duration<double, std::milli>(t_transfer_end - t_transfer_start).count();

    // Example: fill buffer with zeros using the provided fill kernel
    kernel_fill.setArg(0, d_y);
    kernel_fill.setArg(1, 0.0f);
    kernel_fill.setArg(2, N);
    queue.enqueueNDRangeKernel(kernel_fill, cl::NullRange, cl::NDRange(N), cl::NullRange);
    queue.finish();

    // TODO 2: Launch the square kernel
    //   - Set the 3 arguments of kernel_square: d_x, d_y, N
    //   - Enqueue the kernel with global size N
    //   - Call queue.finish() to wait for completion
    // Follow the same pattern as the fill kernel above.

    // TODO 2 (start)

    // TODO 2 (end)

    // Read back results
    auto t_readback_start = std::chrono::high_resolution_clock::now();
    queue.enqueueReadBuffer(d_y, CL_TRUE, 0, N * sizeof(float), h_y_gpu.data());
    auto t_readback_end = std::chrono::high_resolution_clock::now();
    double readback_time = std::chrono::duration<double, std::milli>(t_readback_end - t_readback_start).count();

    // CPU reference
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    cpu_square(h_x, h_y_cpu);
    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(t_cpu_end - t_cpu_start).count();

    if (verify(h_y_gpu, h_y_cpu))
        std::cout << "Single squaring: PASSED" << std::endl;
    else
        std::cout << "Single squaring: FAILED" << std::endl;

    std::cout << "  Host-to-device transfer: " << transfer_time << " ms" << std::endl;
    std::cout << "  Device-to-host transfer: " << readback_time << " ms" << std::endl;
    std::cout << "  CPU time:                " << cpu_time << " ms" << std::endl;

    // ================================================================
    // Part 3: Iterated dynamics  y[i] = f^{50}(x[i])
    // ================================================================
    std::cout << "\n=== Part 3: Iterated dynamics (loop inside kernel) ===" << std::endl;

    queue.enqueueWriteBuffer(d_x, CL_TRUE, 0, N * sizeof(float), h_x.data());

    // TODO 4: Launch the square_iter kernel
    //   - Set the 4 arguments: d_x, d_y, N, N_ITER
    //   - Enqueue with global size N
    //   - finish the queue

    auto t_gpu_start = std::chrono::high_resolution_clock::now();

    // TODO 4 (start)

    // TODO 4 (end)

    auto t_gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_iter_time = std::chrono::duration<double, std::milli>(t_gpu_end - t_gpu_start).count();

    queue.enqueueReadBuffer(d_y, CL_TRUE, 0, N * sizeof(float), h_y_gpu.data());

    // CPU reference
    t_cpu_start = std::chrono::high_resolution_clock::now();
    cpu_square_iter(h_x, h_y_cpu, N_ITER);
    t_cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_iter_time = std::chrono::duration<double, std::milli>(t_cpu_end - t_cpu_start).count();

    if (verify(h_y_gpu, h_y_cpu))
        std::cout << "Iterated dynamics: PASSED" << std::endl;
    else
        std::cout << "Iterated dynamics: FAILED" << std::endl;

    std::cout << "  GPU compute time: " << gpu_iter_time << " ms" << std::endl;
    std::cout << "  CPU compute time: " << cpu_iter_time << " ms" << std::endl;
    std::cout << "  Speedup:          " << cpu_iter_time / gpu_iter_time << "x" << std::endl;

    // ================================================================
    // Part 4: Kernel launch overhead (50 launches from host)
    // ================================================================
    std::cout << "\n=== Part 4: Iterated dynamics (50 kernel launches) ===" << std::endl;

    // We use a single buffer for in-place squaring
    cl::Buffer d_z(context, CL_MEM_READ_WRITE, N * sizeof(float));
    queue.enqueueWriteBuffer(d_z, CL_TRUE, 0, N * sizeof(float), h_x.data());

    // TODO 5: Launch the square_inplace kernel 50 times in a loop
    //   - For each iteration, set the arguments and enqueue
    //   - Call queue.finish() AFTER all 50 launches (not inside the loop)

    auto t_multi_start = std::chrono::high_resolution_clock::now();

    // TODO 5 (start)

    // TODO 5 (end)

    auto t_multi_end = std::chrono::high_resolution_clock::now();
    double gpu_multi_time = std::chrono::duration<double, std::milli>(t_multi_end - t_multi_start).count();

    queue.enqueueReadBuffer(d_z, CL_TRUE, 0, N * sizeof(float), h_y_gpu.data());

    if (verify(h_y_gpu, h_y_cpu))
        std::cout << "50 launches: PASSED" << std::endl;
    else
        std::cout << "50 launches: FAILED" << std::endl;

    std::cout << "  GPU compute time (50 launches): " << gpu_multi_time << " ms" << std::endl;
    std::cout << "  GPU compute time (1 launch):    " << gpu_iter_time << " ms" << std::endl;
    std::cout << "  CPU compute time:               " << cpu_iter_time << " ms" << std::endl;

    return 0;
}
