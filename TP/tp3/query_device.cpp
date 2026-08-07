#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_MAKE_VERSION(major, minor, patch) (((major) << 16) | ((minor) << 8) | (patch))
#include "CL/opencl.hpp"

#include <iostream>
#include <vector>
#include <string>

int main() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return 1;
    }

    for (size_t p = 0; p < platforms.size(); ++p) {
        std::cout << "=== Platform " << p << " ===" << std::endl;
        std::cout << "  Name:    " << platforms[p].getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::cout << "  Vendor:  " << platforms[p].getInfo<CL_PLATFORM_VENDOR>() << std::endl;
        std::cout << "  Version: " << platforms[p].getInfo<CL_PLATFORM_VERSION>() << std::endl;

        std::vector<cl::Device> devices;
        platforms[p].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for (size_t d = 0; d < devices.size(); ++d) {
            const auto& dev = devices[d];
            std::cout << "\n  --- Device " << d << " ---" << std::endl;
            std::cout << "    Name:              " << dev.getInfo<CL_DEVICE_NAME>() << std::endl;

            cl_device_type type = dev.getInfo<CL_DEVICE_TYPE>();
            std::string type_str = (type == CL_DEVICE_TYPE_GPU) ? "GPU" :
                                   (type == CL_DEVICE_TYPE_CPU) ? "CPU" : "OTHER";
            std::cout << "    Type:              " << type_str << std::endl;

            std::cout << "    Compute units:     " << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "    Max work-group:    " << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
            std::cout << "    Clock frequency:   " << dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << " MHz" << std::endl;
            std::cout << "    Global memory:     " << dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024*1024) << " MB" << std::endl;
            std::cout << "    Local memory:      " << dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << " KB" << std::endl;
            std::cout << std::endl;
        }
    }

    return 0;
}
