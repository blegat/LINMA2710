// C++ Program to implement
// Parallel Programming
#include <chrono>
#include <iostream>
#include <omp.h>

// Computes the value of pi using a serial computation.

double compute_pi_serial(long num_steps)
{
    double sum = 0.0;
    // TODO
}

double compute_pi_parallel(long num_steps)
{
    double sum = 0.0;

    // TODO
}

// Driver function
int main()
{
    const long num_steps = 1000000000L;

    // Compute pi using serial computation and time it.
    auto start_time
        = std::chrono::high_resolution_clock::now();
    double pi_serial = compute_pi_serial(num_steps);
    auto end_time
        = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> serial_duration
        = end_time - start_time;

    // Compute pi using parallel computation and time it.
    start_time = std::chrono::high_resolution_clock::now();
    double pi_parallel = compute_pi_parallel(num_steps);
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> parallel_duration
        = end_time - start_time;

    std::cout << "Serial result: " << pi_serial
              << std::endl;
    std::cout << "Parallel result: " << pi_parallel
              << std::endl;
    std::cout << "Serial duration: "
              << serial_duration.count() << " seconds"
              << std::endl;
    std::cout << "Parallel duration: "
              << parallel_duration.count() << " seconds"
              << std::endl;
    std::cout << "Speedup: "
              << serial_duration.count()
                     / parallel_duration.count()
              << std::endl;
    
    // TODO : Store the results in a CSV file for later analysis
    return 0;
}