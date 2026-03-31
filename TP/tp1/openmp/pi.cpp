// C++ Program to implement
// Parallel Programming
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>


// Computes the value of pi using a serial computation.


double compute_pi_serial(long num_steps)
{
    double sum = 0.0;
    for (long i = 0; i < num_steps; i+=2) {
        sum += 1.0/(2*i+1.0);
    }
    for (long i = 1; i < num_steps; i+=2) {
        sum -= 1.0/(2*i+1.0);
    }
    return 4.0 * sum;
}

double compute_pi_parallel(long num_steps, int n_threads)
{
    double sum = 0.0;

    #pragma omp parallel for reduction(+ : sum) num_threads(n_threads)
    for (long i = 0; i < num_steps; i+=2) {
        sum += 1.0/(2*i+1.0);
    }
    #pragma omp parallel for reduction(- : sum) num_threads(n_threads)
    for (long i = 1; i < num_steps; i+=2) {
        sum -= 1.0/(2*i+1.0);
    }
    return 4.0 * sum;
}

// Driver function
int main()
{
    const long num_steps = 1000000000L;
    std::ofstream csv_file("pi_results.csv");
    csv_file << "N Threads,N Terms,Serial,Parallel\n";
    for (size_t n_threads = 1; n_threads <= 64; n_threads *= 2)
    {
        
    
    

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
    double pi_parallel = compute_pi_parallel(num_steps, n_threads);
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
    
    // TODO : Store the results in a CSV file for later analysis using fstream.
    // Store the results in a CSV file for later analysis using fstream.
    csv_file << n_threads << "," << num_steps << "," << serial_duration.count() << "," << parallel_duration.count() << "\n";
    }

    csv_file.close();
    return 0;
}