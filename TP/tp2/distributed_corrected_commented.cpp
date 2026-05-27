// Compile: mpicxx -O3 distributed_corrected.cpp -o distributed_corrected
// Run:     mpirun -np 4 ./distributed_corrected

#include <mpi.h>
#include <cstdio>
#include <vector>

/*
 * Compute y_local[i] = x_local[i]_50, or whatever discrete map/number of steps you want to apply.
 * for every value owned by the current MPI process.
 *
 * This function is purely local:
 * - it does not communicate,
 * - it only uses the portion of the data stored on the current process.
 */
static void compute_bound_local(const std::vector<double> &x_local, std::vector<double> &y_local) 
{
    const int n_local = static_cast<int>(x_local.size());

    const int num_steps = 50; // You can change this to increase the computational load.

    for (int i=0; i<n_local; ++i) {
        double v = x_local[i];

        for (int k=0; k<num_steps; ++k) {
            v = v*v;
        }

        y_local[i] = v;
    }
}

int main(int argc, char **argv) 
{

    // Every MPI program must begin with MPI_Init and end with MPI_Finalize.
    MPI_Init(&argc, &argv);

    // Total number of MPI processes launched in MPI_COMM_WORLD.
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Rank (identifier) of the current process in MPI_COMM_WORLD.
    // Ranks go from 0 to nprocs-1.
    int procid;
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);

    // Retrieve the machine name on which this process is running.
    int name_length = MPI_MAX_PROCESSOR_NAME;
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(proc_name, &name_length);

    std::printf("Process %d/%d is running on node <<%s>>\n", procid, nprocs, proc_name);

    // Global problem size.
    // The goal of this example is to show:
    // 1) how to distribute a vector across processes,
    // 2) how each process computes locally,
    // 3) how to gather the result back on rank 0.
    const int N = 20000000;

    /*
     * counts[p] = number of entries sent to process p
     * displs[p] = starting index of process p inside the global vector
     *
     * Example:
     *   if N = 10 and nprocs = 3
     *   then counts could be [4, 3, 3]
     *   and displs       = [0, 4, 7]
     *
     * This information is required by MPI_Scatterv and MPI_Gatherv
     * because the chunks may have different sizes.
     */
    std::vector<int> counts(nprocs);
    std::vector<int> displs(nprocs);

    // Uniform distribution:
    // - every process gets at least 'base' values,
    // - the first 'rem' processes receive one extra value.
    const int base = N / nprocs;
    const int rem = N % nprocs;

    for (int p=0; p<nprocs; ++p) {
        counts[p] = base+(p<rem ? 1 : 0); // This is a shortcut to say "if p<rem then counts[p] = base+1 else counts[p] = base".
    }

    // Build the displacement array from the counts array.
    displs[0] = 0;
    for (int p=1; p<nprocs; ++p) {
        displs[p] = displs[p-1] + counts[p-1];
    }

    // Number of values stored on the current process.
    const int n_local = counts[procid];

    // Local input/output vectors.
    // Each process only allocates memory for its own chunk.
    std::vector<double> x_local(n_local);
    std::vector<double> y_local(n_local);

    /*
     * Only rank 0 stores the full global vectors x and y.
     *
     * - x is the input vector to distribute
     * - y will receive the gathered results
     *
     * Other ranks keep these vectors empty, which avoids useless memory usage.
     */
    std::vector<double> x;
    std::vector<double> y;

    if (procid==0) {
        x.resize(N);
        y.resize(N);

        // Build the global initial condition vector.
        // If you plan to do a large number of steps, you may want to use values smaller than 1 to avoid overflow.
        for (int i=0; i<N; ++i) {
            x[i] = static_cast<double>(i); 
        }
    }

    /*
     * Step 1: distribute x from rank 0 to all processes.
     *
     * We use MPI_Scatterv instead of MPI_Scatter because the local sizes
     * are not necessarily identical when N is not divisible by nprocs.
     *
     * On rank 0:
     *   send buffer = x.data()
     *
     * On other ranks:
     *   send buffer is ignored, so nullptr is acceptable.
     *
     * Every process receives exactly n_local values into x_local.
     * Check the documentation of MPI_Scatterv for more details on the arguments.
     */
    MPI_Scatterv(
        procid == 0 ? x.data() : nullptr,
        counts.data(),
        displs.data(),
        MPI_DOUBLE,
        x_local.data(),
        n_local,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Step 2: each process computes on its own local data only.
    // This is the parallel part of the program.
    compute_bound_local(x_local, y_local);

    /*
     * Step 3: gather all local results back into y on rank 0.
     *
     * Again, MPI_Gatherv is needed because local chunk sizes may differ.
     *
     * Each process sends its local result y_local,
     * and rank 0 reconstructs the full global vector y.
     */
    MPI_Gatherv(
        y_local.data(),
        n_local,
        MPI_DOUBLE,
        procid == 0 ? y.data() : nullptr,
        counts.data(),
        displs.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Simple verification on rank 0.
    // To check that it works, you can set the number of steps to a small value (e.g. 1) and check that y[i] = x[i]*x[i].
    if (procid == 0) {
        std::printf("\nGather completed on rank 0.\n");
        std::printf("y[123] = %.17g\n", y[123]);
        std::printf("y[N-1] = %.17g\n", y[N - 1]);
    }

    // Cleanly terminate the MPI environment.
    MPI_Finalize();
    return 0;
}