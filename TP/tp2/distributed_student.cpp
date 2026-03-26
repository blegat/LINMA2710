// Compile: mpicxx -[O1 or O2 or O3] distributed.cpp -o distributed
// Run:     mpirun -np [NUM_OF_PROCESSES] ./distributed

#include <mpi.h>

#include <cstdio>
#include <vector>

static void compute_bound_local(const std::vector<double> &x_local, std::vector<double> &y_local) {
	const int n_local = static_cast<int>(x_local.size());
	for (int i = 0; i < n_local; ++i) {
		double v = x_local[i];
		for (int k = 0; k < 1; ++k) {
			v = v * v;
		}
		y_local[i] = v;
	}
}

int main(int argc, char **argv) {
	
	//////////////////////////////////////////////////////////////
    // Initialization of MPI environment and process information
	MPI_Init(&argc, &argv);

	// From now on, you are on a specific MPI process, 
	// with its own id (procid) and the total number of processes (nprocs).
	int nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	int procid;
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);

	// Get and print processor name for each process.
	// This is just to show that processes may be running on different nodes.
	int name_length = MPI_MAX_PROCESSOR_NAME;
	char proc_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Get_processor_name(proc_name, &name_length);
	std::printf("Process %d/%d is running on node <<%s>>\n", procid, nprocs, proc_name);

	
	const int N = 20000000;

	/////////////////////////////


	/*	1)
		Initialize here informations about how to split the global problem 
		into local chunks.
		Try to distribute it uniformly, but be aware that N may not be divisible by nprocs.
	*/

	// TODO
	
	

	/* 2)
		Only rank 0 owns full input/output arrays. Create the full x and y vectors.
		Hint: you defined earlier a variable giving the process currently running.
	*/

	// TODO



	/* 3)
		Split initial conditions x into chunks.
	*/

	// TODO
	
	

	/* 4)
		Each process computes its local chunk.
	*/

	// TODO

	

	/* 5)
		Get the results back. On which process are you storing the vector of final points ?
	*/

	// TODO


	// 6) Print some results on rank 0 to check that the computation was done.
	if (procid == 0) {
		
		std::printf("\nCompleted on rank 0.\n");
		std::printf("y[123] = %.17g\n", y[123]);
		std::printf("y[N-1] = %.17g\n", y[N - 1]);
	}

	// Do not forget to free the memory!
	MPI_Finalize();
	return 0;
}
