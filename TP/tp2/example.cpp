// Compile: mpicxx -O3 example.cpp -o example
// Run:     mpirun -np 4 ./example

#include <mpi.h>
#include <cstdio>
#include <vector>


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

	MPI_Finalize();
	return 0;
}