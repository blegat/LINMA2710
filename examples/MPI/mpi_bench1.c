#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  MPI_Comm comm = MPI_COMM_WORLD;
  int nprocs, procid;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(comm,&nprocs);
  MPI_Comm_rank(comm,&procid);
  //codesnippet mpi_bench1
  int tag = 0;
  for(int size = 1; size <= (1<<20); size <<= 1){
    char* buf = malloc(size);
    if (procid == 0) {
        double tic = MPI_Wtime();
        MPI_Send(buf, size, MPI_CHAR, procid + 1, tag++, comm);
        double toc = MPI_Wtime();
        printf("[%d] I have send %d B in %f sec\n", procid, size, (toc-tic));
    }
    else {
      MPI_Recv(buf, size, MPI_CHAR, procid - 1, tag++, comm, MPI_STATUS_IGNORE);
    }
  }
  //codesnippet end
  MPI_Finalize();
  return 0;
}
