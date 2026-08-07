#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Request rqst;
  int nprocs, procid;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(comm,&nprocs);
  MPI_Comm_rank(comm,&procid);
  //codesnippet mpi_bench2
  for(int size = 1; size <= (1<<20); size <<= 1){
    char* buf = malloc(size);
    if (procid == 0) {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Send(buf, size, MPI_CHAR, procid + 1, 0, comm);
    }
    else {
      MPI_Irecv(buf, size, MPI_CHAR, procid - 1, 0, MPI_COMM_WORLD, &rqst);
      MPI_Barrier(MPI_COMM_WORLD);
      double tic = MPI_Wtime();
      MPI_Wait(&rqst, MPI_STATUS_IGNORE);
      double toc = MPI_Wtime();
      printf("[%d] I have received %d B in %f sec\n", procid, size, (toc-tic));
    }
  }
  //codesnippet end
  MPI_Finalize();
  return 0;
}
