#include <mpi.h>
#include <stdio.h>

void sum(float *vec, int length, int verbose) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int nprocs, procid;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(comm,&nprocs);
  MPI_Comm_rank(comm,&procid);

  int stride = length / nprocs;
  int last = stride * (procid + 1);
  float local_sum = 0;
  if (procid + 1 == nprocs)
    last = length;
  if (verbose >= 1)
    fprintf(stderr, "proc id : %d / %d %d:%d\n", procid, nprocs, stride * procid, last - 1);
  //codesnippet mpi_sum
  for (int i = stride * procid; i < last; i++)
    local_sum += vec[i];
  float total = 0;
  MPI_Reduce(&local_sum, &total, 1, MPI_FLOAT, MPI_SUM, 0, comm);
  if (verbose >= 1)
    fprintf(stderr, "proc id : %d / %d : [local = %f] : [total = %f]\n", procid, nprocs, local_sum, total);
  //codesnippet end
  MPI_Finalize();
}

int main() {
    float vec[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    sum(vec, 9, 1);
    return 0;
}
