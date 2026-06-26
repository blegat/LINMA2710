#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void sum(float *vec, int length, int procid, int verbose);

int main(int argc, char *argv[]) {
  if (argc < 2)
      return 1;
  char *end;
  long total_length = strtol(argv[1], &end, 10);

  int nprocs, procid;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  int stride = total_length / nprocs;
  int first = stride * procid;
  int last = stride * (procid + 1);
  if (procid + 1 == nprocs)
    last = total_length;
  int verbose = 1;
  if (verbose >= 1)
    fprintf(stderr, "proc id : %d / %d %d:%d\n", procid, nprocs, stride * procid, last - 1);

  //codesnippet mpi_sum
  int length = last - first;
  float *vec = new float[length];
  for (int i = 0; i < length; i++)
    vec[i] = first + i;
  sum(vec, length, procid, verbose);
  return 0;
}

void sum(float *vec, int length, int procid, int verbose) {
  float local_sum = 0;
  for (int i = 0; i < length; i++)
    local_sum += vec[i];
  float total = 0;
  MPI_Reduce(&local_sum, &total, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (verbose >= 1)
    fprintf(stderr, "proc id : %d : [local = %f] : [total = %f]\n", procid, local_sum, total);
  //codesnippet end
  MPI_Finalize();
}

