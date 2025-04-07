#include <cstddef>
#include <omp.h>
#include <stdio.h>

void sum_to(float *vec, int length, float *local_results, int num_threads, int verbose) {
  omp_set_dynamic(0); // Force the value `num_threads`
  omp_set_num_threads(num_threads);
  #pragma omp parallel
  {
    int thread_num = omp_get_thread_num();
	int stride = length / num_threads;
    int last = stride * (thread_num + 1);
    if (thread_num + 1 == num_threads)
      last = length;
	if (verbose >= 1)
      fprintf(stderr, "thread id : %d / %d %d:%d\n", thread_num, omp_get_num_threads(), stride * thread_num, last - 1);
	float no_false_sharing = 0;
    #pragma omp simd
    for (int i = stride * thread_num; i < last; i++)
      no_false_sharing += vec[i];
	local_results[thread_num] = no_false_sharing;
  }
}

extern "C" {
float sum(float *vec, int length, int num_threads, int factor, int verbose) {
  float* buffers[2] = {new float[num_threads], new float[num_threads / factor]};
  sum_to(vec, length, buffers[0], num_threads, verbose);
  int prev = num_threads, cur;
  int buffer_idx = 0;
  for (cur = num_threads / factor; cur > 0; cur /= factor) {
	sum_to(buffers[buffer_idx % 2], prev, buffers[(buffer_idx + 1) % 2], cur, verbose);
	prev = cur;
	buffer_idx += 1;
  }
  if (prev == 1)
	return buffers[buffer_idx % 2][0];
  sum_to(buffers[buffer_idx % 2], prev, buffers[(buffer_idx + 1) % 2], 1, verbose);
  return buffers[(buffer_idx + 1) % 2][0];
}
}
