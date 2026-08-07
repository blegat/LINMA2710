//codesnippet
__kernel void vadd(
    __global const float *a,
    __global const float *b,
    __global float *c,
    int verbose) {
  int i = get_global_id(0);
  c[i] = a[i] + b[i];
//codesnippet
  if (verbose != 0)
    if ((i % (get_global_size(0) / verbose)) == 0)
      printf("%4d/%4d | %2d/%2d | %4d/%4d\n", i, get_global_size(0), get_group_id(0), get_num_groups(0), get_local_id(0), get_local_size(0));
}
