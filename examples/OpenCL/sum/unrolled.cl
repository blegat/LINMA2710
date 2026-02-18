// Inspired from https://github.com/JuliaGPU/OpenCL.jl/blob/master/src/mapreduce.jl

//codesnippet
__kernel void simt_sum(volatile __local float* shared)
{
  int items = get_local_size(0);
  int item = get_local_id(0);
  if (items >= 64)
    shared[item] += shared[item + 32];
  if (items >= 32)
    shared[item] += shared[item + 16];
  if (items >= 16)
    shared[item] += shared[item + 8];
  if (items >= 8)
    shared[item] += shared[item + 4];
  if (items >= 4)
    shared[item] += shared[item + 2];
  if (items >= 2)
    shared[item] += shared[item + 1];
}

__kernel void local_sum(__local float* shared) {
  int items = get_local_size(0);
  int item = get_local_id(0);
  if (items >= 512) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (item < 256)
      shared[item] += shared[item + 256];
  }
  if (items >= 256) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (item < 128)
      shared[item] += shared[item + 128];
  }
  if (items >= 128) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (item < 64)
      shared[item] += shared[item + 64];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (item < 32)
    simt_sum(shared);
}
//codesnippet

__kernel void sum(__global float* glob, __local float* shared, __global float *result) {
  int global_size = get_global_size(0);
  int local_size = get_local_size(0);
  int item = get_local_id(0);
  shared[item] = 0;
  for (int i = 0; i < global_size; i += local_size) {
    shared[item] += glob[i + item];
  }
  local_sum(shared);
  if (item == 0)
    *result = shared[item];
}
