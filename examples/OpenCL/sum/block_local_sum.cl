// Inspired from https://github.com/JuliaGPU/OpenCL.jl/blob/master/src/mapreduce.jl

//codesnippet
__kernel void local_sum(__local float* shared, int factor)
{
  int items = get_local_size(0);
  int item = get_local_id(0);
  int index = 0, stride = 1, i;
  while (stride < items) {
    barrier(CLK_LOCAL_MEM_FENCE);
    index = factor * stride * item;
    for (i = 1; i < factor && index + i * stride < items; i++)
      shared[index] += shared[index + i * stride];
    stride *= factor;
  }
}

__kernel void sum(__global float* glob, __local float* shared, __global float* result, int factor) {
  int global_size = get_global_size(0);
  int local_size = get_local_size(0);
  int item = get_local_id(0);
  shared[item] = 0;
  for (int i = 0; i < global_size; i += local_size) {
    shared[item] += glob[i + item];
  }
  local_sum(shared, factor);
  if (item == 0)
    *result = shared[item];
}
//codesnippet
