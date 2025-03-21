// Inspired from https://github.com/JuliaGPU/OpenCL.jl/blob/master/src/mapreduce.jl

//codesnippet
__kernel void local_sum(__global float* glob, __local float* shared, __global float* result)
{
  int items = get_local_size(0);
  int item = get_local_id(0);
  shared[item] = glob[item];
  int index = 0, d = 1;
  float other_val = 0;
  d = 1;
  while (d < items) {
    barrier(CLK_LOCAL_MEM_FENCE);
    index = 2 * d * item;
    if (index < items) {
      other_val = 0;
      if (index + d < items)
        other_val = shared[index+d];
      shared[index] = shared[index] + other_val;
    }
    d *= 2;
  }
  if (item == 0)
    *result = shared[item];
//codesnippet
}
