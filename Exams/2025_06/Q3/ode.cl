__kernel void _diffuse(__local float* a, float F, int num_iter) {
    int i = get_local_id(0); // same as global id
    int prev, next, iter;
    if (i == 0)
        prev = get_local_size(0) - 1; // same as global size
    else
        prev = i - 1;
    if (i == get_local_size(0) - 1)
        next = 0;
    else
        next = i + 1;
    for (iter = 0; iter < num_iter; iter++)
        a[i] = a[i] + F * (a[prev] + a[next] - 2 * a[i]);
}

__kernel void diffuse(__global float* glob, __local float* shared, float F, int num_iter) {
    int i = get_global_id(0);
    shared[i] = glob[i];
    _diffuse(shared, F, num_iter);
    glob[i] = shared[i];
}
