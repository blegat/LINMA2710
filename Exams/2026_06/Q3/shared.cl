#define b 32

__kernel void tiled_qkt(__global const float *Q,
                        __global const float *K,
                        __global float *R,
                        const int n) {
    __local float BQ[b][b];
    __local float BK[b][b];

    const int li = get_local_id(1);
    const int lj = get_local_id(0);
    const int i  = get_global_id(1);
    const int j  = get_global_id(0);
    const int bj = get_group_id(0) * b;

    float acc = 0.0f;

    for (int bk = 0; bk < n; bk += b) {
        BQ[li][lj] = Q[i * n + (bk + lj)];
        BK[li][lj] = K[(bj + li) * n + (bk + lj)];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < b; k++) {
            acc += BQ[li][k] * BK[lj][k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    R[i * n + j] = acc;
}
