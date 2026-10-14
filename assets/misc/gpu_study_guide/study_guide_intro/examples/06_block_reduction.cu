#include "cuda_check.cuh"
#include <cstdio>

__global__ void sumBlocks(const int* input, int* total, int n) {
    extern __shared__ int partial[];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    partial[t] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (t < stride) partial[t] += partial[t + stride];
        __syncthreads();
    }
    if (t == 0) atomicAdd(total, partial[0]);
}

int main() {
    constexpr int N = 20;
    constexpr int threads = 8;
    const int blocks = (N + threads - 1) / threads;
    int *input = nullptr, *total = nullptr;
    CUDA_CHECK(cudaMallocManaged(&input, N * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&total, sizeof(int)));
    for (int i = 0; i < N; ++i) input[i] = i + 1;
    *total = 0;

    sumBlocks<<<blocks, threads, threads * sizeof(int)>>>(input, total, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("sum(1..%d) = %d (expected %d)\n", N, *total, N * (N + 1) / 2);

    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(total));
}

