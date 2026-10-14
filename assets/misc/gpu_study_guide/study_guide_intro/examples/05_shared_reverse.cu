#include "cuda_check.cuh"
#include <cstdio>

__global__ void reverseEight(const int* input, int* output) {
    __shared__ int tile[8];
    int t = threadIdx.x;
    tile[t] = input[t];
    __syncthreads();
    output[t] = tile[7 - t];
}

int main() {
    int *input = nullptr, *output = nullptr;
    CUDA_CHECK(cudaMallocManaged(&input, 8 * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&output, 8 * sizeof(int)));
    for (int i = 0; i < 8; ++i) input[i] = i;

    reverseEight<<<1, 8>>>(input, output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("output:");
    for (int i = 0; i < 8; ++i) std::printf(" %d", output[i]);
    std::puts("");

    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
}

