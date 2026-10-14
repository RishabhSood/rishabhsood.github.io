#include "cuda_check.cuh"
#include <cstdio>

__global__ void helloHierarchy() {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    std::printf("block=%u thread=%u global=%d warp-in-block=%u lane=%u\n",
                blockIdx.x, threadIdx.x, globalId,
                threadIdx.x / warpSize, threadIdx.x % warpSize);
}

int main() {
    helloHierarchy<<<2, 40>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::puts("The line order is deliberately unspecified.");
}

