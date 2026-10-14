#include "cuda_check.cuh"
#include <cstdio>

__global__ void showDivergence() {
    unsigned lane = threadIdx.x % warpSize;
    if (lane < 16) {
        std::printf("path=A lane=%2u active-mask=0x%08x\n", lane, __activemask());
    } else {
        std::printf("path=B lane=%2u active-mask=0x%08x\n", lane, __activemask());
    }
}

int main() {
    showDivergence<<<1, 32>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::puts("One warp took two paths; inactive lanes were masked on each path.");
}

