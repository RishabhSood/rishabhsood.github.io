#include "cuda_check.cuh"
#include <cstdio>

int main() {
    int device = 0, supported = 0;
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, device));
    std::printf("%s (compute capability %d.%d): cluster launch %s\n",
                p.name, p.major, p.minor, supported ? "SUPPORTED" : "NOT SUPPORTED");
    if (!supported)
        std::puts("Expected on pre-Hopper GPUs: clusters require compute capability 9.0+.");
}

