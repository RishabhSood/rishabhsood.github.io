#include "cuda_check.cuh"
#include <cstdio>

int main() {
    int devices = 0, device = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    std::printf("Number of CUDA devices: %d\n", devices);

    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, device));

    int clusterLaunch = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &clusterLaunch, cudaDevAttrClusterLaunch, device));

    std::printf("GPU: %s\n", p.name);
    std::printf("Compute capability: %d.%d\n", p.major, p.minor);
    std::printf("SMs: %d\n", p.multiProcessorCount);
    std::printf("Warp size: %d\n", p.warpSize);
    std::printf("Max threads/block: %d\n", p.maxThreadsPerBlock);
    std::printf("Max resident threads/SM: %d\n", p.maxThreadsPerMultiProcessor);
    std::printf("Registers/SM: %d\n", p.regsPerMultiprocessor);
    std::printf("Shared memory/block: %zu bytes\n", p.sharedMemPerBlock);
    std::printf("L2 cache: %d bytes\n", p.l2CacheSize);
    std::printf("Unified virtual addressing: %s\n", p.unifiedAddressing ? "yes" : "no");
    std::printf("Mapped host memory: %s\n", p.canMapHostMemory ? "yes" : "no");
    std::printf("Managed memory: %s\n", p.managedMemory ? "yes" : "no");
    std::printf("Concurrent managed access: %s\n",
                p.concurrentManagedAccess ? "yes" : "no");
    std::printf("Pageable memory access: %s\n",
                p.pageableMemoryAccess ? "yes" : "no");
    std::printf("Thread-block clusters: %s\n", clusterLaunch ? "yes" : "no");
}
