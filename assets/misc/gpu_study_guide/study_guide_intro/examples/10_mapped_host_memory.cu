#include "cuda_check.cuh"
#include <cmath>
#include <cstddef>
#include <cstdio>

__global__ void scaleMapped(float* values, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] *= 3.0f;
}

int main() {
    // Set this before another Runtime call creates the device context.
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    if (!properties.canMapHostMemory) {
        std::printf("Mapped host memory: SKIP (device does not support it)\n");
        return 0;
    }

    constexpr int N = 17;
    constexpr int threads = 8;
    constexpr std::size_t bytes = N * sizeof(float);
    const int blocks = (N + threads - 1) / threads;

    float* hostPointer = nullptr;
    float* deviceView = nullptr;
    CUDA_CHECK(cudaHostAlloc(&hostPointer, bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&deviceView, hostPointer, 0));

    for (int i = 0; i < N; ++i) hostPointer[i] = static_cast<float>(i);

    // The kernel accesses CPU-owned pinned memory through its device mapping.
    scaleMapped<<<blocks, threads>>>(deviceView, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bool ok = true;
    for (int i = 0; i < N; ++i)
        ok &= std::fabs(hostPointer[i] - 3.0f * i) < 1e-5f;

    std::printf("Mapped host-memory scale: %s\n", ok ? "PASS" : "FAIL");
    CUDA_CHECK(cudaFreeHost(hostPointer));
    return ok ? 0 : 1;
}
