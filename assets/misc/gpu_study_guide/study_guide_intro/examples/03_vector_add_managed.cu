#include "cuda_check.cuh"
#include <cmath>
#include <cstdio>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    constexpr int N = 17;
    constexpr int threads = 8;
    const int blocks = (N + threads - 1) / threads;

    float *a = nullptr, *b = nullptr, *c = nullptr;
    CUDA_CHECK(cudaMallocManaged(&a, N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&b, N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&c, N * sizeof(float)));

    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 2.0f * static_cast<float>(i);
    }

    vectorAdd<<<blocks, threads>>>(a, b, c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // CPU must wait before reading c

    bool ok = true;
    for (int i = 0; i < N; ++i)
        ok &= std::fabs(c[i] - 3.0f * i) < 1e-5f;
    std::printf("Unified-memory vector add: %s\n", ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    return ok ? 0 : 1;
}

