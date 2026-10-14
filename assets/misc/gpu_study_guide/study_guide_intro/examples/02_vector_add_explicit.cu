#include "cuda_check.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    constexpr int N = 17;                 // deliberately not divisible by 8
    constexpr int threads = 8;
    const int blocks = (N + threads - 1) / threads;

    std::vector<float> a(N), b(N), c(N);
    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 100.0f - static_cast<float>(i);
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    vectorAdd<<<blocks, threads>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (float value : c) ok &= std::fabs(value - 100.0f) < 1e-5f;
    std::printf("N=%d, blocks=%d, threads/block=%d, result=%s\n",
                N, blocks, threads, ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return ok ? 0 : 1;
}

