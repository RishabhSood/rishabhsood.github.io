#include "cuda_check.cuh"
#include "cuda_tile.h"
#include <cmath>
#include <cstdio>

__tile_global__ void tileVectorAdd(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ c, int n) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);
    c = ct::assume_aligned(c, 16_ic);
    auto av = ct::partition_view{ct::tensor_span{a, ct::extents{n}}, ct::shape{8_ic}};
    auto bv = ct::partition_view{ct::tensor_span{b, ct::extents{n}}, ct::shape{8_ic}};
    auto cv = ct::partition_view{ct::tensor_span{c, ct::extents{n}}, ct::shape{8_ic}};
    int block = ct::bid().x;
    cv.store_masked(av.load_masked(block) + bv.load_masked(block), block);
}

int main() {
    constexpr int N = 17, tileSize = 8;
    const int blocks = (N + tileSize - 1) / tileSize;
    float *a = nullptr, *b = nullptr, *c = nullptr;
    CUDA_CHECK(cudaMallocManaged(&a, N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&b, N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&c, N * sizeof(float)));
    for (int i = 0; i < N; ++i) { a[i] = i; b[i] = 10.0f * i; }

    // Tile kernels specify blocks but use 1 as the second launch argument.
    tileVectorAdd<<<blocks, 1>>>(a, b, c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    bool ok = true;
    for (int i = 0; i < N; ++i) ok &= std::fabs(c[i] - 11.0f * i) < 1e-5f;
    std::printf("CUDA Tile vector add: %s\n", ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    return ok ? 0 : 1;
}

