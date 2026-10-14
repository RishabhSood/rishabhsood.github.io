#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

void requireCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n",
                     operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void addOne(float* values, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] += 1.0f;
}

int main() {
    constexpr int N = 4;
    constexpr int THREADS = 4;
    float hValues[N] = {10.0f, 20.0f, 30.0f, 40.0f};
    float* dValues = nullptr;
    const std::size_t bytes = N * sizeof(float);

    requireCuda(cudaMalloc(&dValues, bytes), "cudaMalloc");
    requireCuda(cudaMemcpy(dValues, hValues, bytes,
                           cudaMemcpyHostToDevice), "copy to GPU");

    addOne<<<1, THREADS>>>(dValues, N);
    requireCuda(cudaGetLastError(), "addOne launch");
    requireCuda(cudaDeviceSynchronize(), "addOne execution");

    requireCuda(cudaMemcpy(hValues, dValues, bytes,
                           cudaMemcpyDeviceToHost), "copy to CPU");
    requireCuda(cudaFree(dValues), "cudaFree");
    dValues = nullptr;  // cudaFree releases the allocation but does not clear this pointer.

    for (int i = 0; i < N; ++i) std::printf("%.0f ", hValues[i]);
    std::printf("\n");

    const bool ok = hValues[0] == 11.0f && hValues[1] == 21.0f &&
                    hValues[2] == 31.0f && hValues[3] == 41.0f;
    std::printf("First explicit-memory walkthrough: %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
