#include "cuda_check.cuh"
#include <cstdio>

constexpr int BX = 4;
constexpr int BY = 4;
constexpr int R = 1;

__global__ void fourNeighborAverage(const float* input, float* output,
                                    int width, int height) {
    __shared__ float tile[BY + 2 * R][BX + 2 * R];
    int tx = threadIdx.x, ty = threadIdx.y;
    int linearThread = ty * BX + tx;
    constexpr int tileElements = (BX + 2 * R) * (BY + 2 * R);

    // All threads cooperate to load the block plus its one-element halo.
    for (int p = linearThread; p < tileElements; p += BX * BY) {
        int sx = p % (BX + 2 * R);
        int sy = p / (BX + 2 * R);
        int gx = blockIdx.x * BX + sx - R;
        int gy = blockIdx.y * BY + sy - R;
        tile[sy][sx] = (gx >= 0 && gx < width && gy >= 0 && gy < height)
                         ? input[gy * width + gx]
                         : 0.0f;
    }
    __syncthreads();

    int x = blockIdx.x * BX + tx;
    int y = blockIdx.y * BY + ty;
    if (x < width && y < height) {
        output[y * width + x] = 0.25f * (
            tile[ty + R][tx] + tile[ty + R][tx + 2 * R] +
            tile[ty][tx + R] + tile[ty + 2 * R][tx + R]);
    }
}

int main() {
    constexpr int width = 6, height = 5;
    float *input = nullptr, *output = nullptr;
    CUDA_CHECK(cudaMallocManaged(&input, width * height * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&output, width * height * sizeof(float)));
    for (int i = 0; i < width * height; ++i) input[i] = static_cast<float>(i);

    dim3 block(BX, BY);
    dim3 grid((width + BX - 1) / BX, (height + BY - 1) / BY);
    fourNeighborAverage<<<grid, block>>>(input, output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) std::printf("%6.1f", output[y * width + x]);
        std::puts("");
    }
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
}

