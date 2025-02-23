#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>  // Include this header for __syncthreads()

#define K 1024
#define TILE_SIZE 16

// Traditional matrix multiplication (dot product approach)
__global__ void matrixMulTraditional(float* A, float* B, float* C, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < m) {
        float value = 0.0f;
        for (int k = 0; k < m; ++k) {
            value += A[row * m + k] * B[k * m + col];
        }
        C[row * m + col] = value;
    }
}

// Matrix multiplication with tiling (using shared memory)
__global__ void matrixMulTiled(float* A, float* B, float* C, int n) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    // Loop over tiles of A and B
    for (int i = 0; i < n / TILE_SIZE; ++i) {
        // Load tiles into shared memory
        tileA[threadIdx.y][threadIdx.x] = A[row * n + (i * TILE_SIZE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * n + col];

        // Synchronize to make sure the tiles are fully loaded
        __syncthreads();

        // Compute the partial sum for this tile
        for (int j = 0; j < TILE_SIZE; ++j) {
            value += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }

        // Synchronize again before loading the next tile
        __syncthreads();
    }

    // Write the result to global memory
    C[row * n + col] = value;
}

int main() {
    float* A, * B, * C, * d_A, * d_B, * d_C;
    size_t size = K * K * sizeof(float);

    // Allocate memory for matrices A, B, and C
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // Initialize matrices A and B with random values
    for (int i = 0; i < K * K; ++i) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Allocate memory on the device
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices A and B to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Timing variables
    cudaEvent_t start, stop;
    float elapsedTime;

    // Traditional approach timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(K / threadsPerBlock.x, K / threadsPerBlock.y);
    matrixMulTraditional << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, K);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Traditional Matrix Multiplication Time: " << elapsedTime << " ms\n";

    // Reset C matrix for the next calculation
    cudaMemset(d_C, 0, size);

    // Tiling approach timing
    cudaEventRecord(start, 0);
    matrixMulTiled << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, K);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Tiled Matrix Multiplication Time: " << elapsedTime << " ms\n";

    // Copy result from device to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}


