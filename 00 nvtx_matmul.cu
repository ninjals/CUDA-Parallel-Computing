#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 16

__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < N && col < N) {
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrixMul(float* A, float* B, float* C, int N) {
    nvtxRangePush("Matrix Multiplication");

    float* d_A, * d_B, * d_C;
    int size = N * N * sizeof(float);

    nvtxRangePush("Memory Allocation");
    cudaError_t err;
    err = cudaMalloc(&d_A, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error for d_A: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_B, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error for d_B: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_C, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error for d_C: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    nvtxRangePop();

    nvtxRangePush("Memory Copy H2D");
    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error from A to d_A: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error from B to d_B: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    nvtxRangePop();

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    nvtxRangePush("Kernel Execution");
    matrixMulKernel << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device synchronization error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    nvtxRangePop();

    nvtxRangePush("Memory Copy D2H");
    err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error from d_C to C: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    nvtxRangePop();

    nvtxRangePush("Memory Deallocation");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    nvtxRangePop();

    nvtxRangePop();  // End of Matrix Multiplication
}

int main() {
    const int N = 1024;
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    // Initialize matrices A and B here...
    // Example initialization
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    matrixMul(A, B, C, N);

    // Use result in C...

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}


