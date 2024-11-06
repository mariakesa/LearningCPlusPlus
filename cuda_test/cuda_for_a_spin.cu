#include <iostream>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    printf("Hello, World from the GPU!\n");
}

int main() {
    // Launch the kernel with a single thread
    helloFromGPU<<<1, 1>>>();

    // Check for any errors during launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // Wait for the GPU to finish before returning to the CPU
    cudaDeviceSynchronize();

    // Print message from the CPU
    std::cout << "Hello, World from the CPU!" << std::endl;

    return 0;
}
