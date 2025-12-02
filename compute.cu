#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"
#include "compute.h" 

// External globals
extern vector3 *hVel, *hPos;
extern double *mass;

// Device globals
vector3 *d_vel, *d_pos, *d_accels;
double *d_mass;

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Kernel 1: Pairwise accelerations
__global__ void pairwise_kernel(vector3 *accels, vector3 *pos, double *mass) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i < NUMENTITIES && j < NUMENTITIES) {
        if (i == j) {
            FILL_VECTOR(accels[i * NUMENTITIES + j], 0, 0, 0);
        } else {
            vector3 distance;
            distance[0] = pos[i][0] - pos[j][0];
            distance[1] = pos[i][1] - pos[j][1];
            distance[2] = pos[i][2] - pos[j][2];

            double magnitude_sq = distance[0]*distance[0] + distance[1]*distance[1] + distance[2]*distance[2];
            double magnitude = sqrt(magnitude_sq);
            double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;

            FILL_VECTOR(accels[i * NUMENTITIES + j], 
                        accelmag * distance[0] / magnitude, 
                        accelmag * distance[1] / magnitude, 
                        accelmag * distance[2] / magnitude);
        }
    }
}

// Kernel 2: Update Physics
__global__ void update_kernel(vector3 *accels, vector3 *pos, vector3 *vel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NUMENTITIES) {
        vector3 accel_sum = {0, 0, 0};
        for (int j = 0; j < NUMENTITIES; j++) {
            accel_sum[0] += accels[i * NUMENTITIES + j][0];
            accel_sum[1] += accels[i * NUMENTITIES + j][1];
            accel_sum[2] += accels[i * NUMENTITIES + j][2];
        }

        vel[i][0] += accel_sum[0] * INTERVAL;
        vel[i][1] += accel_sum[1] * INTERVAL;
        vel[i][2] += accel_sum[2] * INTERVAL;

        pos[i][0] += vel[i][0] * INTERVAL;
        pos[i][1] += vel[i][1] * INTERVAL;
        pos[i][2] += vel[i][2] * INTERVAL;
    }
}

void initDeviceMemory() {
    cudaCheckError(cudaMalloc((void**)&d_pos, sizeof(vector3) * NUMENTITIES));
    cudaCheckError(cudaMalloc((void**)&d_vel, sizeof(vector3) * NUMENTITIES));
    cudaCheckError(cudaMalloc((void**)&d_mass, sizeof(double) * NUMENTITIES));
    cudaCheckError(cudaMalloc((void**)&d_accels, sizeof(vector3) * NUMENTITIES * NUMENTITIES));
}

void freeDeviceMemory() {
    cudaFree(d_pos); cudaFree(d_vel); cudaFree(d_mass); cudaFree(d_accels);
}

void copyToDevice() {
    cudaCheckError(cudaMemcpy(d_pos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_vel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice));
}

void copyFromDevice() {
    cudaCheckError(cudaMemcpy(hPos, d_pos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(hVel, d_vel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost));
}

void compute() {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NUMENTITIES + 15) / 16, (NUMENTITIES + 15) / 16);
    pairwise_kernel<<<numBlocks, threadsPerBlock>>>(d_accels, d_pos, d_mass);
    cudaCheckError(cudaGetLastError());

    int threadsPerBlock1D = 256;
    int numBlocks1D = (NUMENTITIES + 255) / 256;
    update_kernel<<<numBlocks1D, threadsPerBlock1D>>>(d_accels, d_pos, d_vel);
    cudaCheckError(cudaGetLastError());
    
    cudaDeviceSynchronize();
}