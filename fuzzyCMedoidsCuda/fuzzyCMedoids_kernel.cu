#ifndef _FUZZYCMEDOIDS_KERNEL_H_
#define _FUZZYCMEDOIDS_KERNEL_H_

#include <stdio.h>
#include "cmedoids.h"

__device__ void calculateCost(float* d, float* m, float* costs, int index);
__device__ float calculateDist(int i, int x, float* d, float* m);

__global__ void fuzzyCMedoids(float* data, float* medoids, float* cost) {
	int i = (blockIdx.x * NUM_THREADS) + threadIdx.x;
	int start = i * STEP_SIZE;
	int end = 0;

	__shared__ float costs[NUM_CLUSTERS];
	__shared__ float ourMedoids[NUM_CLUSTERS * NUM_DIMENSIONS];

	if (threadIdx.x == 0) {
		for (int i = 0; i < NUM_CLUSTERS * NUM_DIMENSIONS; i++) {
			ourMedoids[i] = medoids[i];
		}
	}

	if (blockIdx.x == (NUM_BLOCKS - 1) && threadIdx.x == (NUM_THREADS - 1)) {
		end = NUM_DATA_POINTS;
	}
	else {
		end = start + STEP_SIZE;
	}

	__syncthreads();

	if (start < NUM_DATA_POINTS && end <= NUM_DATA_POINTS) {
		for (int x = start; x < end; x++) {
			calculateCost(data, ourMedoids, costs, x);
		}
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		for (int x = 0; x < NUM_CLUSTERS; x++) {
			*cost += costs[x];
		}
	}
}

__device__ void calculateCost(float* d, float* m, float* costs, int index) {
	float dist;
	float leastDist = -1;
	int cluster = 0;

	for (int j = 0; j < NUM_CLUSTERS; j++) {
		dist = calculateDist(index, j, d, m);

		if (leastDist == -1 || dist < leastDist) {
			leastDist = dist;
			cluster = j;
		}
	}

	costs[cluster] += leastDist;
}

__device__ float calculateDist(int i, int x, float* d, float* m) {
	float sum = 0;

	for (int j = 0; j < NUM_DIMENSIONS; j++) {
		sum += pow(d[i + j * NUM_DIMENSIONS] - m[j + x * NUM_DIMENSIONS], 2);
	}

	return sqrt(sum);
}

#endif
