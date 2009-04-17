#ifndef _FUZZYCMEDOIDS_KERNEL_H_
#define _FUZZYCMEDOIDS_KERNEL_H_

#include <stdio.h>
#include "cmedoids.h"

__device__ void calculateCost(float* d, float* m, float* costs, int index);
__device__ float calculateDist(int i, int x, float* d, float* m);

__global__ void fuzzyCMedoids(float* data, float* medoids, float* cost) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int start;
	int end;

	start = (i + j * NUM_DATA_POINTS) * STEP_SIZE;

	if (blockIdx.x == (NUM_BLOCKS - 1) && threadIdx.x == (NUM_THREADS - 1)) {
		end = NUM_DATA_POINTS;
	}
	else {
		end = start + STEP_SIZE;
	}

	int mSize = NUM_CLUSTERS * NUM_DIMENSIONS;

	__shared__ float costs[NUM_CLUSTERS];
	__shared__ float ourMedoids[NUM_CLUSTERS * NUM_DIMENSIONS];

	int count = 0;
	int dataSize = 0;

	if (blockIdx.x == (NUM_BLOCKS - 1) && threadIdx.x == (NUM_THREADS- 1)) {
		__shared__ float ourData[BLOCK_DATA_SIZE_LAST];

		for (int x = start; x < end && count < BLOCK_DATA_SIZE_LAST; x++) {
			for (int y = 0; y < NUM_DIMENSIONS; y++) {
				ourData[count + y] = data[x + y];
				count++;
			}
		}

		dataSize = BLOCK_DATA_SIZE_LAST;
		end = NUM_DATA_POINTS;
	}
	else if (threadIdx.x == 0) {
		__shared__ float ourData[BLOCK_DATA_SIZE];

		for (int x = start; x < end && count < BLOCK_DATA_SIZE; x++) {
			for (int y = 0; y < NUM_DIMENSIONS; y++) {
				ourData[count + y * BLOCK_DATA_SIZE] = data[x + y * NUM_DATA_POINTS];

			}
			count++;
		}

		dataSize = BLOCK_DATA_SIZE_LAST;
		end = start + STEP_SIZE;
	}

	start = (i + j * BLOCK_DATA_SIZE) * STEP_SIZE;

	if (threadIdx.x == 0) {
		for (int a = 0; a < mSize; a++) {
			ourMedoids[a] = medoids[a];
		}

		for (int z = 0; z < NUM_CLUSTERS; z++) {
			costs[z] = 0;
		}
	}

	__syncthreads();

	if (start < NUM_DATA_POINTS && end <= NUM_DATA_POINTS) {
		//for (int x = start; x < end; x++) {
		for (int x = 0; x < dataSize; x++) {
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
