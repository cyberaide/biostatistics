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
		for (int i = 0; i < NUM_CLUSTERS; i++) {
			costs[i] = 0;
		}

		for (int i = 0; i < NUM_CLUSTERS * NUM_DIMENSIONS; i++) {
			ourMedoids[i] = medoids[i];
		}
	}

	__syncthreads();

	end = start + STEP_SIZE;

	/*if (end > NUM_DATA_POINTS) {
		end = NUM_DATA_POINTS;
	}*/

	if (start < NUM_DATA_POINTS) {
		for (int x = start; x < end && x < NUM_DATA_POINTS; x++) {
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
	float temp = 0;

	// Euclidean
#if DISTANCE_MEASURE == 0
	for (int j = 0; j < NUM_DIMENSIONS; j++) {
		temp = d[i + j * NUM_DATA_POINTS] - m[j + x * NUM_DIMENSIONS];
		sum += temp * temp;
	}

	sum = sqrt(sum);
#endif

	// Manhattan
#if DISTANCE_MEASURE == 1
	for (int j = 0; j < NUM_DIMENSIONS; j++) {
		temp = d[i + j * NUM_DATA_POINTS] - m[j + x * NUM_DIMENSIONS];
		sum += abs(temp);
	}
#endif

	// Maximum
#if DISTANCE_MEASURE == 2
	for (int j = 0; j < NUM_DIMENSIONS; j++) {
		temp = abs(d[i + j * NUM_DATA_POINTS] - m[j + x * NUM_DIMENSIONS]);

		if (temp > sum) {
			sum = temp;
		}
	}
#endif

	return sum;
}

#endif
