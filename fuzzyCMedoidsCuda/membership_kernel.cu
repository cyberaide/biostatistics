/*
 * membership_kernel.cu
 *
 *  Created on: Apr 11, 2009
 *      Author: doug
 */

#ifndef MEMBERSHIP_KERNEL_CU_
#define MEMBERSHIP_KERNEL_CU_

#include "cmedoids.h"

__device__ void calculateMembership(float* d, float* md, float* mb, int m, int index);

__global__ void calcMembership(float* data, float* medoids, float* memb) {
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
	}

	if (start < NUM_DATA_POINTS && end <= NUM_DATA_POINTS) {
		for (int x = 0; x < dataSize; x++) {
			calculateMembership(data, ourMedoids, memb, 2, x);
		}
	}

	//__syncthreads();
}

__device__ void calculateMembership(float* d, float* md, float* mb, int m, int index) {
	float numerator;
	float denominator = 0;
	float exp = 1 / (m - 1);
	float base;

	for (int j = 0; j < NUM_CLUSTERS; j++) {
		base = calculateDist(index, j, d, md);
		numerator = pow(base, exp);

		for (int x = 0; x < NUM_CLUSTERS; x++) {
			base = calculateDist(index, x, d, md);
			denominator += pow(base, exp);
		}

		mb[j + index * NUM_CLUSTERS] = numerator / denominator;

		denominator = 0;
	}
}

#endif /* MEMBERSHIP_KERNEL_CU_ */
