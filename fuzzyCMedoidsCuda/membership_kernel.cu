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
	int start = threadIdx.x * STEP_SIZE_MEMB;
	int end = 0;

	__shared__ float ourMedoids[NUM_CLUSTERS * NUM_DIMENSIONS];

	if (threadIdx.x == 0) {
		for (int i = 0; i < NUM_CLUSTERS * NUM_DIMENSIONS; i++) {
			ourMedoids[i] = medoids[i];
		}
	}

	__syncthreads();

	if (threadIdx.x == (NUM_THREADS - 1)) {
		end = NUM_DATA_POINTS;
	}
	else {
		end = start + STEP_SIZE_MEMB;
	}

	if (end > NUM_DATA_POINTS) {
		end = NUM_DATA_POINTS;
	}

	if (start < NUM_DATA_POINTS) {
		for (int x = start; x < end; x++) {
			calculateMembership(data, ourMedoids, memb, 2, x);
		}
	}
}

__device__ void calculateMembership(float* d, float* md, float* mb, int m, int index) {
	float numerator;
	float denominator = 0;
	float exp = 1;
	float base;

	if ((m - 1) != 0) {
		exp = 1 / (m - 1);
	}

	//for (int j = 0; j < NUM_CLUSTERS; j++) {
		base = calculateDist(index, blockIdx.x, d, md);
		numerator = pow(base, exp);

		for (int x = 0; x < NUM_CLUSTERS; x++) {
			base = calculateDist(index, x, d, md);
			denominator += pow(base, exp);
		}

		mb[blockIdx.x + index * NUM_CLUSTERS] = numerator / denominator;
		denominator = 0;
	//}
}

#endif /* MEMBERSHIP_KERNEL_CU_ */
