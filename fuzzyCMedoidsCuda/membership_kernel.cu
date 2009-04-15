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
	int start;
	int end;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	/*if (ss == 0) {
		start = i + j * dims[1];
	}
	else {*/
		start = (i + j * NUM_DATA_POINTS) * STEP_SIZE;

		if (blockIdx.x == (NUM_BLOCKS - 1) && threadIdx.x == (NUM_THREADS - 1)) {
			end = NUM_DATA_POINTS;
		}
		else {
			end = start + STEP_SIZE;
		}

		if (start < NUM_DATA_POINTS && end <= NUM_DATA_POINTS) {
			for (int x = start; x < end; x++) {
				calculateMembership(data, medoids, memb, 2, x);
			}
		}
	//}

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
