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
	//int i = blockIdx.x * blockDim.x + threadIdx.x;
	//int j = blockIdx.y * blockDim.y + threadIdx.y;
	//int start = (i + j * NUM_DATA_POINTS) * STEP_SIZE_MEMB;
	//int start = threadIdx.x * NUM_THREADS;
	int start = threadIdx.x * STEP_SIZE_MEMB;
	int end = 0;

	//blockIdx.x == (NUM_BLOCKS - 1) &&
	if (threadIdx.x == (NUM_THREADS - 1)) {
		end = NUM_DATA_POINTS;
	}
	else {
		//end = start + NUM_THREADS;
		end = start + STEP_SIZE_MEMB;
	}

	if (end > NUM_DATA_POINTS) {
		end = NUM_DATA_POINTS;
	}

	if (start < NUM_DATA_POINTS) {
		/*for (int x = start; x < end; x++) {
			calculateMembership(data, medoids, memb, 2, x);
		}*/
	}

	__syncthreads();
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
