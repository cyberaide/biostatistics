#ifndef _FUZZYKMEDOIDS_KERNEL_H_
#define _FUZZYKMEDOIDS_KERNEL_H_

#include <stdio.h>

float calculateCost(float* d, float* m, int* c, int nc, int dims[], int det);

__global__ void fuzzyCMedoids(float* data, float* medoids, int* result, int* dims, int numBlocks, int numThreads, int stepSize) {
	int start;
	int end;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (stepSize == 0) {
		start = i + j * dims[1];
	}
	else {
		start = (i + j * dims[1]) * stepSize;

		if (blockIdx.x == (numBlocks - 1) && threadIdx.x == (numThreads - 1)) {
			end = dims[1];
		}
		else {
			end = start + stepSize;
		}

		for (int i = start; i < end; i++) {
			if (start < dims[1] && end <= dims[1]) {
				printf("%d\n", i);
			}
		}
	}
}

#endif
