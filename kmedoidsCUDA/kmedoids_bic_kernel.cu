/*
 * kmedoids_bic_kernel.cu
 *
 *  Created on: Nov 16, 2008
 *      Author: doug
 */

#ifndef _KMEDOIDS_BIC_KERNEL_H_
#define _KMEDOIDS_BIC_KERNEL_H_

#include <math.h>
#include <math_functions.h>

__device__ float calcBIC(float* d, float* m, int* dims, int k, int i);
__device__ float calcDist(int i, int x, float* d, float* m, int* d_dims);

__global__ void calculateBIC(float* data, float* medoids, float* g_bic, int* dims, int* nc) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = i + j * dims[1];

	if (index < dims[1]) {
		float BIC;
		float highestBIC = -1;
		int bestNumClusters;

		for (int x = 1; x <= *nc; x++) {
			BIC = calcBIC(data, medoids, dims, x, index);
			/*printf("%f\n", BIC);
			if (highestBIC == -1 || BIC > highestBIC) {
				highestBIC = BIC;
				bestNumClusters = x;
			}*/

			__syncthreads();
			g_bic[x - 1] += BIC;
		}

		/*__syncthreads();
		g_bic[bestNumClusters] += highestBIC;*/
	}
}

__device__ float calcDist(int i, int x, float* d, float* m, int* d_dims) {
	float sum = 0;

	for (int j = 0; j < d_dims[0]; j++) {
		sum += pow(d[i + j * d_dims[1]] - m[j + x * d_dims[0]], 2);
	}

	return sqrt(sum);
}

__device__ float calcBIC(float* d, float* m, int* dims, int k, int i) {
	/*if (k == 0) {
		return 0;
	}
	else {*/
		float RSS = 0;
		float dist = 0;
		float currentDist = -1;
		int medoidIndex = -1;

		// calculate residual sum of squares
		for (int x = 0; x < k; x++) {
			dist = calcDist(i, x, d, m, dims);

			if (currentDist == -1 || dist < currentDist) {
				currentDist = dist;
				medoidIndex = x;
			}
		}

		for (int j = 0; j < dims[0]; j++) {
			RSS += pow(d[i + j * dims[0]] - m[j + medoidIndex * dims[0]], 2);
		}

		//return (dims[1] * log10f(RSS / dims[1])) + (k * log10f(dims[1]));
		return RSS;
	//}
}

#endif
