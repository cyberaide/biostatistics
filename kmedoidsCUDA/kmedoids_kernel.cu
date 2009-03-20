#ifndef _KMEDOIDS_KERNEL_H_
#define _KMEDOIDS_KERNEL_H_

#include <stdio.h>
#include <math.h>

__device__ float calculateCost(float* g_data, float* g_medoids, int* g_odata, int i, int* d_dims, int d_numClusters, float* d_memb, int d_det);
__device__ float calculateDist(int i, int x, float* d, float* m, int* d_dims);
__device__ void calculateMembership(float* d, float* m, float* memb, int* d_dims, int n, int i);

__global__ void kmedoids(float* g_data, float* g_medoids, float* g_cost, int* g_odata, int* d_dims, int* d_numClusters, float* d_memb, int* d_det) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = i + j * d_dims[1];

	__shared__ float total;
	total = 0;

	if (index < d_dims[1]) {
		total += calculateCost(g_data, g_medoids, g_odata, index, d_dims, *d_numClusters, d_memb, *d_det);

		__syncthreads();
		*g_cost += total;
	}
}

__device__ float calculateCost(float* g_data, float* g_medoids, int* g_odata, int i, int* d_dims, int d_numClusters, float* d_memb, int d_det) {
	float dist;
	float leastDist = -1;
	int currentCluster = -1;

	for (int x = 0; x < d_numClusters; x++) {
		dist = calculateDist(i, x, g_data, g_medoids, d_dims);

		if (leastDist == -1 || dist < leastDist) {
			leastDist = dist;
			currentCluster = x;
		}
	}

	if (d_det == 1) {
		calculateMembership(g_data, g_medoids, d_memb, d_dims, d_numClusters, i);
	}

	__syncthreads();
	g_odata[i] = currentCluster;

	return leastDist;
}

__device__ float calculateDist(int i, int x, float* d, float* m, int* d_dims) {
	float sum = 0;

	if (d_dims[2] == 0) {
		for (int j = 0; j < d_dims[0]; j++) {
			sum += pow(d[i + j * d_dims[1]] - m[j + x * d_dims[0]], 2);
		}

		sum = sqrt(sum);
	}
	else if (d_dims[2] == 1) {
		for (int j = 0; j < d_dims[0]; j++) {
			sum += abs(d[i + j * d_dims[1]] - m[j + x * d_dims[0]]);
		}
	}

	return sum;
}

__device__ void calculateMembership(float* d, float* m, float* memb, int* d_dims, int n, int i) {
	float dist;
	float totalDist = 0;

	for (int j = 0; j < n; j++) {
		dist = calculateDist(i, j, d, m, d_dims);

		for (int x = 0; x < n; x++) {
			totalDist += calculateDist(i, x, d, m, d_dims);
		}

		memb[i + j * n] = 1 - (dist / totalDist);

		totalDist = 0;
	}
}

#endif
