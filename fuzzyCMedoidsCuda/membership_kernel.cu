/*
 * membership_kernel.cu
 *
 *  Created on: Apr 11, 2009
 *      Author: doug
 */

#ifndef MEMBERSHIP_KERNEL_CU_
#define MEMBERSHIP_KERNEL_CU_

void calculateMembership(float* d, float* md, float* mb,  int* dims, int nc, int m, int index);

__global__ void calcMembership(float* data, float* medoids, float* memb, int* dims, int nc, int nb, int nt, int ss) {
	int start;
	int end;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (ss == 0) {
		start = i + j * dims[1];
	}
	else {
		start = (i + j * dims[1]) * ss;

		if (blockIdx.x == (nb - 1) && threadIdx.x == (nt - 1)) {
			end = dims[1];
		}
		else {
			end = start + ss;
		}

		//printf("%d, %d\n", start, end);
		if (start < dims[1] && end <= dims[1]) {
			for (int x = start; x < end; x++) {
				calculateMembership(data, medoids, memb, dims, nc, 2, x);
				//printf("%d\n", x);
			}
		}
	}

	__syncthreads();
}

__device__ void calculateMembership(float* d, float* md, float* mb,  int* dims, int nc, int m, int index) {
	float numerator;
	float denominator = 0;
	float exp = 1 / (m - 1);
	float base;

	//for (int i = 0; i < dims[1]; i++) {
		for (int j = 0; j < nc; j++) {
			base = calculateDist(index, j, d, md, dims, nc);
			numerator = pow(base, exp);

			for (int x = 0; x < nc; x++) {
				base = calculateDist(index, x, d, md, dims, nc);
				denominator += pow(base, exp);
			}

			mb[j + index * nc] = numerator / denominator;
			//printf("%d\n", index);
			denominator = 0;
		}
	//}
}

#endif /* MEMBERSHIP_KERNEL_CU_ */
