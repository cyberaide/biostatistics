/*#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>*/
#include <cutil.h>
#include <fuzzyCMedoids_kernel.cu>

void usage();

extern "C" void initRand();
extern "C" float randFloat();
extern "C" float randFloatRange(float min, float max);
extern "C" int randInt(int max);
extern "C" int randIntRange(int min, int max);
extern "C" int getRandIndex(int w, int h);
extern "C" bool contains(float* f, float p[], int n, int dims[]);
extern "C" void setCenters(float* d, float* m, int n, int dims[]);
extern "C" void getPoints(float* d, float p[], int dims[], int i);
extern "C" float* readData(char* f, int* dims);
extern "C" void writeData(float* d, float* m, int* c, int* dims, int nc, float* memb, const char* f);
extern "C" int clusterColor(float i, int nc);

int main( int argc, char** argv) {
	if (argc != 3) {
		usage();
		return EXIT_FAILURE;
	}

	initRand();

	int dimSize = 3 * sizeof(int);
	int* det = (int*)malloc(sizeof(int));
	//int* numClusters = (int*)malloc(sizeof(int));
	//*numClusters = atoi(argv[1]);
	int numClusters = atoi(argv[1]);

	int* dims = (int*)malloc(dimSize);
	float* data = readData(argv[2], dims);

	int medoidSize = sizeof(float) * numClusters * dims[0];
	int dataSize = sizeof(float) * dims[0] * dims[1];
	int resultSize = sizeof(int) * dims[1];
	int membSize = sizeof(float) * numClusters * dims[1];

	// local
	float* membership = (float*)malloc(membSize);
	float* medoids = (float*)malloc(medoidSize);
	float* finalMedoids = (float*)malloc(medoidSize);
	float* oldCost = (float*)malloc(sizeof(float));
	float* newCost = (float*)malloc(sizeof(float));
	int* result = (int*)malloc(resultSize);

	// cuda
	float* d_data;
	float* d_memb;
	float* d_medoids;
	float* d_cost;
	int* d_dims;
	int* d_result;

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_dims, dimSize));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_data, dataSize));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_medoids, medoidSize));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_memb, membSize));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_result, resultSize));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_cost, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpy(d_dims, dims, dimSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_data, data, dataSize, cudaMemcpyHostToDevice));

	//CUDA_SAFE_CALL(cudaMalloc((void**) &d_matrixC, dataSize));
	//CUDA_SAFE_CALL(cudaMemcpy(d_matrixA, matrixA, dataSize, cudaMemcpyHostToDevice));

	int blocks = dims[1];
	int threads = dims[1];
	int dim2 = dims[0] * dims[1];

	if (blocks > 22) {
		blocks == (int)sqrt(dims[1]);

		if (blocks > 22) {
			blocks = 22;
		}
	}

	if (threads > 512) {
		threads = (int)sqrt(dims[1]);

		if (threads > 512) {
			threads = 512;
		}
	}

	int blockDim = blocks * threads;

	*oldCost = 1;
	*newCost = 0;

	int MAXITER = 5;
	int iter = 0;

	while (*oldCost > *newCost && iter < MAXITER) {
		setCenters(data, medoids, numClusters, dims);
		CUDA_SAFE_CALL(cudaMemcpy(d_medoids, medoids, medoidSize, cudaMemcpyHostToDevice));

		memcpy(finalMedoids, medoids, medoidSize);

		if (dim2 > blockDim) {
			fuzzyCMedoids<<<blocks, threads>>>(d_data, d_medoids, d_result, d_dims, d_cost, numClusters, blocks, threads, dims[1] / blockDim);
		}
		else {
			fuzzyCMedoids<<<blocks, threads>>>(d_data, d_medoids, d_result, d_dims, d_cost, numClusters, blocks, threads, 0);
		}

		CUDA_SAFE_CALL(cudaMemcpy(oldCost, d_cost, sizeof(float), cudaMemcpyDeviceToHost));

		setCenters(data, medoids, numClusters, dims);
		CUDA_SAFE_CALL(cudaMemcpy(d_medoids, medoids, medoidSize, cudaMemcpyHostToDevice));

		if (dim2 > blockDim) {
			fuzzyCMedoids<<<blocks, threads>>>(d_data, d_medoids, d_result, d_dims, d_cost, numClusters, blocks, threads, dims[1] / blockDim);
		}
		else {
			fuzzyCMedoids<<<blocks, threads>>>(d_data, d_medoids, d_result, d_dims, d_cost, numClusters, blocks, threads, 0);
		}

		CUDA_SAFE_CALL(cudaMemcpy(newCost, d_cost, sizeof(float), cudaMemcpyDeviceToHost));

		printf("%d: %f - %f\n", iter, *oldCost, *newCost);
		iter++;
	}

	free(dims);
	free(data);
	free(membership);
	free(medoids);
	free(finalMedoids);
	free(result);

	CUDA_SAFE_CALL(cudaFree(d_data));
	CUDA_SAFE_CALL(cudaFree(d_medoids));
	CUDA_SAFE_CALL(cudaFree(d_result));
	CUDA_SAFE_CALL(cudaFree(d_dims));

    return EXIT_SUCCESS;
}

void usage() {
	printf("Usage: kmedoidsCUDA <# clusters> <distance metric> <vol type> <vol min> <vol max> <input file> <output file>\n\n");
	printf("Distance Metric:\n");
	printf("\tEuclidean = 0\n");
	printf("\tMahattan  = 1\n");
	printf("\tMaximum   = 2\n");
	printf("Volume Type:\n");
	printf("\tBox     = 0\n");
	printf("\tSphere  = 1\n");
}
