/*#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>*/
#include <cutil.h>
#include <fuzzyCMedoids_kernel.cu>
#include <membership_kernel.cu>
#include "cmedoids.h"

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
extern "C" void writeData(float* d, float* m, int* dims, int nc, float* memb, const char* f);
extern "C" int clusterColor(float i, int nc);

int main( int argc, char** argv) {
	if (argc != 2) {
		usage();
		return EXIT_FAILURE;
	}

	CUT_DEVICE_INIT(argc, argv);

	initRand();

	int dimSize = 2 * sizeof(int);
	int numClusters = NUM_CLUSTERS;

	int* dims = (int*)malloc(dimSize);
	float* data = readData(argv[1], dims);

	int medoidSize = sizeof(float) * numClusters * dims[0];
	int dataSize = sizeof(float) * dims[0] * dims[1];
	int membSize = sizeof(float) * numClusters * dims[1];

	// host
	float* membership = (float*)malloc(membSize);
	float* medoids = (float*)malloc(medoidSize);
	float* finalMedoids = (float*)malloc(medoidSize);
	float* oldCost = (float*)malloc(sizeof(float));
	float* newCost = (float*)malloc(sizeof(float));

	// cuda
	float* d_data;
	float* d_memb;
	float* d_medoids;
	float* d_cost;

	*oldCost = 1;
	*newCost = 0;

	int MAXITER = 5;
	int iter = 0;

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_data, dataSize));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_medoids, medoidSize));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_memb, membSize));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_cost, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpy(d_data, data, dataSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_memb, membership, membSize, cudaMemcpyHostToDevice));

	//for (int i = 0; i < 10; i++) {
		unsigned int timer = 0;
		CUT_SAFE_CALL( cutCreateTimer( &timer));
		CUT_SAFE_CALL( cutStartTimer( timer));

		//while (*oldCost > *newCost && iter < MAXITER) {
		//while (iter < MAXITER) {
			*oldCost = 0;
			*newCost = 0;

			CUDA_SAFE_CALL(cudaMemcpy(d_cost, oldCost, sizeof(float), cudaMemcpyHostToDevice));

			setCenters(data, medoids, numClusters, dims);
			CUDA_SAFE_CALL(cudaMemcpy(d_medoids, medoids, medoidSize, cudaMemcpyHostToDevice));

			memcpy(finalMedoids, medoids, medoidSize);

			fuzzyCMedoids<<<NUM_BLOCKS, NUM_THREADS>>>(d_data, d_medoids, d_cost);

			CUDA_SAFE_CALL(cudaMemcpy(oldCost, d_cost, sizeof(float), cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL(cudaMemcpy(d_cost, newCost, sizeof(float), cudaMemcpyHostToDevice));

			setCenters(data, medoids, numClusters, dims);
			CUDA_SAFE_CALL(cudaMemcpy(d_medoids, medoids, medoidSize, cudaMemcpyHostToDevice));

			fuzzyCMedoids<<<NUM_BLOCKS, NUM_THREADS>>>(d_data, d_medoids, d_cost);

			CUDA_SAFE_CALL(cudaMemcpy(newCost, d_cost, sizeof(float), cudaMemcpyDeviceToHost));

			//printf("%d: %f - %f\n", iter, *oldCost, *newCost);
			//iter++;
		//}

		CUDA_SAFE_CALL(cudaMemcpy(d_medoids, finalMedoids, medoidSize, cudaMemcpyHostToDevice));

		calcMembership<<<NUM_BLOCKS, 50>>>(d_data, d_medoids, d_memb);

		CUDA_SAFE_CALL(cudaMemcpy(membership, d_memb, membSize, cudaMemcpyDeviceToHost));

		cudaThreadSynchronize();
		CUT_SAFE_CALL( cutStopTimer( timer));
		//printf("\nProcessing time: %f (ms)\n", cutGetTimerValue( timer));
		printf("%f\n", cutGetTimerValue( timer));
		CUT_SAFE_CALL( cutDeleteTimer( timer));

		CUDA_SAFE_CALL(cudaMemcpy(membership, d_memb, membSize, cudaMemcpyDeviceToHost));

		*oldCost = 1;
		*newCost = 0;
		//iter = 0;
	//}

	printf("Saving output file.\n");
	writeData(data, finalMedoids, dims, numClusters, membership, "output.dat");

	free(dims);
	free(data);
	free(membership);
	free(medoids);
	free(finalMedoids);

	CUDA_SAFE_CALL(cudaFree(d_data));
	CUDA_SAFE_CALL(cudaFree(d_medoids));
	CUDA_SAFE_CALL(cudaFree(d_memb));

    return EXIT_SUCCESS;
}

void usage() {
	printf("Usage: ./fuzzyCMedoids <input file>\n");

	/*printf("Usage: kmedoidsCUDA <# clusters> <distance metric> <vol type> <vol min> <vol max> <input file> <output file>\n\n");
	printf("Distance Metric:\n");
	printf("\tEuclidean = 0\n");
	printf("\tMahattan  = 1\n");
	printf("\tMaximum   = 2\n");
	printf("Volume Type:\n");
	printf("\tBox     = 0\n");
	printf("\tSphere  = 1\n");*/
}
