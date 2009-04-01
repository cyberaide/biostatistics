/*#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>*/
#include <cutil.h>
#include <kmedoids_kernel.cu>
#include <kmedoids_bic_kernel.cu>

void calcVol(float results[][3], float* memb, int* out, int* dims, float min, float max, int n);
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
	if (argc != 8) {
		usage();
		return EXIT_FAILURE;
	}

	initRand();

	int GPUCount;
	int device = 0;

	CUDA_SAFE_CALL(cudaGetDeviceCount(&GPUCount));

	if (GPUCount > 1) {
		device = 1;
		CUDA_SAFE_CALL(cudaSetDevice(device));
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	printf("\nUsing device - %s\n\n", prop.name);

	//CUT_DEVICE_INIT(argc, argv);

	const int dimSize = 3 * sizeof(int);
	int* det = (int*)malloc(sizeof(int));
	int* numClusters = (int*)malloc(sizeof(int));
	*numClusters = atoi(argv[1]);

	int* dims = (int*)malloc(dimSize);
	float* temp = readData(argv[6], dims);
	float* data = (float*)malloc(sizeof(float) * dims[0] * dims[1]);

	memcpy(data, temp, sizeof(float) * dims[0] * dims[1]);
	free(temp);

	dims[2] = atoi(argv[2]);

	int volType = atoi(argv[3]);
	float volMin = atof(argv[4]);
	float volMax = atof(argv[5]);

	const int sizeFloat = dims[0] * dims[1] * sizeof(float);
	const int sizeMedoid = dims[0] * *numClusters * sizeof(float);
	//const int sizeInt = dims[0] * dims[1] * sizeof(int);
	const int sizeInt = dims[1] * sizeof(int);
	const int costSize = sizeof(float);
	const int sizeMemb = sizeof(float) * *numClusters * dims[1];
	const int bicSize = *numClusters * sizeof(float);

	float* costs = (float*)malloc(costSize);
	float* medoids = (float*)malloc(sizeMedoid);
	float* finalMedoids = (float*)malloc(sizeMedoid);
	float* membership = (float*)malloc(sizeMemb);
	float* BIC = (float*)malloc(bicSize);

	/*for (int i = 0; i < *numClusters; i++) {
		BIC[i] = 0;
	}*/

	// allocate memory for host result
	int* h_odata = (int*)malloc(sizeInt);

	// allocate device memory for data, medoids, and cost
	float* d_idata;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_idata, sizeFloat));

	float* d_medoids;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_medoids, sizeMedoid));

	float* d_membership;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_membership, sizeMemb));

	float* d_costs;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_costs, costSize));

	int* d_dims;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_dims, dimSize));

	int* d_numClusters;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_numClusters, sizeof(int)));

	int* d_det;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_det, sizeof(int)));

	float* d_bic;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_bic, bicSize));

	// allocate device memory for result
	int* d_odata;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_odata, sizeInt));

	// copy data to device
	CUDA_SAFE_CALL(cudaMemcpy(d_idata, data, sizeFloat, cudaMemcpyHostToDevice));

	// copy dims to device
	CUDA_SAFE_CALL(cudaMemcpy(d_dims, dims, dimSize, cudaMemcpyHostToDevice));

	// copy number of clusters to device.
	CUDA_SAFE_CALL(cudaMemcpy(d_numClusters, numClusters, sizeof(int), cudaMemcpyHostToDevice));

	// setup execution parameters
	int numThreads = 380; //300
	int test = (int)sqrt(dims[1]);

	if (test < numThreads) {
		numThreads = test;
	}

	int numBlocks = (int)sqrt(numThreads) * (int)sqrt(numThreads);

	float oldCost = 1;
	float newCost = 0;

	setCenters(data, medoids, *numClusters, dims);
	CUDA_SAFE_CALL(cudaMemcpy(d_medoids, medoids, sizeMedoid, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_bic, BIC, bicSize, cudaMemcpyHostToDevice));

	calculateBIC<<<numBlocks, numThreads>>>(d_idata, d_medoids, d_bic, d_dims, d_numClusters);

	CUDA_SAFE_CALL(cudaMemcpy(BIC, d_bic, bicSize, cudaMemcpyDeviceToHost));

	int bestNumClusters;
	float BICResult = -1;
	float BICTemp;

	for (int i = 1; i <= *numClusters; i++) {
		BICTemp = (dims[1] * log(BIC[i - 1] / dims[1])) + (i * log(dims[1]));

		if (BICResult < BICTemp) {
			bestNumClusters = i;
			BICResult = BICTemp;
		}
	}

	printf("Best number of clusters: %d\n", bestNumClusters);
	printf("Highest BIC: %f\n\n", BICResult);

	*numClusters = bestNumClusters;

	// copy number of clusters to device.
	CUDA_SAFE_CALL(cudaMemcpy(d_numClusters, numClusters, sizeof(int), cudaMemcpyHostToDevice));

	int choose = 0;

    /*for (int t = 0; t < 100; t++) {
    	unsigned int timer = 0;
    	CUT_SAFE_CALL(cutCreateTimer(&timer));
		CUT_SAFE_CALL(cutStartTimer(timer));*/

		/*while (newCost < oldCost) {
			if (choose == 1) {
				setCenters(data, medoids, *numClusters, dims);
				CUDA_SAFE_CALL(cudaMemcpy(d_medoids, medoids, sizeMedoid, cudaMemcpyHostToDevice));
			}

			*costs = 0;
			CUDA_SAFE_CALL(cudaMemcpy(d_costs, costs, costSize, cudaMemcpyHostToDevice));

			*det = 1;
			CUDA_SAFE_CALL(cudaMemcpy(d_det, det, sizeof(int), cudaMemcpyHostToDevice));

			kmedoids<<<numBlocks, numThreads>>>(d_idata, d_medoids, d_costs, d_odata, d_dims, d_numClusters, d_membership, d_det);

			// copy result from device to host
			CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata, sizeInt, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL(cudaMemcpy(membership, d_membership, sizeMemb, cudaMemcpyDeviceToHost));

			CUDA_SAFE_CALL(cudaMemcpy(costs, d_costs, costSize, cudaMemcpyDeviceToHost));
			oldCost = *costs;

			memcpy(finalMedoids, medoids, sizeMedoid);
			setCenters(data, medoids, *numClusters, dims);

			CUDA_SAFE_CALL(cudaMemcpy(d_medoids, medoids, sizeMedoid, cudaMemcpyHostToDevice));

			*costs = 0;
			CUDA_SAFE_CALL(cudaMemcpy(d_costs, costs, costSize, cudaMemcpyHostToDevice));

			*det = 0;
			CUDA_SAFE_CALL(cudaMemcpy(d_det, det, sizeof(int), cudaMemcpyHostToDevice));

			kmedoids<<<numBlocks, numThreads>>>(d_idata, d_medoids, d_costs, d_odata, d_dims, d_numClusters, d_membership, d_det);

			CUDA_SAFE_CALL(cudaMemcpy(costs, d_costs, costSize, cudaMemcpyDeviceToHost));
			newCost = *costs;

			//printf("%f : %f\n", oldCost, newCost);

			if (choose == 0) {
				choose = 1;
			}
		}*/

		/*CUT_SAFE_CALL(cutStopTimer(timer));
		printf("%f\n", cutGetTimerValue(timer));
		CUT_SAFE_CALL(cutDeleteTimer( timer));

		oldCost = 1;
		newCost = 0;
    }*/

	// check if kernel execution generated an error
	CUT_CHECK_ERROR("Kernel execution failed");

	float volResults[*numClusters][3];
	calcVol(volResults, membership, h_odata, dims, volMin, volMax, *numClusters);

	for (int i = 0; i < *numClusters; i++) {
		printf("Cluster #%d\n", i + 1);
		printf("Medoid: ");

		for (int j = 0; j < dims[0]; j++) {
			printf("%f ", finalMedoids[j + i * dims[0]]);
		}

		printf("\n");

		printf("Volume: %f\n", volResults[i][1]);
		printf("Occupancy: %f\n", volResults[i][2]);
		printf("Density: %f\n\n", volResults[i][0]);
	}

	writeData(data, finalMedoids, h_odata, dims, *numClusters, membership, argv[7]);

	free(costs);
	free(h_odata);
	free(data);
	free(medoids);
	free(finalMedoids);
	free(dims);
	free(membership);
	free(BIC);

	CUDA_SAFE_CALL(cudaFree(d_odata));
	CUDA_SAFE_CALL(cudaFree(d_idata));
	CUDA_SAFE_CALL(cudaFree(d_medoids));
	CUDA_SAFE_CALL(cudaFree(d_dims));
	CUDA_SAFE_CALL(cudaFree(d_membership));
	CUDA_SAFE_CALL(cudaFree(d_bic));

    //CUT_EXIT(argc, argv);
	return EXIT_SUCCESS;
}

void calcVol(float results[][3], float* memb, int* out, int* dims, float min, float max, int n) {
	float tempMin;
	float tempMax;
	float value;
	int index;

	for (int i = 0; i < n; i++) {
		tempMin = 0;
		tempMax = 0;
		results[i][0] = 0; // density
		results[i][1] = 0; // volume
		results[i][2] = 0; // occupancy

		for (int x = 0; x < dims[1]; x++) {
			index = x + i * n;

			if (memb[index] > 0 && memb[index] <= 100 && out[x] == i) {
				value = 1 / (100 - (memb[index] * 100));

				if (value > min && value < max) {
					results[i][2]++;

					if (tempMin == 0 || tempMin > memb[index]) {
						tempMin = memb[index];
					}

					if (tempMax == 0 || tempMax < memb[index]) {
						tempMax = memb[index];
					}
				}
			}
		}

		results[i][1] = tempMax - tempMin;
		results[i][0] = results[i][2] / results[i][1];
	}
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
