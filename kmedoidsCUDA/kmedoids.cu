#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <kmedoids_kernel.cu>
#include <kmedoids_bic_kernel.cu>

void initRand();
int randInt(int max);
int randInt(int min, int max);
float randFloat();
float randFloat(float min, float max);
void chooseMedoids(float* m, float* d, int n, int* dims);
bool contains(float* m, float p[], int n, int* dims);
void getPoints(float* d, float p[], int* dims, int i);
void calcVol(float results[][3], float* memb, int* out, int* dims, float min, float max, int n);
void usage();

extern "C"
void writeData(float* d, float* m, int* c, int* dims, int nc, float* memb, char* f);

extern "C"
float* readData(char* f, int* dims);

int main( int argc, char** argv) {
	if (argc != 8) {
		usage();
		return EXIT_FAILURE;
	}

	CUT_DEVICE_INIT(argc, argv);

	const int dimSize = 3 * sizeof(int);
	int* det = (int*)malloc(sizeof(int));
	int* numClusters = (int*)malloc(sizeof(int));
	*numClusters = atoi(argv[1]);

	int* dims = (int*)malloc(dimSize);
	float* data = readData(argv[6], dims);

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
	float* membership = (float*)malloc(sizeMemb);
	float* BIC = (float*)malloc(bicSize);

	for (int i = 0; i < *numClusters; i++) {
		BIC[i] = 0;
	}

	// allocate memory for host result
	int* h_odata = (int*)malloc(sizeInt);

	initRand();

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
	float tempMedoid[dims[0]];
	int mi;
	int di;

	chooseMedoids(medoids, data, *numClusters, dims);
	CUDA_SAFE_CALL(cudaMemcpy(d_medoids, medoids, sizeMedoid, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_bic, BIC, bicSize, cudaMemcpyHostToDevice));

	calculateBIC<<<numBlocks, numThreads>>>(d_idata, d_medoids, d_bic, d_dims, d_numClusters);

	CUDA_SAFE_CALL(cudaMemcpy(BIC, d_bic, bicSize, cudaMemcpyDeviceToHost));

	int bestNumClusters;
	float BICResult = -1;

	for (int i = 1; i <= *numClusters; i++) {
		if (BICResult == -1 || BICResult < BIC[i - 1]) {
			bestNumClusters = i;
			BICResult = BIC[i - 1];
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

		while (newCost < oldCost) {
			if (choose == 1) {
				chooseMedoids(medoids, data, *numClusters, dims);
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

			mi = randInt(*numClusters) - 1;
			di = randInt(dims[0] * dims[1]) - 1;

			for (int i = 0; i < dims[0]; i++) {
				tempMedoid[i] = medoids[i + mi * dims[0]];
				medoids[i + mi * dims[0]] = data[di + i * dims[0]];
			}

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
		}

		/*CUT_SAFE_CALL(cutStopTimer(timer));
		printf("%f\n", cutGetTimerValue(timer));
		CUT_SAFE_CALL(cutDeleteTimer( timer));

		oldCost = 1;
		newCost = 0;
    }*/

	// check if kernel execution generated an error
	CUT_CHECK_ERROR("Kernel execution failed");

	for (int i = 0; i < dims[0]; i++) {
		medoids[i + mi * dims[0]] = tempMedoid[i];
	}

	float volResults[*numClusters][3];
	calcVol(volResults, membership, h_odata, dims, volMin, volMax, *numClusters);

	for (int i = 0; i < *numClusters; i++) {
		printf("Cluster #%d\n", i + 1);
		printf("Medoid: ");

		for (int j = 0; j < dims[0]; j++) {
			printf("%f ", medoids[j + i * dims[0]]);
		}

		printf("\n");

		printf("Volume: %f\n", volResults[i][1]);
		printf("Occupancy: %f\n", volResults[i][2]);
		printf("Density: %f\n\n", volResults[i][0]);
	}

	writeData(data, medoids, h_odata, dims, *numClusters, membership, argv[7]);

	free(costs);
	free(h_odata);
	free(data);
	free(medoids);
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

void initRand() {
    srand((unsigned)(time(0)));
}

int randInt(int max) {
	return int(rand() % max) + 1;
}

int randInt(int min, int max) {
    if (min > max) {
        return max + int(rand() % (min - max));
    }
    else {
        return min + int(rand() % (max - min));
    }
}

float randFloat() {
    return rand() / (float(RAND_MAX) + 1);
}

float randFloat(float min, float max) {
    if (min > max) {
        return randFloat() * (min - max) + max;
    }
    else {
        return randFloat() * (max - min) + min;
    }
}

void chooseMedoids(float* m, float* d, int n, int* dims) {
	float temp[dims[0]];

	for (int i = 0; i < n; i++) {
		getPoints(d, temp, dims, randInt(dims[1]) - 1);

		while (contains(m, temp, n, dims)) {
			getPoints(d, temp, dims, randInt(dims[1]) - 1);
		}

		for (int j = 0; j < dims[0]; j++) {
			m[j + i * dims[0]] = temp[j];
		}
	}
}

void getPoints(float* d, float p[], int* dims, int i) {
	for (int j = 0; j < dims[0]; j++) {
		p[j] = d[i + j * dims[1]];
	}
}

bool contains(float* m, float p[], int n, int* dims) {
	int count = 1;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < dims[0]; j++) {
			if (m[i + j * dims[0]] == p[j]) {
				count++;
			}
		}
	}

	if (count == dims[0]) {
		return true;
	}
	else {
		return false;
	}
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
