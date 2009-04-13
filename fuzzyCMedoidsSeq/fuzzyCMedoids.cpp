/*
 * fuzzyKMedoids.cpp
 *
 *  Created on: Apr 2, 2009
 *      Author: doug
 */

//#include <sys/time.h>
//#include <sys/resource.h>
#include <time.h>
//#include "../../common/inc/cutil.h"
#include "../clusteringutils/clusteringutils.h"

float calculateCost(float* d, float* m, int nc, int dims[]);
void calculateMembership(float* data, float* medoids, float* memb, int dims[], int numClusters, int m);
float calculateDist(int i, int x, float* d, float* m, int dims[], int n);
void usage();

int main(int argc, char** argv) {
	if (argc != 3) {
		usage();
		return EXIT_FAILURE;
	}

	initRand();

	int numClusters = atoi(argv[1]);
	int dims[2];

	// read the data from a file
	float* data;
	data = readData(argv[2], dims);

	int oldCost = 1;
	int newCost = 0;
	int sizeMedoids = sizeof(float) * numClusters * dims[0];
	int sizeMemb = sizeof(float) * numClusters * dims[1];
	int MAXITER = 5;
	int iter = 0;

	float* medoids = (float*)malloc(sizeMedoids);
	float* finalMedoids = (float*)malloc(sizeMedoids);
	float* membership = (float*)malloc(sizeMemb);

	clock_t start, end;

	for (int i = 0; i < 10; i++) {
		start = clock();

		//while (oldCost > newCost && iter < MAXITER) {
		//while (iter < MAXITER) {
			setCenters(data, medoids, numClusters, dims);
			oldCost = calculateCost(data, medoids, numClusters, dims);
			//asdf
			memcpy(finalMedoids, medoids, sizeMedoids);

			setCenters(data, medoids, numClusters, dims);
			newCost = calculateCost(data, medoids, numClusters, dims);

			//cout << iter << ": " << oldCost << " - " << newCost << endl;
			//iter++;
		//}

		calculateMembership(data, finalMedoids, membership, dims, numClusters, 2);

		end = clock();
		//cout << endl << "Processing time: " << ((float)(end - start) / (float)(CLOCKS_PER_SEC)) * (float)1e3 << " (ms)" << endl;
		cout << ((float)(end - start) / (float)(CLOCKS_PER_SEC)) * (float)1e3 << endl;

		oldCost = 1;
		newCost = 0;
		//iter = 0;
	}

	cout << "Saving output file." << endl;
	writeData(data, finalMedoids, dims, numClusters, membership, "output.dat");

	free(data);
	free(medoids);
	free(finalMedoids);
	free(membership);

	return EXIT_SUCCESS;
}

float calculateCost(float* d, float* m, int nc, int dims[]) {
	float cost = 0;
	float dist = 0;
	float leastDist = -1;

	for (int i = 0; i < dims[1]; i++) {
		for (int j = 0; j < nc; j++) {
			dist = calculateDist(i, j, d, m, dims, nc);

			if (leastDist == -1 || dist < leastDist) {
				leastDist = dist;
			}
		}

		cost += leastDist;
		leastDist = -1;
	}

	return cost;
}

void calculateMembership(float* data, float* medoids, float* memb,  int dims[], int numClusters, int m) {
	float numerator;
	float denominator = 0;
	float exp = 1 / (m - 1);
	float base;

	for (int i = 0; i < dims[1]; i++) {
		for (int j = 0; j < numClusters; j++) {
			base = calculateDist(i, j, data, medoids, dims, numClusters);
			numerator = pow(base, exp);

			for (int x = 0; x < numClusters; x++) {
				base = calculateDist(i, x, data, medoids, dims, numClusters);
				denominator += pow(base, exp);
			}

			memb[j + i * numClusters] = numerator / denominator;
			denominator = 0;
		}
	}
}

float calculateDist(int i, int x, float* d, float* m, int dims[], int n) {
	float sum = 0;

	for (int j = 0; j < dims[0]; j++) {
		sum += pow(d[i + j * dims[1]] - m[j + x * dims[0]], 2);
	}

	return sqrt(sum);
}

void usage() {
	cout << "Usage: ./fuzzyCMedoids <clusters> <input file>" << endl;
}
