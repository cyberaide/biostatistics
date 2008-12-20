/*
 * writeData.cpp
 *
 *  Created on: Oct 9, 2008
 *      Author: doug
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

int clusterColor(float i, int nc);

extern "C"
void writeData(float* d, float* m, int* c, int* dims, int nc, float* memb, char* f);

void writeData(float* d, float* m, int* c, int* dims, int nc, float* memb, char* f) {
	ofstream file;
	/*file.open("medoids.dat");

	for (int i = 0; i < nc; i++) {
		for (int j = 0; j < 2; j++) {
			file << fixed << setprecision(10) << m[j + i * dims[0]] << " ";
		}

		file << clusterColor(i + 1, nc) << endl;
	}

	file.close();
	file.open("output.dat");

	for (int i = 0; i < dims[1]; i++) {

		for (int j = 0; j < 2; j++) {
			if (c[i] >= 0 && c[i] <= nc) {
				file << fixed << setprecision(10) << " " << d[i + j * dims[1]] << " ";
			}
		}

		if (c[i] >= 0 && c[i] <= nc) {
			file << clusterColor(c[i] + 1, nc) << endl;
		}
	}

	file.close();*/
	file.open(f);

	file << "Data: last " << nc << " columns indicate cluster membership." << endl << endl;

	for (int i = 0; i < dims[1]; i++) {
		if (c[i] >= 0 && c[i] <= nc) {
			for (int j = 0; j < dims[0]; j++) {
				file << fixed << setprecision(10) << d[i + j * dims[1]] << " ";
			}

			for (int x = 0; x < nc; x++) {
				file << fixed << setprecision(10) << memb[i + x * nc] << " ";
			}

			file << endl;
		}
	}

	int identity[nc][nc];

	for (int i = 0; i < nc; i++) {
		for (int j = 0; j < nc; j++) {
			if (i == j) {
				identity[i][j] = 1;
			}
			else {
				identity[i][j] = 0;
			}
		}
	}

	file << endl << "Medoids: last " << nc << " columns is the identity matrix." << endl << endl;

	for (int i = 0; i < nc; i++) {
		for (int j = 0; j < dims[0]; j++) {
			file << fixed << setprecision(10) << m[j + i * dims[0]] << " ";
		}

		for (int j = 0; j < nc; j++) {
			file << identity[i][j] << " ";
		}

		file << endl;
	}

	file.close();
}

int clusterColor(float i, int nc) {
	return (int)((i / nc) * 256);
}
