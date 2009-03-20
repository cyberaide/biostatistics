/*
 * test.h
 *
 *  Created on: Mar 16, 2009
 *      Author: doug
 */

#ifndef CLUSTERINGUTILS_H_
#define CLUSTERINGUTILS_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
//#include <algorithm>

using namespace std;

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

void initRand() {
	srand((unsigned)(time(0)));
}

float randFloat() {
	return rand() / (float(RAND_MAX) + 1);
}

float randFloatRange(float min, float max) {
	if (min > max) {
		return randFloat() * (min - max) + max;
	}
	else {
		return randFloat() * (max - min) + min;
	}
}

int randInt(int max) {
	return int(rand() % max) + 1;
}

int randIntRange(int min, int max) {
	if (min > max) {
		return max + int(rand() % (min - max));
	}
	else {
		return min + int(rand() % (max - min));
	}
}

int getRandIndex(int w, int h) {
	return (randIntRange(0, w) - 1) + (randIntRange(0, h) - 1) * w;
}

bool contains(float* f, float p[], int n, int dims[]) {
	int count = 1;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < dims[0]; j++) {
			if (f[i + j * dims[0]] == p[j]) {
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

void setCenters(float* d, float* m, int n, int dims[]) {
	float temp[dims[0]];

	for (int i = 0; i < n; i++) {
		getPoints(d, temp, dims, randIntRange(0, dims[1]));

		while (contains(m, temp, n, dims)) {
			getPoints(d, temp, dims, randIntRange(0, dims[1]));
		}

		for (int j = 0; j < dims[0]; j++) {
			m[j + i * dims[0]] = temp[j];
		}
	}
}

void getPoints(float* d, float p[], int dims[], int i) {
	for (int j = 0; j < dims[0]; j++) {
		p[j] = d[i + j * dims[1]];
	}
}

float* readData(char* f, int* dims) {
	string line1;
	ifstream file(f);
	vector<string> lines;
	int dim = 0;
	char* temp;

	if (file.is_open()) {
		while(!file.eof()) {
			getline(file, line1);

			if (!line1.empty()) {
				lines.push_back(line1);
			}
		}

		file.close();
	}
	else {
		cout << "Unable to read the file " << f << endl;
		return NULL;
	}

	line1 = lines[0];
	string line2 (line1.begin(), line1.end());

	temp = strtok((char*)line1.c_str(), " ");

	while(temp != NULL) {
		dim++;
		temp = strtok(NULL, " ");
	}

	dims[0] = dim;
	dims[1] = (int)lines.size();

	float* data = (float*)malloc(sizeof(float) * dims[0] * dims[1]);

	temp = strtok((char*)line2.c_str(), " ");

	for (int i = 0; i < dims[1]; i++) {
		if (i != 0) {
			temp = strtok((char*)lines[i].c_str(), " ");
		}

		for (int j = 0; j < dims[0] && temp != NULL; j++) {
			data[i + j * dims[1]] = atof(temp);
			temp = strtok(NULL, " ");
		}
	}

	return data;
}

void writeData(float* d, float* m, int* c, int* dims, int nc, float* memb, const char* f) {
	ofstream file;

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

#endif /* CLUSTERINGUTILS_H_ */
