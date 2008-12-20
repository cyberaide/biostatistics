/*
 * readData.cu
 *
 *  Created on: Nov 4, 2008
 *      Author: doug
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

extern "C"
float* readData(char* f, int* dims);

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
