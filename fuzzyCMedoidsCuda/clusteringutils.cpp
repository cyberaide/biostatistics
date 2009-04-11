/*
 * clusteringutils.cpp
 *
 *  Created on: Apr 1, 2009
 *      Author: doug
 */

#include "../clusteringutils/clusteringutils.h"

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
