#ifndef __MDL_H__
#define __MDL_H__ 1

int* TabuSearch(float* matrix, char* inputFile);
int* MDL(float* events, float* clusters, float* mdlTime, char* inputFile);
int* MDLGPU(float* d_events, float* d_clusters, float* distanceMatrix, float* mdlTime, char* inputFile);

#endif
