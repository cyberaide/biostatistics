#ifndef _CMEANSCU_H_
#define _CMEANSCU_H_


__global__ void UpdateClusterCentersGPU(const float* oldClusters, const float* events, float* newClusters, float* distanceMatrix);

__device__ float MembershipValueGPU(const float* events, int clusterIndex, int eventIndex, const float* distanceMatrix);

__device__ float CalculateDistanceGPU(const float* clusters, const float* events, int clusterIndex, int eventIndex);

__global__ void EvaluateSolutionGPU(float* matrix, long config, float* score);
__device__ float CalculateQIJ(float* events, float* clusters, int cluster_index_I, int cluster_index_J, float * EI, float * EJ, float *numMem, float* distanceMatrix);
__device__ float CalculateQII(float* events, float* clusters, int cluster_index_I, float * EI,  float *numMem, float* distanceMatrix);
__global__ void CalculateQMatrixGPUUpgrade(const float* events, const float* clusters, float* matrix, float* distanceMatrix);

__global__ void ComputeDistanceMatrix(const float* clusters, const float* events, float* matrix);

__device__ float MembershipValueDist(const float* events, int clusterIndex, int eventIndex, float distance, float* distanceMatrix);


#endif
