#ifndef _CMEANSCU_H_
#define _CMEANSCU_H_

__global__ void UpdateClusterCentersGPU(const float* oldClusters, const float* events, float* newClusters, float* distanceMatrix, float* denominator_result, int my_num_events);

__device__ float MembershipValueGPU(int clusterIndex, int eventIndex, const float* distanceMatrix, int my_num_events);

__device__ float CalculateDistanceGPU(const float* clusters, const float* events, int clusterIndex, int eventIndex, int my_num_events);

__global__ void EvaluateSolutionGPU(float* matrix, long config, float* score);
__global__ void CalculateQMatrixGPU(const float* events, const float* clusters, float* matrix);
__device__ float CalculateQIJ(float* events, float* distanceMatrix, int cluster_index_I, int cluster_index_J, float * EI, float * EJ, float *numMem, int my_num_events);
__device__ float CalculateQII(float* events, float* distanceMatrix, int cluster_index_I, float * EI,  float *numMem, int my_num_events);
__global__ void CalculateQMatrixGPUUpgrade(const float* events, const float* clusters, float* matrix, float* distanceMatrix, int start_row, int my_num_events);

__global__ void ComputeDistanceMatrix(const float* clusters, const float* events, float* matrix, int my_num_events);
__global__ void ComputeMembershipMatrix(float* distances, float* memberships, int my_num_events);

__device__ float MembershipValueDist(float* distanceMatrix, int clusterIndex, int eventIndex, float distance, int my_num_events);

#endif
