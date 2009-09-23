#include <stdio.h>
#include <float.h>
#include <kmeans.h>

__device__ float CalcDist(float* refVecs,  const float* allEvents, int clusterIndex, int numThreadOffset);


__global__ void kmeans_distance(const float* allClusters, const float* allEvents, int* cM){

	
	
	__shared__ float myClusters[NUM_CLUSTERS*ALL_DIMENSIONS];
	float min, tmp;
	int min_clust = 0;
	int i,j;

	

	
	//pull all clusters into shared memory
	for(i =0; i < NUM_CLUSTERS*ALL_DIMENSIONS; i += NUM_THREADS) { 
		myClusters[threadIdx.x + i] = allClusters[threadIdx.x + i];
		
	}
	__syncthreads();
	
	for(i = blockIdx.x*NUM_THREADS; i < NUM_EVENTS; i+=NUM_THREADS*NUM_BLOCKS){ //get events in groups of NUM_THREADS

		min = FLT_MAX;

		
		for(j = 0; j < NUM_CLUSTERS; j++){ 
			tmp = CalcDist(myClusters,  allEvents, j, i);
			if(tmp < min){
				min = tmp;
				min_clust = j;
			}
			cM[threadIdx.x + i] = min_clust;
		}
		__syncthreads();
	}



}

__device__ float CalcDist(float* refVecs,  const float* allEvents, int clusterIndex, int numThreadOffset){
	float sum = 0.0;
	int i, j;

		for(i = 0; i < ALL_DIMENSIONS; i++){
			float tmp = allEvents[(threadIdx.x + numThreadOffset)*ALL_DIMENSIONS + i] - refVecs[clusterIndex*ALL_DIMENSIONS + i];
			sum += tmp*tmp;
		}

	return sqrt(sum);
}