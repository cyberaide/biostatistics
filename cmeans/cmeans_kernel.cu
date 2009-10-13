#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
#include <cmeans.h>
#include <cmeanscu.h>
#include <float.h>






__global__ void UpdateClusterCentersGPU(const float* oldClusters, const float* events, float* newClusters){

	float membershipValue;//, denominator;
	
	__shared__ float numerators[ALL_DIMENSIONS*NUM_NUM];
	__shared__ float myClusters[ALL_DIMENSIONS*NUM_CLUSTERS];
	__shared__ float denominators[NUM_NUM];
	
	
	  int index = threadIdx.x+threadIdx.y*THREADS_PER_EVENT;
	  denominators[threadIdx.y] = 0;
	  for(int j = 0; j < ALL_DIMENSIONS*NUM_NUM; j+=NUM_THREADS){
		  if(j + index < ALL_DIMENSIONS*NUM_NUM)	{
			numerators[j+index] = 0;
		  }
	  }
	  for(int j = 0; j < ALL_DIMENSIONS*NUM_CLUSTERS; j+=NUM_THREADS){
			if(j + index < ALL_DIMENSIONS*NUM_CLUSTERS)	{
				myClusters[j+index] = oldClusters[j+index];
			}
	  }
	 


	  __syncthreads();
		  
	  for(int j = 0; j < NUM_EVENTS; j+=NUM_NUM){
		  if((j+threadIdx.y) < NUM_EVENTS) {
			  
			membershipValue = MembershipValueGPU(myClusters, events, blockIdx.x, j+threadIdx.y);
		
			for(int k = 0; k < ALL_DIMENSIONS; k+=THREADS_PER_EVENT){
				numerators[threadIdx.y*ALL_DIMENSIONS + threadIdx.x + k] += events[(threadIdx.y + j)*ALL_DIMENSIONS + threadIdx.x+ k]*membershipValue;
					
			}
			denominators[threadIdx.y] += membershipValue;
				
		  }
	  } 

	  

	 
	  __syncthreads();
	  
	
	if(index < ALL_DIMENSIONS){
	for(int j = 1; j < NUM_NUM; j++){
		numerators[index] += numerators[j*ALL_DIMENSIONS +index];
	  }  
	  numerators[index] = numerators[index] / (float)NUM_NUM;
	  }
	  if(index == 0){
		  for(int j = 1; j < NUM_NUM; j++){
			denominators[0] += denominators[j];
		  }
		  denominators[0] = denominators[0]/NUM_NUM;
	  }
	  __syncthreads();

		
	  for(int j = 0; j < ALL_DIMENSIONS; j+=NUM_THREADS){
		  if(j+index< ALL_DIMENSIONS){
			newClusters[blockIdx.x*ALL_DIMENSIONS + j  + index] = numerators[index + j ]/denominators[0];
		  }
	  }  
	
	  
	
}

__device__ float MembershipValueGPU(const float* clusters, const float* events, int clusterIndex, int eventIndex){
	float myClustDist = 0;
	myClustDist = CalculateDistanceGPU(clusters, events, clusterIndex, eventIndex);
	
	float sum =0;
	float otherClustDist;
	for(int j = 0; j< NUM_CLUSTERS; j++){
		otherClustDist = CalculateDistanceGPU(clusters, events, j, eventIndex);
		
		if(otherClustDist < .000001)
			return 0.0;
		sum += pow((myClustDist/otherClustDist),(2/(FUZZINESS-1)));
		
	}
	return 1/sum;
}

__device__ float CalculateDistanceGPU(const float* clusters, const float* events, int clusterIndex, int eventIndex){

	float sum = 0;
	float tmp;
#if DISTANCE_MEASURE == 0
			for(int i = 0; i < ALL_DIMENSIONS; i++){
				tmp = events[eventIndex*ALL_DIMENSIONS + i] - clusters[clusterIndex*ALL_DIMENSIONS + i];
				sum += tmp*tmp;
			}
			sum = sqrt(sum);
#endif
#if DISTANCE_MEASURE == 1
			for(int i = 0; i < ALL_DIMENSIONS; i++){
				tmp = events[eventIndex*ALL_DIMENSIONS + i] - clusters[clusterIndex*ALL_DIMENSIONS + i];
				sum += abs(tmp);
			}
#endif
#if DISTANCE_MEASURE == 2 
			for(int i = 0; i < ALL_DIMENSIONS; i++){
				tmp = abs(events[eventIndex*ALL_DIMENSIONS + i] - clusters[clusterIndex*ALL_DIMENSIONS + i]);
				if(tmp > sum)
					sum = tmp;
			}
#endif

	return sum;
}


__device__ float CalculateQII(const float* events, const float* clusters, int cluster_index_I, float* EI, float* numMem){
	
	EI[threadIdx.x] = 0;
	numMem[threadIdx.x] = 0;
	
	for(int i = 0; i < NUM_EVENTS; i+=Q_THREADS){
		float distance = CalculateDistanceGPU(clusters, events, cluster_index_I, i+threadIdx.x);
		float memVal = MembershipValueDist(clusters, events,  cluster_index_I, i+threadIdx.x, distance);
		
		if(memVal > MEMBER_THRESH){
			EI[threadIdx.x] += pow(memVal, 2) * pow(distance, 2);
			numMem[threadIdx.x]++;
		}
	}
	
	//printf("block = %d, numMem = %f, EI = %f\n", blockIdx.x, numMem, EI);
	__syncthreads();
	
	if(threadIdx.x == 0){
		for(int i = 1; i < Q_THREADS; i++){
			EI[0] += EI[i];
			numMem[0] += numMem[i];
		}
	}
	__syncthreads();

	return ((((float)K1) * numMem[0]) - (((float)K2) * EI[0]) - (((float)K3) * ALL_DIMENSIONS));

}


__device__ float CalculateQIJ(const float* events, const float* clusters, int cluster_index_I, int cluster_index_J, float * EI, float * EJ, float *numMem){
	
	
	EI[threadIdx.x] = 0;
	EJ[threadIdx.x] = 0;
	numMem[threadIdx.x] = 0;
	
	for(int i = 0; i < NUM_EVENTS; i+=Q_THREADS){
		if(i+threadIdx.x < NUM_EVENTS){
			float distance = CalculateDistanceGPU(clusters, events, cluster_index_I, i+threadIdx.x);
			float memValI = MembershipValueDist(clusters, events, cluster_index_I, i+threadIdx.x, distance);
		
			if(memValI > MEMBER_THRESH){
				EI[threadIdx.x] += pow(memValI, 2) * pow(distance, 2);
				
			}
			
			distance = CalculateDistanceGPU(clusters, events, cluster_index_J, i+threadIdx.x);
			float memValJ = MembershipValueDist(clusters, events, cluster_index_J, i+threadIdx.x, distance);
			if(memValJ > MEMBER_THRESH){
				EJ[threadIdx.x] += pow(memValJ, 2) * pow(distance, 2);
			}
			if(memValI > MEMBER_THRESH && memValJ > MEMBER_THRESH){
				numMem[threadIdx.x]++;
			}
			
		}
		
	
	}
	__syncthreads();

	if(threadIdx.x == 0){
		for(int i = 1; i < Q_THREADS; i++){
			EI[0] += EI[i];
			EJ[0] += EJ[i];
			numMem[0] += numMem[i];
		}
	}

	__syncthreads();
	float EB = (EI[0] > EJ[0]) ? EI[0] : EJ[0];
	return ((-1*((float)K1)*numMem[0]) + ((float)K2)*EB);

}

__global__ void CalculateQMatrixGPU(const float* events, const float* clusters, float* matrix){
	__shared__ float myClusters[NUM_CLUSTERS*ALL_DIMENSIONS];
	__shared__ float EI[Q_THREADS];
	__shared__ float EJ[Q_THREADS];
	__shared__ float numMem[Q_THREADS];
	for(int j = 0; j < NUM_CLUSTERS*ALL_DIMENSIONS; j+= Q_THREADS){
		if(j+threadIdx.x < NUM_CLUSTERS*ALL_DIMENSIONS){
			myClusters[j+threadIdx.x] = clusters[j+threadIdx.x];

		}
	}
	__syncthreads();
	for(int j = 0; j < NUM_CLUSTERS; j++){
			
		if(blockIdx.x == j){
			matrix[blockIdx.x*NUM_CLUSTERS + j ] = CalculateQII(events, myClusters, blockIdx.x, EI, numMem);
				
		} else{
				
			matrix[blockIdx.x*NUM_CLUSTERS + j] = CalculateQIJ(events, myClusters, blockIdx.x, j, EI, EJ, numMem);
				
		}
			
		__syncthreads();
	}
	
}

__global__ void CalculateQMatrixGPUUpgrade(const float* events, const float* clusters, float* matrix){
	__shared__ float myClusters[NUM_CLUSTERS*ALL_DIMENSIONS];
	__shared__ float EI[Q_THREADS];
	__shared__ float EJ[Q_THREADS];
	__shared__ float numMem[Q_THREADS];
	for(int j = 0; j < NUM_CLUSTERS*ALL_DIMENSIONS; j+= Q_THREADS){
		if(j+threadIdx.x < NUM_CLUSTERS*ALL_DIMENSIONS){
			myClusters[j+threadIdx.x] = clusters[j+threadIdx.x];

		}
	}
	__syncthreads();
	//printf("blockIdx.x = %d, blockIdx.y = %d\n", blockIdx.x, blockIdx.y);
	if(blockIdx.x == blockIdx.y){
		matrix[blockIdx.x*NUM_CLUSTERS + blockIdx.y ] = CalculateQII(events, myClusters, blockIdx.x, EI, numMem);
	}
	else{
		matrix[blockIdx.x*NUM_CLUSTERS + blockIdx.y] = CalculateQIJ(events, myClusters, blockIdx.x, blockIdx.y, EI, EJ, numMem);
	}	
	
	
}

/*__global__ void EvaluateSolutionGPU(float* matrix, long config, float* score){
	float partial[NUM_CLUSTERS] = {0};
	for(int i = 0; i < NUM_CLUSTERS; i++){
		for(int j = 0; j < NUM_CLUSTERS; j++){
			partial[i] += ((config & (1 << (NUM_CLUSTERS - j - 1))) == 0) ? 0 : matrix[i + j*NUM_CLUSTERS];
		}
	} 
	float myScore = 0;
	for(int i = 0; i < NUM_CLUSTERS; i++){
		myScore += ((config & (1 << (NUM_CLUSTERS - i - 1))) == 0) ? 0 : partial[i];
	}
	*score = myScore;
}*/



__device__ float MembershipValueDist(const float* clusters, const float* events, int clusterIndex, int eventIndex, float distance){
	float sum =0;
	float otherClustDist;
	for(int j = 0; j< NUM_CLUSTERS; j++){
		otherClustDist = CalculateDistanceGPU(clusters, events, j, eventIndex);	
		if(otherClustDist < .000001)
			return 0.0;
		sum += pow((float)(distance/otherClustDist),float(2/(FUZZINESS-1)));
	}
	return 1/sum;
}


