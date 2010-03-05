#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
#include <cmeansMultiGPU.h>
#include <cmeansMultiGPUcu.h>
#include <float.h>

__device__ float parallelSum(float* data, const unsigned int ndata) {
  const unsigned int tid = threadIdx.x;
  float t;

  __syncthreads();

  // Butterfly sum.  ndata MUST be a power of 2.
  for(unsigned int bit = ndata >> 1; bit > 0; bit >>= 1) {
    t = data[tid] + data[tid^bit];  __syncthreads();
    data[tid] = t;                  __syncthreads();
  }
  return data[tid];
}

__global__ void UpdateClusterCentersGPU(const float* oldClusters, const float* events, float* newClusters, float* memberships, float* denoms, int my_num_events) {

    float membershipValue;//, denominator;

    int d = blockIdx.y;
    int event_matrix_offset = my_num_events*d;
    int membership_matrix_offset = my_num_events*blockIdx.x;

    __shared__ float numerators[NUM_THREADS_UPDATE];

    // Sum of the memberships computed by each thread
    // The sum of all of these denominators together is effectively the size of the cluster
    __shared__ float denominators[NUM_THREADS_UPDATE];
        
    int tid = threadIdx.x;

    // initialize numerators and denominators to 0
    denominators[tid] = 0;
    numerators[tid] = 0;

    __syncthreads();
     
    // Compute new membership value for each event
    // Add its contribution to the numerator and denominator for that thread
    for(int j = tid; j < my_num_events; j+=NUM_THREADS_UPDATE){
        membershipValue = memberships[membership_matrix_offset + j];
        numerators[tid] += events[event_matrix_offset + j]*membershipValue;
        denominators[tid] += membershipValue;
    } 

    __syncthreads();

    if(tid == 0){
        // Sum up the numerator/denominator, one for this block
        for(int j = 1; j < NUM_THREADS_UPDATE; j++){
            numerators[0] += numerators[j];
        }  
        if(d == 0) {
            for(int j = 1; j < NUM_THREADS_UPDATE; j++){
                denominators[0] += denominators[j];
            }
            denoms[blockIdx.x] = denominators[0];
        }
        // Set the new center for this block    
        //newClusters[blockIdx.x*NUM_DIMENSIONS + d] = numerators[0]/denominators[0];
        newClusters[blockIdx.x*NUM_DIMENSIONS + d] = numerators[0];
    }
}

__global__ void UpdateClusterCentersGPU2(const float* oldClusters, const float* events, float* newClusters, float* memberships, int my_num_events) {
    float membershipValue;
    float eventValue;

    // Compute cluster range for this block
    int c_start = blockIdx.x*NUM_CLUSTERS_PER_BLOCK;
    int num_c = NUM_CLUSTERS_PER_BLOCK;

    // Handle boundary condition
    if(blockIdx.x == gridDim.x-1 && NUM_CLUSTERS % NUM_CLUSTERS_PER_BLOCK) {
        num_c = NUM_CLUSTERS % NUM_CLUSTERS_PER_BLOCK;
    }

    // Dimension index
    int d = blockIdx.y;
    int event_matrix_offset = my_num_events*d;

    __shared__ float numerators[NUM_THREADS_UPDATE*NUM_CLUSTERS_PER_BLOCK];

    int tid = threadIdx.x;

    // initialize numerators and denominators to 0
    for(int c = 0; c < num_c; c++) {
        numerators[c*NUM_THREADS_UPDATE+tid] = 0;
    }

    // Compute new membership value for each event
    // Add its contribution to the numerator and denominator for that thread
    for(int j = tid; j < my_num_events; j+=NUM_THREADS_UPDATE){
        eventValue = events[event_matrix_offset + j];
        for(int c = 0; c < num_c; c++) {
            membershipValue = memberships[(c+c_start)*my_num_events + j];
            numerators[c*NUM_THREADS_UPDATE+tid] += eventValue*membershipValue;
        }
    }

    __syncthreads();

    for(int c = 0; c < num_c; c++) {
        numerators[c*NUM_THREADS_UPDATE+tid] = parallelSum(&numerators[NUM_THREADS_UPDATE*c],NUM_THREADS_UPDATE);
    }

    __syncthreads();

    if(tid == 0){
        for(int c = 0; c < num_c; c++) {
            // Set the new center for this block
            newClusters[(c+c_start)*NUM_DIMENSIONS + d] = numerators[c*NUM_THREADS_UPDATE];
        }
    }

}

__global__ void UpdateClusterCentersGPU3(const float* oldClusters, const float* events, float* newClusters, float* memberships, int my_num_events) {
    float membershipValue;
    float eventValue;

    // Compute cluster range for this block
    int c_start = blockIdx.y*NUM_CLUSTERS_PER_BLOCK;
    int num_c = NUM_CLUSTERS_PER_BLOCK;

    // Handle boundary condition
    if(blockIdx.y == gridDim.y-1 && NUM_CLUSTERS % NUM_CLUSTERS_PER_BLOCK) {
        num_c = NUM_CLUSTERS % NUM_CLUSTERS_PER_BLOCK;
    }

    // Dimension index
    int d = blockIdx.x;
    int event_matrix_offset = my_num_events*d;

    __shared__ float numerators[NUM_THREADS_UPDATE*NUM_CLUSTERS_PER_BLOCK];

    int tid = threadIdx.x;

    // initialize numerators and denominators to 0
    for(int c = 0; c < num_c; c++) {
        numerators[c*NUM_THREADS_UPDATE+tid] = 0;
    }

    // Compute new membership value for each event
    // Add its contribution to the numerator and denominator for that thread
    for(int j = tid; j < my_num_events; j+=NUM_THREADS_UPDATE){
        eventValue = events[event_matrix_offset + j];
        numerators[0*NUM_THREADS_UPDATE+tid] += eventValue*memberships[(0+c_start)*my_num_events + j];
        numerators[1*NUM_THREADS_UPDATE+tid] += eventValue*memberships[(1+c_start)*my_num_events + j];
        numerators[2*NUM_THREADS_UPDATE+tid] += eventValue*memberships[(2+c_start)*my_num_events + j];
        numerators[3*NUM_THREADS_UPDATE+tid] += eventValue*memberships[(3+c_start)*my_num_events + j];
    }

    __syncthreads();

    for(int c = 0; c < num_c; c++) {
        numerators[c*NUM_THREADS_UPDATE+tid] = parallelSum(&numerators[NUM_THREADS_UPDATE*c],NUM_THREADS_UPDATE);
    }

    __syncthreads();

    if(tid == 0){
        for(int c = 0; c < num_c; c++) {
            // Set the new center for this block
            newClusters[(c+c_start)*NUM_DIMENSIONS + d] = numerators[c*NUM_THREADS_UPDATE];
        }
    }

}

__global__ void ComputeClusterSizes(float* memberships, float* sizes, int my_num_events) {
    __shared__ float partial_sums[512];

    partial_sums[threadIdx.x] = 0.0f;
    for(int i=threadIdx.x; i < my_num_events; i += 512) {
        partial_sums[threadIdx.x] += memberships[blockIdx.x*my_num_events+i];
    }

    __syncthreads();

    float sum = parallelSum(partial_sums,512);

    __syncthreads();

    if(threadIdx.x) {
        sizes[blockIdx.x] = sum;
    }

}


__global__ void ComputeDistanceMatrix(const float* clusters, const float* events, float* matrix, int my_num_events) {
    
    // copy the relavant center for this block into shared memory   
    __shared__ float center[NUM_DIMENSIONS];
    for(int j = threadIdx.x; j < NUM_DIMENSIONS; j+=NUM_THREADS_DISTANCE){
        center[j] = clusters[blockIdx.y*NUM_DIMENSIONS+j];
    }

    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < my_num_events) {
        matrix[blockIdx.y*my_num_events+i] = CalculateDistanceGPU(center,events,blockIdx.y,i,my_num_events);
    }
}

__global__ void ComputeMembershipMatrix(float* distances, float* memberships, int my_num_events) {
    float membershipValue;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // For each event
    if(i < my_num_events) {
        membershipValue = MembershipValueGPU(blockIdx.y, i, distances, my_num_events);
        #if FUZZINESS == 2 
            // This is much faster than the pow function
            membershipValue = membershipValue*membershipValue;
        #else
            membershipValue = __powf(membershipValue,FUZZINESS);
        #endif
        memberships[blockIdx.y*my_num_events+i] = membershipValue;
    }

}

__global__ void ComputeMembershipMatrixLinear(float* distances, int my_num_events) {
    float membershipValue;
    float denom = 0.0f;
    float dist;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // For each event
    if(i < my_num_events) {
        for(int c=0; c < NUM_CLUSTERS; c++) {
            dist = distances[c*my_num_events+i];
            #if FUZZINESS == 2
                dist = dist*dist;
            #else
                dist = __powf(dist,2.0f/(FUZZINESS-1.0f));
            #endif
            denom += 1.0f / dist;
        }
        
        for(int c=0; c < NUM_CLUSTERS; c++) {
            // not enough shared memory to store an array of distances
            // for each thread, so just recompute them like above
            dist = distances[c*my_num_events+i];
            #if FUZZINESS == 2
                dist = dist*dist;
                membershipValue = 1.0f/(dist*denom); // u
                membershipValue *= membershipValue; // u^p, p=2
            #else
                dist = __powf(dist,2.0f/(FUZZINESS-1.0f)); // u
                membershipValue = __powf(dist*denom,-FUZZINESS); // u^p
            #endif
            distances[c*my_num_events+i] = membershipValue;
        } 
    }
}

__global__ void ComputeNormalizedMembershipMatrix(float* distances, float* memberships, int my_num_events) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < my_num_events) {
        memberships[blockIdx.y*my_num_events+i] = MembershipValueGPU(blockIdx.y, i, distances,my_num_events);
    }
}
__global__ void ComputeNormalizedMembershipMatrixLinear(float* distances, int my_num_events) {
    float membershipValue;
    float denom = 0.0f;
    float dist;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // For each event
    if(i < my_num_events) {
        for(int c=0; c < NUM_CLUSTERS; c++) {
            dist = distances[c*my_num_events+i];
            #if FUZZINESS == 2
                dist = dist*dist;
            #else
                dist = __powf(dist,2.0f/(FUZZINESS-1.0f));
            #endif
            denom += 1.0f / dist;
        }
        
        for(int c=0; c < NUM_CLUSTERS; c++) {
            // not enough shared memory to store an array of distances
            // for each thread, so just recompute them like above
            dist = distances[c*my_num_events+i];
            #if FUZZINESS == 2
                dist = dist*dist;
                membershipValue = 1.0f/(dist*denom); // u
            #else
                dist = __powf(dist,2.0f/(FUZZINESS-1.0f)); // u
            #endif
            distances[c*my_num_events+i] = membershipValue;
        } 
    }
}

__device__ float MembershipValueGPU(int clusterIndex, int eventIndex, const float* distanceMatrix, int my_num_events){
	float myClustDist = 0;
    // Compute the distance from this event to the given cluster
    myClustDist = distanceMatrix[clusterIndex*my_num_events+eventIndex];
	
	float sum =0;
	float otherClustDist;
    // Compute the distance to all other clusters
    // Note: This is kind of inefficient, because the distance to every other cluster
    // is being re-computed by every other block
    // If each block handled a certain set of events rather than a cluster
    // we might be able to avoid this.
	for(int j = 0; j< NUM_CLUSTERS; j++){
        otherClustDist = distanceMatrix[j*my_num_events+eventIndex];
		#if FUZZINESS == 2
            sum += (myClustDist/otherClustDist)*(myClustDist/otherClustDist);
        #else
            sum += __powf((myClustDist/otherClustDist),(2.0f/(FUZZINESS-1.0f)));
        #endif
	}
	return 1/sum;
}

__device__ float CalculateDistanceGPU(const float* clusters, const float* events, int clusterIndex, int eventIndex, int my_num_events){

	float sum = 0;
	float tmp;
#if DISTANCE_MEASURE == 0
    #pragma unroll 1 // Prevent compiler from unrolling this loop too much, eats up too many registers
    for(int i = 0; i < NUM_DIMENSIONS; i++){
        tmp = events[i*my_num_events+eventIndex] - clusters[i];
        //tmp = events[i*my_num_events+eventIndex] - clusters[clusterIndex*NUM_DIMENSIONS +i];
        //tmp = events[eventIndex*NUM_DIMENSIONS + i] - clusters[clusterIndex*NUM_DIMENSIONS + i];
        sum += tmp*tmp;
    }
    sum = sqrt(sum+1e-30);
#endif
#if DISTANCE_MEASURE == 1
    #pragma unroll 1 // Prevent compiler from unrolling this loop too much, eats up too many registers
    for(int i = 0; i < NUM_DIMENSIONS; i++){
        tmp = events[i*my_num_events+eventIndex] - clusters[i];
        //tmp = events[i*my_num_events+eventIndex] - clusters[clusterIndex*NUM_DIMENSIONS +i];
        //tmp = events[eventIndex*NUM_DIMENSIONS + i] - clusters[clusterIndex*NUM_DIMENSIONS + i];
        sum += abs(tmp)+1e-30;
    }
#endif
#if DISTANCE_MEASURE == 2 
    #pragma unroll 1 // Prevent compiler from unrolling this loop too much, eats up too many registers
    for(int i = 0; i < NUM_DIMENSIONS; i++){
        tmp = abs(events[i*my_num_events + eventIndex] - clusters[i]);
        //tmp = abs(events[i*my_num_events + eventIndex] - clusters[clusterIndex*NUM_DIMENSIONS + i]);
        //tmp = abs(events[eventIndex*NUM_DIMENSIONS + i] - clusters[clusterIndex*NUM_DIMENSIONS + i]);
        if(tmp > sum)
            sum = tmp+1e-30;
    }
#endif

	return sum;
}


__device__ float CalculateQII(const float* events, float* distanceMatrix, int cluster_index_I, float* EI, float* numMem, int my_num_events){
	
	EI[threadIdx.x] = 0;
	numMem[threadIdx.x] = 0;
	
	for(int i = threadIdx.x; i < my_num_events; i+=Q_THREADS){
        float distance = distanceMatrix[cluster_index_I*my_num_events+i];
		float memVal = MembershipValueDist(distanceMatrix, cluster_index_I, i, distance, my_num_events);
		
		if(memVal > MEMBER_THRESH){
			EI[threadIdx.x] += pow(memVal, 2) * pow(distance, 2);
			numMem[threadIdx.x]++;
		}
	}
	
	__syncthreads();
	
	if(threadIdx.x == 0){
		for(int i = 1; i < Q_THREADS; i++){
			EI[0] += EI[i];
			numMem[0] += numMem[i];
		}
	}
	__syncthreads();

	return ((((float)K1) * numMem[0]) - (((float)K2) * EI[0]) - (((float)K3) * NUM_DIMENSIONS));

}


__device__ float CalculateQIJ(const float* events, float* distanceMatrix, int cluster_index_I, int cluster_index_J, float * EI, float * EJ, float *numMem, int my_num_events){
	
	
	EI[threadIdx.x] = 0;
	EJ[threadIdx.x] = 0;
	numMem[threadIdx.x] = 0;
	
	for(int i = threadIdx.x; i < my_num_events; i+=Q_THREADS){
        float distance = distanceMatrix[cluster_index_I*my_num_events+i];
        float memValI = MembershipValueDist(distanceMatrix, cluster_index_I, i, distance, my_num_events);
    
        if(memValI > MEMBER_THRESH){
            EI[threadIdx.x] += pow(memValI, 2) * pow(distance, 2);
            
        }
        
        distance = distanceMatrix[cluster_index_J*my_num_events+i];
        float memValJ = MembershipValueDist(distanceMatrix, cluster_index_J, i, distance, my_num_events);
        if(memValJ > MEMBER_THRESH){
            EJ[threadIdx.x] += pow(memValJ, 2) * pow(distance, 2);
        }
        if(memValI > MEMBER_THRESH && memValJ > MEMBER_THRESH){
            numMem[threadIdx.x]++;
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

__global__ void CalculateQMatrixGPUUpgrade(const float* events, const float* clusters, float* matrix, float* distanceMatrix, int start_row, int my_num_events){

	__shared__ float EI[Q_THREADS];
	__shared__ float EJ[Q_THREADS];
	__shared__ float numMem[Q_THREADS];

    int row = blockIdx.x + start_row;
    int col = blockIdx.y;

	if(row == col){
		matrix[row*NUM_CLUSTERS + col ] = CalculateQII(events, distanceMatrix, row, EI, numMem, my_num_events);
	}
	else{
		matrix[row*NUM_CLUSTERS + col] = CalculateQIJ(events, distanceMatrix, row, col, EI, EJ, numMem, my_num_events);
	}	
}

__device__ float MembershipValueDist(float* distanceMatrix, int clusterIndex, int eventIndex, float distance, int my_num_events){
	float sum =0.0f;
	float otherClustDist;
	for(int j = 0; j< NUM_CLUSTERS; j++){
        otherClustDist = distanceMatrix[j*my_num_events+eventIndex];
		sum += __powf((float)(distance/otherClustDist),(2.0f/(FUZZINESS-1.0f)));
	}
	return 1.0f/sum;
}
