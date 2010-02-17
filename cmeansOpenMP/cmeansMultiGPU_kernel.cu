#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
#include <cmeansMultiGPU.h>
#include <cmeansMultiGPUcu.h>
#include <float.h>

__global__ void UpdateClusterCentersGPU(const float* oldClusters, const float* events, float* newClusters, float* memberships, float* denoms, int start, int finish) {

    float membershipValue;//, denominator;

    int d = blockIdx.y;
    int event_matrix_offset = NUM_EVENTS*d;
    int membership_matrix_offset = NUM_EVENTS*blockIdx.x;

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
    for(int j = start+tid; j < finish; j+=NUM_THREADS_UPDATE){
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

__global__ void ComputeDistanceMatrix(const float* clusters, const float* events, float* matrix, int start, int stop) {
    
    // copy the relavant center for this block into shared memory   
    __shared__ float center[NUM_DIMENSIONS];
    for(int j = threadIdx.x; j < NUM_DIMENSIONS; j+=NUM_THREADS_DISTANCE){
        center[j] = clusters[blockIdx.y*NUM_DIMENSIONS+j];
    }

    __syncthreads();

    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if(i < stop) {
        matrix[blockIdx.y*NUM_EVENTS+i] = CalculateDistanceGPU(center,events,blockIdx.y,i);
    }
}

__global__ void ComputeMembershipMatrix(float* distances, float* memberships, int start, int stop) {
    float membershipValue;

    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    // For each event
    if(i < stop) {
        membershipValue = MembershipValueGPU(blockIdx.y, i, distances);
        #if FUZZINESS == 2 
            // This is much faster than the pow function
            membershipValue = membershipValue*membershipValue;
        #else
            membershipValue = __powf(membershipValue,FUZZINESS);
        #endif
        memberships[blockIdx.y*NUM_EVENTS+i] = membershipValue;
    }

}

__global__ void ComputeMembershipMatrixLinear(float* distances, int start, int stop) {
    float membershipValue;
    float denom = 0.0f;
    float dist;

    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    // For each event
    if(i < stop) {
        for(int c=0; c < NUM_CLUSTERS; c++) {
            dist = distances[c*NUM_EVENTS+i];
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
            dist = distances[c*NUM_EVENTS+i];
            #if FUZZINESS == 2
                dist = dist*dist;
                membershipValue = 1.0f/(dist*denom); // u
                membershipValue *= membershipValue; // u^p, p=2
            #else
                dist = __powf(dist,2.0f/(FUZZINESS-1.0f)); // u
                membershipValue = __powf(dist*denom,-FUZZINESS); // u^p
            #endif
            distances[c*NUM_EVENTS+i] = membershipValue;
        } 
    }
}

__global__ void ComputeNormalizedMembershipMatrix(float* distances, float* memberships, int start, int stop) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if(i < stop) {
        memberships[blockIdx.y*NUM_EVENTS+i] = MembershipValueGPU(blockIdx.y, i, distances);
    }
}
__global__ void ComputeNormalizedMembershipMatrixLinear(float* distances, int start, int stop) {
    float membershipValue;
    float denom = 0.0f;
    float dist;

    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    // For each event
    if(i < stop) {
        for(int c=0; c < NUM_CLUSTERS; c++) {
            dist = distances[c*NUM_EVENTS+i];
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
            dist = distances[c*NUM_EVENTS+i];
            #if FUZZINESS == 2
                dist = dist*dist;
                membershipValue = 1.0f/(dist*denom); // u
            #else
                dist = __powf(dist,2.0f/(FUZZINESS-1.0f)); // u
            #endif
            distances[c*NUM_EVENTS+i] = membershipValue;
        } 
    }
}

__device__ float MembershipValueGPU(int clusterIndex, int eventIndex, const float* distanceMatrix){
	float myClustDist = 0;
    // Compute the distance from this event to the given cluster
    myClustDist = distanceMatrix[clusterIndex*NUM_EVENTS+eventIndex];
	
	float sum =0;
	float otherClustDist;
    // Compute the distance to all other clusters
    // Note: This is kind of inefficient, because the distance to every other cluster
    // is being re-computed by every other block
    // If each block handled a certain set of events rather than a cluster
    // we might be able to avoid this.
	for(int j = 0; j< NUM_CLUSTERS; j++){
        otherClustDist = distanceMatrix[j*NUM_EVENTS+eventIndex];
		#if FUZZINESS == 2
            sum += (myClustDist/otherClustDist)*(myClustDist/otherClustDist);
        #else
            sum += __powf((myClustDist/otherClustDist),(2.0f/(FUZZINESS-1.0f)));
        #endif
	}
	return 1/sum;
}

__device__ float CalculateDistanceGPU(const float* clusters, const float* events, int clusterIndex, int eventIndex){

	float sum = 0;
	float tmp;
#if DISTANCE_MEASURE == 0
    #pragma unroll 1 // Prevent compiler from unrolling this loop too much, eats up too many registers
    for(int i = 0; i < NUM_DIMENSIONS; i++){
        tmp = events[i*NUM_EVENTS+eventIndex] - clusters[i];
        //tmp = events[i*NUM_EVENTS+eventIndex] - clusters[clusterIndex*NUM_DIMENSIONS +i];
        //tmp = events[eventIndex*NUM_DIMENSIONS + i] - clusters[clusterIndex*NUM_DIMENSIONS + i];
        sum += tmp*tmp;
    }
    sum = sqrt(sum+1e-30);
#endif
#if DISTANCE_MEASURE == 1
    #pragma unroll 1 // Prevent compiler from unrolling this loop too much, eats up too many registers
    for(int i = 0; i < NUM_DIMENSIONS; i++){
        tmp = events[i*NUM_EVENTS+eventIndex] - clusters[i];
        //tmp = events[i*NUM_EVENTS+eventIndex] - clusters[clusterIndex*NUM_DIMENSIONS +i];
        //tmp = events[eventIndex*NUM_DIMENSIONS + i] - clusters[clusterIndex*NUM_DIMENSIONS + i];
        sum += abs(tmp)+1e-30;
    }
#endif
#if DISTANCE_MEASURE == 2 
    #pragma unroll 1 // Prevent compiler from unrolling this loop too much, eats up too many registers
    for(int i = 0; i < NUM_DIMENSIONS; i++){
        tmp = abs(events[i*NUM_EVENTS + eventIndex] - clusters[i]);
        //tmp = abs(events[i*NUM_EVENTS + eventIndex] - clusters[clusterIndex*NUM_DIMENSIONS + i]);
        //tmp = abs(events[eventIndex*NUM_DIMENSIONS + i] - clusters[clusterIndex*NUM_DIMENSIONS + i]);
        if(tmp > sum)
            sum = tmp+1e-30;
    }
#endif

	return sum;
}


__device__ float CalculateQII(const float* events, float* distanceMatrix, int cluster_index_I, float* EI, float* numMem){
	
	EI[threadIdx.x] = 0;
	numMem[threadIdx.x] = 0;
	
	for(int i = threadIdx.x; i < NUM_EVENTS; i+=Q_THREADS){
        float distance = distanceMatrix[cluster_index_I*NUM_EVENTS+i];
		float memVal = MembershipValueDist(distanceMatrix, cluster_index_I, i, distance);
		
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


__device__ float CalculateQIJ(const float* events, float* distanceMatrix, int cluster_index_I, int cluster_index_J, float * EI, float * EJ, float *numMem){
	
	
	EI[threadIdx.x] = 0;
	EJ[threadIdx.x] = 0;
	numMem[threadIdx.x] = 0;
	
	for(int i = threadIdx.x; i < NUM_EVENTS; i+=Q_THREADS){
        float distance = distanceMatrix[cluster_index_I*NUM_EVENTS+i];
        float memValI = MembershipValueDist(distanceMatrix, cluster_index_I, i, distance);
    
        if(memValI > MEMBER_THRESH){
            EI[threadIdx.x] += pow(memValI, 2) * pow(distance, 2);
            
        }
        
        distance = distanceMatrix[cluster_index_J*NUM_EVENTS+i];
        float memValJ = MembershipValueDist(distanceMatrix, cluster_index_J, i, distance);
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

__global__ void CalculateQMatrixGPUUpgrade(const float* events, const float* clusters, float* matrix, float* distanceMatrix, int start_row){

	__shared__ float EI[Q_THREADS];
	__shared__ float EJ[Q_THREADS];
	__shared__ float numMem[Q_THREADS];

    int row = blockIdx.x + start_row;
    int col = blockIdx.y;

	if(row == col){
		matrix[row*NUM_CLUSTERS + col ] = CalculateQII(events, distanceMatrix, row, EI, numMem);
	}
	else{
		matrix[row*NUM_CLUSTERS + col] = CalculateQIJ(events, distanceMatrix, row, col, EI, EJ, numMem);
	}	
}

__device__ float MembershipValueDist(float* distanceMatrix, int clusterIndex, int eventIndex, float distance){
	float sum =0.0f;
	float otherClustDist;
	for(int j = 0; j< NUM_CLUSTERS; j++){
        otherClustDist = distanceMatrix[j*NUM_EVENTS+eventIndex];
		sum += __powf((float)(distance/otherClustDist),(2.0f/(FUZZINESS-1.0f)));
	}
	return 1.0f/sum;
}
