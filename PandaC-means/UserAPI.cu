/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: reduce.cu 
	Time: 2012-07-01 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */

#ifndef __REDUCE_CU__
#define __REDUCE_CU__

#include "Panda.h"
#include "UserAPI.h"

//invoke cmeans_cpu_map_cpp compiled with g++
void cpu_map2(void *key, void *val, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx){
	
	cmeans_cpu_map_cpp(key, val, keySize, valSize);
	CPUEmitMapOutput(key, val, sizeof(KM_KEY_T), sizeof(KM_VAL_T), d_g_state, map_task_idx);

}


void cpu_map(void *key, void *val, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx){

	KM_KEY_T* pKey = (KM_KEY_T*)key;
	KM_VAL_T* pVal = (KM_VAL_T*)val;
	
	int dim = pKey->dim;
	//int dim_4;
	int K = pKey->K;
	int start = pKey->start;
	int end = pKey->end;
	int index = pKey->local_map_id;
	//TODO there could be problem here when running C-means with more than one GPU
	//index = 0;

	float *point	= (float*)(pVal->d_Points);
	float *cluster	= (float*)(pVal->d_Clusters);

	float * tempClusters = pVal->d_tempClusters+index*dim*K;
	float * tempDenominators = pVal->d_tempDenominators+index*K;

	float denominator = 0.0f;
	float membershipValue = 0.0f;

	float *distances = (float *)malloc(sizeof(float)*K);
	float *numerator = (float *)malloc(sizeof(float)*K);
	
	for(int i=0; i<K; i++){
		distances[i]=0.0f;
		numerator[i]=0.0f;
	}//for

	//printf("map_task_id 0:%d thread_id:%d\n",map_task_idx,THREAD_ID);
	for (int i=start; i<end; i++){
		float *curPoint = (float*)(pVal->d_Points + i*dim);
		for (int k = 0; k < K; ++k)
		{
			float* curCluster = (float*)(pVal->d_Clusters + k*dim);
			distances[k] = 0.0;
			//printf("dim:%d\n",dim);
			//dim_4 = dim;
			float delta = 0.0;	
			
			for (int j = 0; j < dim; ++j)
			{
				delta = curPoint[j]-curCluster[j];
				distances[k] += (delta*delta);
			}//for
			
			numerator[k] = powf(distances[k],2.0f/(2.0-1.0))+1e-30;
			denominator  = denominator + 1.0f/(numerator[k]+1e-30);
		}//for

		for (int k = 0; k < K; ++k)
		{
			membershipValue = 1.0f/powf(numerator[k]*denominator,(float)2.0);
			for(int d =0; d<dim; d++){
				//float pt = curePoint[d].x;
				tempClusters[k*dim+d] += (curPoint[d])*membershipValue;
				
			}
			tempDenominators[k]+= membershipValue;
		}//for 
	}//for
	//printf("map_task_id 1:%d\n",map_task_idx);
	
	free(distances);
	free(numerator);
	
	//TODO
	pKey->local_map_id = 0;
	pKey->end = 0;
	pKey->start = 0;
	pKey->global_map_id = 0;
	
	CPUEmitMapOutput(key, val, sizeof(KM_KEY_T), sizeof(KM_VAL_T), d_g_state, map_task_idx);

}//void

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
        }//for
    }//if

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
}//ComputeDistanceMatrix

__global__ void ComputeMembershipMatrixLinear(float* distances, int my_num_events) {
    float membershipValue;
    float denom = 0.0f;
    float dist;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // For each event
    if(i < my_num_events) {
        for(int c=0; c < NUM_CLUSTERS; c++) {
            dist = distances[c*my_num_events+i];
            #if FUZZINESS_SQUARE
                dist = dist*dist;
            #else
                dist = __powf(dist,2.0f/(FUZZINESS-1.0f))+1e-30;
            #endif
            denom += 1.0f / dist; // what if dist is really big?
        }

        for(int c=0; c < NUM_CLUSTERS; c++) {
            // not enough shared memory to store an array of distances
            // for each thread, so just recompute them like above
            dist = distances[c*my_num_events+i];
            #if FUZZINESS_SQUARE
                dist = dist*dist;
                membershipValue = 1.0f/(dist*denom); // u
                membershipValue *= membershipValue; // u^p, p=2
            #else
                dist = __powf(dist,2.0f/(FUZZINESS-1.0f))+1e-30;
                membershipValue = 1.0f/(dist*denom); // u
                membershipValue = __powf(membershipValue,FUZZINESS); // u^p
            #endif
            distances[c*my_num_events+i] = membershipValue;
        }
    }
}//ComputeMembershipMatrixLinear


void gpu_card_map(void *key, void *val, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){

	KM_KEY_T* pKey = (KM_KEY_T*)key;
	KM_VAL_T* pVal = (KM_VAL_T*)val;
	
	int dim = pKey->dim;
	int K = pKey->K;
	int start = pKey->start;
	int end = pKey->end;
	int index = pKey->local_map_id;
	int tid = 0;

	int my_num_events = end-start;
	int events_per_gpu = my_num_events;

    // printf("GPU %d, Starting Event: %d, Ending Event: %d, My Num Events: %d\n",tid,start,finish,my_num_events);

		float*myEvents = pVal->d_Points;
		float*myClusters=pVal->d_Clusters;

		float* transposedEvents = (float*)malloc(sizeof(float)*NUM_EVENTS*NUM_DIMENSIONS);
		for(int i=0; i<NUM_EVENTS; i++) {
        for(int j=0; j<NUM_DIMENSIONS; j++) {
            transposedEvents[j*NUM_EVENTS+i] = myEvents[i*NUM_DIMENSIONS+j];
        }
		}


        float* d_distanceMatrix;
        checkCudaErrors(cudaMalloc((void**)&d_distanceMatrix, sizeof(float)*my_num_events*NUM_CLUSTERS));

        #if !LINEAR
            float* d_memberships;
            checkCudaErrors(cudaMalloc((void**)&d_memberships, sizeof(float)*my_num_events*NUM_CLUSTERS));
        #endif

        float* d_E;
        checkCudaErrors(cudaMalloc((void**)&d_E, sizeof(float)*my_num_events*NUM_DIMENSIONS));
        float* d_C;
        checkCudaErrors(cudaMalloc((void**)&d_C, sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS));
        float* d_nC;
        checkCudaErrors(cudaMalloc((void**)&d_nC, sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS));
        float* d_denoms;
        checkCudaErrors(cudaMalloc((void**)&d_denoms, sizeof(float)*NUM_CLUSTERS));

		int size = sizeof(float)*NUM_DIMENSIONS*my_num_events;
		float* temp_fcs_data = (float*) malloc(size);
        for(int d=0; d < NUM_DIMENSIONS; d++) {
		memcpy(&temp_fcs_data[d*my_num_events],&transposedEvents[d*NUM_EVENTS + tid*events_per_gpu],sizeof(float)*my_num_events);
        }//for
        checkCudaErrors(cudaMemcpy( d_E, temp_fcs_data, size,cudaMemcpyHostToDevice) );
        cudaThreadSynchronize();
        free(temp_fcs_data);

        size = sizeof(float)*NUM_DIMENSIONS*NUM_CLUSTERS;
        checkCudaErrors(cudaMemcpy(d_C, myClusters, size, cudaMemcpyHostToDevice));

        printf("Starting C-means\n");
        int iterations = 0;

        int num_blocks_distance = my_num_events / NUM_THREADS_DISTANCE;
        if(my_num_events % NUM_THREADS_DISTANCE) {
            num_blocks_distance++;
        }
        int num_blocks_membership = my_num_events / NUM_THREADS_MEMBERSHIP;
        if(my_num_events % NUM_THREADS_DISTANCE) {
            num_blocks_membership++;
        }
        int num_blocks_update = NUM_CLUSTERS / NUM_CLUSTERS_PER_BLOCK;
        if(NUM_CLUSTERS % NUM_CLUSTERS_PER_BLOCK) {
            num_blocks_update++;
        }

        //size = sizeof(float)*NUM_DIMENSIONS*my_num_events;

    	

	ComputeDistanceMatrix<<< dim3(num_blocks_distance,NUM_CLUSTERS), NUM_THREADS_DISTANCE  >>>(d_C, d_E, d_distanceMatrix, my_num_events);
	ComputeMembershipMatrixLinear<<< num_blocks_membership, NUM_THREADS_MEMBERSHIP  >>>(d_distanceMatrix, my_num_events);
	UpdateClusterCentersGPU3<<< dim3(NUM_DIMENSIONS,num_blocks_update), NUM_THREADS_UPDATE >>>(d_C, d_E, d_nC, d_distanceMatrix, my_num_events);
	ComputeClusterSizes<<< NUM_CLUSTERS, 512 >>>( d_distanceMatrix, d_denoms, my_num_events);

	//Emit
	//GPUEmitMapOutput(key, val, sizeof(KM_KEY_T), sizeof(KM_VAL_T), d_g_state, map_task_idx);
    //cudaMemcpy(tempClusters[tid], d_nC, sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS, cudaMemcpyDeviceToHost);
    //cudaMemcpy(tempDenominators[tid], d_denoms, sizeof(float)*NUM_CLUSTERS, cudaMemcpyDeviceToHost);


}

#if 0
void gpu_card_map2(void *key, void *val, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){


	float* d_points	=	NULL;
	float* d_cluster =	NULL;
		//int* d_change	=	NULL;
	int* d_clusterId =	NULL;
		
	float* d_tempClusters = NULL;
	float* d_tempDenominators = NULL;
	float* d_distanceMatrix = NULL;

	if (tid<numgpus)
			checkCudaErrors(cudaSetDevice(tid));

	checkCudaErrors(cudaMalloc((void**)&d_distanceMatrix, sizeof(float)*numPt*NUM_CLUSTERS));
	checkCudaErrors(cudaMalloc((void**)&d_points, numPt*dim*sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_points, h_points, numPt*dim*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_clusterId, numPt*sizeof(int)));
	checkCudaErrors(cudaMemset(d_clusterId, 0, numPt*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_cluster, K*dim*sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_cluster, h_cluster, K*dim*sizeof(int), cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMalloc((void**)&d_change, sizeof(int)));
		//checkCudaErrors(cudaMemset(d_change, 0, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_tempClusters,K*dim*numMapperGPU*sizeof(float)));
	checkCudaErrors(cudaMemset(d_tempClusters, 0, sizeof(float)*K*dim*numMapperGPU));
	checkCudaErrors(cudaMalloc((void**)&d_tempDenominators,numMapperGPU * K * sizeof(float)));
		
	checkCudaErrors(cudaMemset(d_tempDenominators, 0, sizeof(float)*K*numMapperGPU));
	thread_info[global_dev_id].tid = global_dev_id;
		//thread_info[dev_id].num_gpu_core_groups = num_gpu_core_groups;
		thread_info[global_dev_id].device_type = GPU_CARD_ACC;
		
		cudaDeviceProp gpu_dev;
		cudaGetDeviceProperties(&gpu_dev, dev_id);
		ShowLog("Configure GPU_CARD_ACC Device ID:%d: Device Name:%s", dev_id, gpu_dev.name);
		thread_info[dev_id].device_name = gpu_dev.name;



	KM_KEY_T* pKey = (KM_KEY_T*)key;
	KM_VAL_T* pVal = (KM_VAL_T*)val;
	
	int dim = pKey->dim;
	int K = pKey->K;
	int start = pKey->start;
	int end = pKey->end;
	int index = pKey->local_map_id;

	int my_num_events = end-start;

	float *d_E =(float*)(pVal->d_Points);
	float *d_C = (float*)(pVal->d_Clusters);
	float *d_distanceMatrix = (float*)(pVal->d_distanceMatrix);
	
	float* d_denoms;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_denoms, sizeof(float)*NUM_CLUSTERS));


	float* d_nC;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nC, sizeof(float)*NUM_CLUSTERS*dim));


	float * tempClusters = pVal->d_tempClusters+index*dim*K;
	float * tempDenominators = pVal->d_tempDenominators+index*K;

	float denominator = 0.0f;
	float membershipValue = 0.0f;

	float *distances = (float *)malloc(sizeof(float)*K);
	float *numerator = (float *)malloc(sizeof(float)*K);
	
	for(int i=0; i<K; i++){
		distances[i]=0.0f;
		numerator[i]=0.0f;
	}//for

		int num_blocks_distance = my_num_events / NUM_THREADS_DISTANCE;
        if(my_num_events % NUM_THREADS_DISTANCE) {
            num_blocks_distance++;
        }
        int num_blocks_membership = my_num_events / NUM_THREADS_MEMBERSHIP;
        if(my_num_events % NUM_THREADS_DISTANCE) {
            num_blocks_membership++;
        }
        int num_blocks_update = NUM_CLUSTERS / NUM_CLUSTERS_PER_BLOCK;
        if(NUM_CLUSTERS % NUM_CLUSTERS_PER_BLOCK) {
            num_blocks_update++;
        }


	ComputeDistanceMatrix<<< dim3(num_blocks_distance,NUM_CLUSTERS), NUM_THREADS_DISTANCE  >>>(d_C, d_E, d_distanceMatrix, my_num_events);
	ComputeMembershipMatrixLinear<<< num_blocks_membership, NUM_THREADS_MEMBERSHIP  >>>(d_distanceMatrix, my_num_events);
	UpdateClusterCentersGPU3<<< dim3(NUM_DIMENSIONS,num_blocks_update), NUM_THREADS_UPDATE >>>(d_C, d_E, d_nC, d_distanceMatrix, my_num_events);
	ComputeClusterSizes<<< NUM_CLUSTERS, 512 >>>( d_distanceMatrix, d_denoms, my_num_events);

	//Emit
	//GPUEmitMapOutput(key, val, sizeof(KM_KEY_T), sizeof(KM_VAL_T), d_g_state, map_task_idx);
    //cudaMemcpy(tempClusters[tid], d_nC, sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS, cudaMemcpyDeviceToHost);
    //cudaMemcpy(tempDenominators[tid], d_denoms, sizeof(float)*NUM_CLUSTERS, cudaMemcpyDeviceToHost);

}

#endif


__device__ void gpu_core_map(void *key, void *val, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){

	KM_KEY_T* pKey = (KM_KEY_T*)key;
	KM_VAL_T* pVal = (KM_VAL_T*)val;
	
	int dim = pKey->dim;
	int dim_4;
	int K = pKey->K;
	int start = pKey->start;
	int end = pKey->end;
	int index = pKey->local_map_id;
	//TODO there could be problem here when running C-means with more than one GPU
	//index = 0;

	float4 *point =(float4*)(pVal->d_Points);
	float* cluster = (float*)(pVal->d_Clusters);

	float * tempClusters = pVal->d_tempClusters+index*dim*K;
	float * tempDenominators = pVal->d_tempDenominators+index*K;

	float denominator = 0.0f;
	float membershipValue = 0.0f;

	float *distances = (float *)malloc(sizeof(float)*K);
	float *numerator = (float *)malloc(sizeof(float)*K);
	
	
	for(int i=0; i<K; i++){
		distances[i]=0.0f;
		numerator[i]=0.0f;
	}//for

	

	for (int i=start; i<end; i++){
		float4* curPoint = (float4*)(pVal->d_Points + i*dim);
		for (int k = 0; k < K; ++k)
		{
			float4* curCluster = (float4*)(pVal->d_Clusters + k*dim);
			distances[k] = 0.0;
			//printf("dim:%d\n",dim);
			dim_4 = dim/4;
			float delta = 0.0;	
			
			for (int j = 0; j < dim_4; ++j)
			{
				float4 pt = curPoint[j];
				float4 cl = curCluster[j];

				delta = pt.x-cl.x;
				distances[k] += (delta*delta);
				delta = pt.y-cl.y;
				distances[k] += (delta*delta);
				delta = pt.z-cl.z;
				distances[k] += (delta*delta);
				delta = pt.w-cl.w;
				distances[k] += (delta*delta);

			}//for
				
			int remainder = dim & 0x00000003;
			float* rPoint = (float*)(curPoint+dim_4);
			float* rCluster = (float*)(curCluster+dim_4);
			
			for (int j = 0; j < remainder; j++)
			{
				float pt = rPoint[j];
				float cl = rCluster[j];
				delta = pt - cl;
				distances[k] += (delta*delta);				
			}			
			numerator[k] = powf(distances[k],2.0f/(2.0-1.0))+1e-10;
			denominator  = denominator + 1.0f/(numerator[k]+1e-10);
		}//for

		for (int k = 0; k < K; ++k)
		{
			membershipValue = 1.0f/powf(numerator[k]*denominator,(float)2.0);
			for(int d =0;d<dim_4;d++){
				//float pt = curePoint[d].x;
				tempClusters[k*dim+d] += (curPoint[d].x)*membershipValue;
				tempClusters[k*dim+d] += (curPoint[d].y)*membershipValue;
				tempClusters[k*dim+d] += (curPoint[d].z)*membershipValue;
				tempClusters[k*dim+d] += (curPoint[d].w)*membershipValue;
			}
			tempDenominators[k]+= membershipValue;
		}//for 
	}//for
	//printf("map_task_id 1:%d\n",map_task_idx);
	
	free(distances);
	free(numerator);
	
	//TODO
	pKey->local_map_id = 0;
	pKey->end = 0;
	pKey->start = 0;
	pKey->global_map_id = 0;
	
	GPUEmitMapOutput(key, val, sizeof(KM_KEY_T), sizeof(KM_VAL_T), d_g_state, map_task_idx);
	
}//map2





__device__ int gpu_compare(const void *key_a, int len_a, const void *key_b, int len_b)
{
	//KM_KEY_T *ka = (KM_KEY_T*)key_a;
	//KM_KEY_T *kb = (KM_KEY_T*)key_b;

	return 0;

	/*
	if (ka->i > kb->i)
		return 1;

	if (ka->i > kb->i)
		return -1;

	if (ka->i == kb->i)
		return 0;
	*/
}


int cpu_compare(const void *key_a, int len_a, const void *key_b, int len_b)
{
	//KM_KEY_T *ka = (KM_KEY_T*)key_a;
	//KM_KEY_T *kb = (KM_KEY_T*)key_b;

	return 0;

	/*
	if (ka->i > kb->i)
		return 1;

	if (ka->i > kb->i)
		return -1;

	if (ka->i == kb->i)
		return 0;
		*/

}


void cpu_reduce(void *key, val_t* vals, int keySize, int valCount, cpu_context* d_g_state){
	cmeans_cpu_reduce_cpp(key,  vals, keySize, valCount);
	CPUEmitReduceOutput(key,vals,sizeof(KM_KEY_T), sizeof(KM_VAL_T), d_g_state);
}

//-------------------------------------------------------------------------
//Reduce Function in this application
//-------------------------------------------------------------------------

void cpu_reduce2(void *key, val_t* vals, int keySize, int valCount, cpu_context* d_g_state)
{

		KM_KEY_T* pKey = (KM_KEY_T*)key;
        int dim = pKey->dim;
        int K = pKey->K;



        float* myClusters = (float*) malloc(sizeof(float)*dim*K);
        float* myDenominators = (float*) malloc(sizeof(float)*K);
        memset(myClusters,0,sizeof(float)*dim*K);
        memset(myDenominators,0,sizeof(float)*K);

        float *tempClusters = NULL;
        float *tempDenominators = NULL;
		

        for (int i = 0; i < valCount; i++)
        {
                int index = pKey->local_map_id;


				KM_VAL_T* pVal = (KM_VAL_T*)(vals[i].val);
                tempClusters = pVal->d_tempClusters + index*K*dim;
                tempDenominators = pVal->d_tempDenominators+ index*K;
                for (int k = 0; k< K; k++){
                        for (int j = 0; j< dim; j++)
                                myClusters[k*dim+j] += tempClusters[k*dim+j];
                        myDenominators[k] += tempDenominators[k];
                }//for
        }//end for


        for (int k = 0; k< K; k++){
			for (int i = 0; i < dim; i++){
						//printf("K:%d dim:%d myDenominators[i]:%f",K,dim,myDenominators[i]);
                        myClusters[i] /= ((float)myDenominators[i]+0.0001);
						//printf("%f ",myClusters[i]);
			}//for
			//printf("\n");
        }//for

		
		free(myClusters);
		free(myDenominators);

		CPUEmitReduceOutput(key,vals,sizeof(KM_KEY_T), sizeof(KM_VAL_T), d_g_state);

}

__device__ void gpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, gpu_context *d_g_state, int map_task_idx){
		
		
}//reduce2

void cpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, cpu_context *d_g_state, int map_task_idx){
		
		
}//reduce2

__device__ void gpu_reduce(void *key, val_t* vals, int keySize, int valCount, gpu_context d_g_state)
{
		//printf("valCount:%d\n",valCount);
		KM_KEY_T* pKey = (KM_KEY_T*)key;
        //KM_VAL_T* pVal = (KM_VAL_T*)vals;
        int dim = pKey->dim;
        int K = pKey->K;
				
        float* myClusters = (float*) malloc(sizeof(float)*dim*K);
        float* myDenominators = (float*) malloc(sizeof(float)*K);
        memset(myClusters,0,sizeof(float)*dim*K);
        memset(myDenominators,0,sizeof(float)*K);

        float *tempClusters = NULL;
        float *tempDenominators = NULL;
        for (int i = 0; i < valCount; i++)
        {
                int index = pKey->local_map_id;
				KM_VAL_T* pVal = (KM_VAL_T*)(vals[i].val);
                tempClusters = pVal->d_tempClusters + index*K*dim;
                tempDenominators = pVal->d_tempDenominators+ index*K;
                for (int k = 0; k< K; k++){
                        for (int j = 0; j< dim; j++)
                                myClusters[k*dim+j] += tempClusters[k*dim+j];
                        myDenominators[k] += tempDenominators[k];
                }//for
        }//end for

        for (int k = 0; k< K; k++){
			for (int i = 0; i < dim; i++){
                        myClusters[i] /= ((float)myDenominators[i]+0.001);
						//printf("%f ",myClusters[i]);
			}//for
			//printf("\n");
        }//for

		//printf("TID reduce2:%d\n",TID);
		GPUEmitReduceOuput(key,vals,sizeof(KM_KEY_T), sizeof(KM_VAL_T), &d_g_state);
		
		free(myClusters);
		free(myDenominators);
				
}//reduce2

#endif //__REDUCE_CU__
