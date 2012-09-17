/*	

Copyright 2012 The Trustees of Indiana University.  All rights reserved.
CGL MapReduce Framework on GPUs and CPUs
Code Name: Panda 0.2
File: main.cu 
Time: 2012-07-01 
Developer: Hui Li (lihui@indiana.edu)

This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

*/

#include "Panda.h"
#include "Global.h"
#include <ctype.h>


//-----------------------------------------------------------------------
//usage: C-means datafile
//param: datafile 
//-----------------------------------------------------------------------


static float *GenPointsFloat(int numPt, int dim)
{
	float *matrix = (float*)malloc(sizeof(float)*numPt*dim);
	srand(time(0));
	for (int i = 0; i < numPt; i++)
		for (int j = 0; j < dim; j++)
			matrix[i*dim+j] = (float)((rand() % 100)/73.0);
	return matrix;
}//static float 

static float *GenInitCentersFloat(float* points, int numPt, int dim, int K)
{
	float* centers = (float*)malloc(sizeof(float)*K*dim);

	for (int i = 0; i < K; i++)
		for (int j = 0; j < dim; j++)
			centers[i*dim+j] = points[i*dim + j];
	return centers;
}//


int main(int argc, char** argv) 
{		

	if (argc != 6)
	{
		printf("Panda C-means\n");
		printf("usage: %s numPt Dimensions numClusters numMapperPerGPU maxIter\n", argv[0]);
		exit(-1);
	}//if

	//printf("start %s  %s  %s\n",argv[0],argv[1],argv[2]);
	int numPt = atoi(argv[1]);
	int dim = atoi(argv[2]);
	int K = atoi(argv[3]);
	int numMapper = atoi(argv[4]);
	int maxIter = atoi(argv[5]);
		
	DoLog("numPt:%d	dim:%d	K:%d	numMapper:%d	maxIter:%d",numPt,dim,K,numMapper,maxIter);
		
	float* h_points = GenPointsFloat(numPt, dim);
	float* h_cluster = GenInitCentersFloat(h_points, numPt, dim, K);
	
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
			
	pthread_t *no_threads = (pthread_t*)malloc(sizeof(pthread_t)*num_gpus);
	thread_info_t *thread_info = (thread_info_t*)malloc(sizeof(thread_info_t)*num_gpus);
		
	for (int i=0; i<num_gpus; i++){
		
		int tid = i;
		float* d_points = NULL;
		float* d_cluster = NULL;
		int* d_change = NULL;
		int* d_clusterId = NULL;
		
		float* d_tempClusters = NULL;
		float* d_tempDenominators = NULL;

		checkCudaErrors(cudaSetDevice(tid));
				
		checkCudaErrors(cudaMalloc((void**)&d_points, numPt*dim*sizeof(int)));
		checkCudaErrors(cudaMemcpy(d_points, h_points, numPt*dim*sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void**)&d_clusterId, numPt*sizeof(int)));
		checkCudaErrors(cudaMemset(d_clusterId, 0, numPt*sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_cluster, K*dim*sizeof(int)));
		checkCudaErrors(cudaMemcpy(d_cluster, h_cluster, K*dim*sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void**)&d_change, sizeof(int)));
		checkCudaErrors(cudaMemset(d_change, 0, sizeof(int)));
		
		checkCudaErrors(cudaMalloc((void**)&d_tempClusters,K*dim*numMapper*sizeof(float)));
		checkCudaErrors(cudaMemset(d_tempClusters, 0, sizeof(float)*K*dim*numMapper));
		checkCudaErrors(cudaMalloc((void**)&d_tempDenominators,numMapper * K * sizeof(float)));
		
		checkCudaErrors(cudaMemset(d_tempDenominators, 0, sizeof(float)*K*numMapper));
		
		thread_info[i].tid = i;
		thread_info[i].num_gpus = num_gpus;
		thread_info[i].device_type = GPU_ACC;
		
		cudaDeviceProp gpu_dev;
		cudaGetDeviceProperties(&gpu_dev, i);
		DoLog("Configure Device ID:%d: Device Name:%s", i, gpu_dev.name);
		thread_info[i].device_name = gpu_dev.name;
		
		gpu_context *d_g_state = GetDGlobalState();
		thread_info[i].d_g_state = d_g_state;
	
		
		KM_VAL_T val;
		val.ptrPoints = (int *)d_points;
		val.ptrClusters = (int *)d_cluster;
		val.d_Points = d_points;
		val.d_Clusters = d_cluster;
		val.ptrChange = d_change;
		
		KM_KEY_T key;
		key.dim = dim;
		key.K = K;
		key.ptrClusterId = d_clusterId;
		
		int numPtPerGPU = numPt/num_gpus;
		int start = i*numPtPerGPU;
		int end = start+numPtPerGPU;
		if (i==num_gpus-1)
			end = numPt;
		
		int numPtPerMap = (end-start)/numMapper;
		int start_i,end_i;
		start_i = start;
		for (int j = 0; j < numMapper; j++)
		{	
			end_i = start_i + numPtPerMap;
			if (i<(end-start)%numMapper)
				end_i++;
			
			//DoLog("start_i:%d, start_j:%d",start_i,end_i);
			key.point_id = start_i;
			key.start = start_i;
			key.end = end_i;
			key.i = i*numMapper+j;

			val.d_Points = d_points;
			val.d_tempDenominators = d_tempDenominators;
			val.d_tempClusters = d_tempClusters;

			AddMapInputRecord2(d_g_state, &key, &val, sizeof(KM_KEY_T), sizeof(KM_VAL_T));
			start_i = end_i;
		}//for
	}//for

	int iter = 0;
	while (iter<maxIter)
	{

		for (int i=0; i<num_gpus; i++){
			if (pthread_create(&(no_threads[i]),NULL,Panda_Map,(char *)&(thread_info[i]))!=0) 
				perror("Thread creation failed!\n");
		}//for num_gpus

		for (int i=0; i<num_gpus; i++){
			void *exitstat;
			if (pthread_join(no_threads[i],&exitstat)!=0) perror("joining failed");
		}//for

		int gpu_id;
		cudaGetDevice(&gpu_id);
		DoLog("current gpu_id:%d",gpu_id);

		if(gpu_id !=(num_gpus-1)){
			checkCudaErrors(cudaSetDevice(num_gpus-1));
			DoLog("changing GPU context to device:%d",num_gpus-1);
		}//if
		
			if (num_gpus == 1){
			gpu_context dummy_d_g_state;
			dummy_d_g_state.h_sorted_keys_shared_buff = NULL;
			dummy_d_g_state.h_sorted_vals_shared_buff = NULL;
			dummy_d_g_state.h_intermediate_keyval_pos_arr = NULL;
			dummy_d_g_state.d_sorted_keyvals_arr_len = 0;
			Panda_Shuffle_Merge(&dummy_d_g_state, thread_info[0].d_g_state);
			}//if

	
		for (int i=1; i<num_gpus; i++){
			/*
			if(i==0){
				gpu_context dummy_d_g_state;
				dummy_d_g_state.h_sorted_keys_shared_buff = NULL;
				dummy_d_g_state.h_sorted_vals_shared_buff = NULL;
				dummy_d_g_state.h_intermediate_keyval_pos_arr = NULL;
				dummy_d_g_state.d_sorted_keyvals_arr_len = 0;
				Panda_Shuffle_Merge(&dummy_d_g_state, thread_info[0].d_g_state);
			}else{
			*/
			Panda_Shuffle_Merge(thread_info[i-1].d_g_state, thread_info[i].d_g_state);
			//}//else
		}//for

		cudaThreadSynchronize();
		Panda_Reduce(&thread_info[num_gpus-1]);
		
		iter++;
		cudaThreadSynchronize();

	}//while iterations
	return 0;
}//		