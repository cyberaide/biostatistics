/*	
Copyright 2012 The Trustees of Indiana University.  All rights reserved.
CGL MapReduce Framework on GPUs and CPUs

Code Name: Panda 

File: PandaSort.cu 
First Version:		2012-07-01 V0.1
Current Version:	2012-09-01 V0.3	
Last Updates:		2012-09-16

Developer: Hui Li (lihui@indiana.edu)

This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//includes CUDA
#include <cuda_runtime.h>

#ifndef _PANDASORT_CU_
#define _PANDASORT_CU_

#include "Panda.h"
#include "UserAPI.h"

void initialize(cmp_type_t *d_data, int rLen, cmp_type_t value)
{
	cudaThreadSynchronize();
}



__global__ void copyDataFromDevice2Host1(gpu_context d_g_state)
{	

	int num_records_per_thread = (d_g_state.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx + num_records_per_thread*STRIDE;

	if(thread_end_idx>d_g_state.num_input_record)
		thread_end_idx = d_g_state.num_input_record;

	if (thread_start_idx >= thread_end_idx)
		return;

	int begin=0;
	int end=0;
	for (int i=0; i<thread_start_idx; i++){
		begin += d_g_state.d_intermediate_keyval_total_count[i];
	}//for
	end = begin + d_g_state.d_intermediate_keyval_total_count[thread_start_idx];

	int start_idx = 0;
	for(int i=begin;i<end;i++){
		keyval_t * p1 = &(d_g_state.d_intermediate_keyval_arr[i]);
		keyval_pos_t * p2 = NULL;
		keyval_arr_t *kv_arr_p = d_g_state.d_intermediate_keyval_arr_arr_p[thread_start_idx];

		char *shared_buff = (char *)(kv_arr_p->shared_buff);
		int shared_arr_len = *kv_arr_p->shared_arr_len;
		int shared_buff_len = *kv_arr_p->shared_buff_len;


		for (int idx = start_idx; idx<(shared_arr_len); idx++){
			p2 = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(shared_arr_len - idx ));

			if ( p2->next_idx != -2 ){
				continue;
			}//if
			start_idx = idx+1;

			p1->keySize = p2->keySize;
			p1->valSize = p2->valSize;
			p1->task_idx = i;
			p2->task_idx = i;
			break;
		}//for
	}//for

}	

__global__ void copyDataFromDevice2Host3(gpu_context d_g_state)
{	

	int num_records_per_thread = (d_g_state.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx+num_records_per_thread*STRIDE;

	if(thread_end_idx>d_g_state.num_input_record)
		thread_end_idx = d_g_state.num_input_record;

	if (thread_start_idx >= thread_end_idx)
		return;

	int begin, end, val_pos, key_pos;
	char *val_p,*key_p;

	begin=0;
	end=0;

	for (int i=0; i<thread_start_idx; i++) 		
		begin = begin + d_g_state.d_intermediate_keyval_total_count[i];

	end = begin + d_g_state.d_intermediate_keyval_total_count[thread_start_idx];

	keyval_arr_t *kv_arr_p = d_g_state.d_intermediate_keyval_arr_arr_p[thread_start_idx];
	char *shared_buff = (char *)(kv_arr_p->shared_buff);
	int shared_arr_len = *kv_arr_p->shared_arr_len;
	int shared_buff_len = *kv_arr_p->shared_buff_len;

	for(int i=begin;i<end;i++){

		val_pos = d_g_state.d_intermediate_keyval_pos_arr[i].valPos;
		key_pos = d_g_state.d_intermediate_keyval_pos_arr[i].keyPos;

		val_p = (char*)(d_g_state.d_intermediate_vals_shared_buff)+val_pos;
		key_p = (char*)(d_g_state.d_intermediate_keys_shared_buff)+key_pos;
		keyval_pos_t * p2 = NULL;//&(d_g_state.d_intermediate_keyval_arr_arr_p[map_task_idx]->arr[i-begin]);

		int start_idx = 0;
		for(int idx = start_idx; idx<(shared_arr_len); idx++){

			p2 = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(shared_arr_len - idx ));
			//TODO reverse inner loop to outside loop
			if (p2->next_idx != -2)		continue;
			if (p2->task_idx != i) 		continue;
			start_idx = idx+1;
			memcpy(key_p, shared_buff + p2->keyPos, p2->keySize);
			memcpy(val_p, shared_buff + p2->valPos, p2->valSize);

			break;	
		}//for
		if (p2->task_idx != i)	ShowWarn("copyDataFromDevice2Host3 p2->task_idx %d != i %d\n",p2->task_idx, i);
	}//for

}//__global__	

__global__ void copyDataFromDevice2Host2(gpu_context d_g_state)
{	

	int num_records_per_thread = (d_g_state.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx+num_records_per_thread*STRIDE;

	if(thread_end_idx>d_g_state.num_input_record)
		thread_end_idx = d_g_state.num_input_record;

	if (thread_start_idx >= thread_end_idx)
		return;

	int begin, end;
	begin=end=0;
	for (int i=0; i<thread_start_idx; i++) 	
		begin = begin + d_g_state.d_intermediate_keyval_total_count[i];
	end = begin + d_g_state.d_intermediate_keyval_total_count[thread_start_idx];

	keyval_arr_t *kv_arr_p = d_g_state.d_intermediate_keyval_arr_arr_p[thread_start_idx];
	char *shared_buff = (char *)(kv_arr_p->shared_buff);
	int shared_arr_len = *kv_arr_p->shared_arr_len;
	int shared_buff_len = *kv_arr_p->shared_buff_len;

	int val_pos, key_pos;
	char *val_p,*key_p;
	int counter = 0;
	for(int local_idx = 0; local_idx<(shared_arr_len); local_idx++){

		keyval_pos_t *p2 = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(shared_arr_len - local_idx ));
		if (p2->next_idx != -2)		continue;
		//	if (p2->task_idx != i) 		continue;
		int global_idx = p2->task_idx;
		val_pos = d_g_state.d_intermediate_keyval_pos_arr[global_idx].valPos;
		key_pos = d_g_state.d_intermediate_keyval_pos_arr[global_idx].keyPos;

		val_p = (char*)(d_g_state.d_intermediate_vals_shared_buff)+val_pos;
		key_p = (char*)(d_g_state.d_intermediate_keys_shared_buff)+key_pos;

		memcpy(key_p, shared_buff + p2->keyPos, p2->keySize);
		memcpy(val_p, shared_buff + p2->valPos, p2->valSize);

		counter++;
	}
	if(counter!=end-begin)
		ShowWarn("counter!=end-begin counter:%d end-begin:%d",counter,end-begin);
	
	free(shared_buff);
	//TODO
	//free shared_buff	
	

}//__global__	



void StartCPUShuffle2(thread_info_t *thread_info){

	cpu_context *d_g_state = (cpu_context*)(thread_info->d_g_state);
	job_configuration *cpu_job_conf = (job_configuration*)(thread_info->job_conf);


	//TODO put all jobs related object to job_conf
	bool configured;	
	int cpu_group_id;	
	int num_input_record;
	int num_cpus;	

	keyval_t * input_keyval_arr;
	keyval_arr_t *intermediate_keyval_arr_arr_p = d_g_state->intermediate_keyval_arr_arr_p;

	int total_count = 0;
	int index = 0;
	for(int i=0;i<d_g_state->num_input_record;i++){
		total_count += intermediate_keyval_arr_arr_p[i].arr_len;
	}//for

	d_g_state->sorted_intermediate_keyvals_arr = NULL;
	keyvals_t * sorted_intermediate_keyvals_arr = d_g_state->sorted_intermediate_keyvals_arr;

	int sorted_key_arr_len = 0;
	for(int i=0;i<d_g_state->num_input_record;i++){
		int len = intermediate_keyval_arr_arr_p[i].arr_len;
		for (int j=0;j<len;j++){


			char *key_i = (char *)(intermediate_keyval_arr_arr_p[i].cpu_arr[j].key);
			int keySize_i = (intermediate_keyval_arr_arr_p[i].arr[j].keySize);

			char *val_i = (char *)(intermediate_keyval_arr_arr_p[i].cpu_arr[j].val);
			int valSize_i = (intermediate_keyval_arr_arr_p[i].arr[j].valSize);

			int k = 0;
			for (; k<sorted_key_arr_len; k++){
				char *key_k = (char *)(sorted_intermediate_keyvals_arr[k].key);
				int keySize_k = sorted_intermediate_keyvals_arr[k].keySize;

				if ( cpu_compare(key_i, keySize_i, key_k, keySize_k) != 0 )
					continue;

				//found the match
				val_t *vals = sorted_intermediate_keyvals_arr[k].vals;
				sorted_intermediate_keyvals_arr[k].val_arr_len++;
				sorted_intermediate_keyvals_arr[k].vals = (val_t*)realloc(vals, sizeof(val_t)*(sorted_intermediate_keyvals_arr[k].val_arr_len));

				int index = sorted_intermediate_keyvals_arr[k].val_arr_len - 1;
				sorted_intermediate_keyvals_arr[k].vals[index].valSize = valSize_i;
				sorted_intermediate_keyvals_arr[k].vals[index].val = (char *)malloc(sizeof(char)*valSize_i);
				memcpy(sorted_intermediate_keyvals_arr[k].vals[index].val,val_i,valSize_i);
				break;

			}//for

			if (k == sorted_key_arr_len){

				if (sorted_key_arr_len == 0)
					sorted_intermediate_keyvals_arr = NULL;

				sorted_key_arr_len++;
				sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(sorted_intermediate_keyvals_arr, sizeof(keyvals_t)*sorted_key_arr_len);
				int index = sorted_key_arr_len-1;
				keyvals_t* kvals_p = (keyvals_t *)&(sorted_intermediate_keyvals_arr[index]);

				kvals_p->keySize = keySize_i;
				kvals_p->key = malloc(sizeof(char)*keySize_i);
				memcpy(kvals_p->key, key_i, keySize_i);

				kvals_p->vals = (val_t *)malloc(sizeof(val_t));
				kvals_p->val_arr_len = 1;

				kvals_p->vals[0].valSize = valSize_i;
				kvals_p->vals[0].val = (char *)malloc(sizeof(char)*valSize_i);
				memcpy(kvals_p->vals[0].val,val_i, valSize_i);

			}//if
		}//for j;
	}//for i;
	d_g_state->sorted_intermediate_keyvals_arr = sorted_intermediate_keyvals_arr;
	d_g_state->sorted_keyvals_arr_len = sorted_key_arr_len;

	ShowLog("CPU_GROUP_ID:[%d] #Intermediate Records:%d; #Intermediate Records:%d After Shuffle",d_g_state->cpu_group_id, total_count,sorted_key_arr_len);

}


void StartCPUShuffle(cpu_context *d_g_state){

#ifdef DEV_MODE
	bool configured;	
	int cpu_group_id;	
	int num_input_record;
	int num_cpus;	

	keyval_t * input_keyval_arr;
	keyval_arr_t *intermediate_keyval_arr_arr_p = d_g_state->intermediate_keyval_arr_arr_p;

	int total_count = 0;
	int index = 0;
	for(int i=0;i<d_g_state->num_input_record;i++){
		total_count += intermediate_keyval_arr_arr_p[i].arr_len;
	}//for

	ShowLog("total intermediate record count:%d\n",total_count);

	d_g_state->sorted_intermediate_keyvals_arr = NULL;
	keyvals_t * sorted_intermediate_keyvals_arr = d_g_state->sorted_intermediate_keyvals_arr;

	int sorted_key_arr_len = 0;
	for(int i=0;i<d_g_state->num_input_record;i++){
		int len = intermediate_keyval_arr_arr_p[i].arr_len;
		for (int j=0;j<len;j++){

			char *key_i = (char *)(intermediate_keyval_arr_arr_p[i].arr[j].key);
			int keySize_i = (intermediate_keyval_arr_arr_p[i].arr[j].keySize);


			char *val_i = (char *)(intermediate_keyval_arr_arr_p[i].arr[j].val);
			int valSize_i = (intermediate_keyval_arr_arr_p[i].arr[j].valSize);

			int k = 0;
			for (; k<sorted_key_arr_len; k++){
				char *key_k = (char *)(sorted_intermediate_keyvals_arr[k].key);
				int keySize_k = sorted_intermediate_keyvals_arr[k].keySize;

				if ( cpu_compare(key_i, keySize_i, key_k, keySize_k) != 0 )
					continue;

				//found the match
				val_t *vals = sorted_intermediate_keyvals_arr[k].vals;
				sorted_intermediate_keyvals_arr[k].val_arr_len++;
				sorted_intermediate_keyvals_arr[k].vals = (val_t*)realloc(vals, sizeof(val_t)*(sorted_intermediate_keyvals_arr[k].val_arr_len));

				int index = sorted_intermediate_keyvals_arr[k].val_arr_len - 1;
				sorted_intermediate_keyvals_arr[k].vals[index].valSize = valSize_i;
				sorted_intermediate_keyvals_arr[k].vals[index].val = (char *)malloc(sizeof(char)*valSize_i);
				memcpy(sorted_intermediate_keyvals_arr[k].vals[index].val,val_i,valSize_i);
				break;

			}//for

			if (k == sorted_key_arr_len){

				if (sorted_key_arr_len == 0)
					sorted_intermediate_keyvals_arr = NULL;

				sorted_key_arr_len++;
				sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(sorted_intermediate_keyvals_arr, sizeof(keyvals_t)*sorted_key_arr_len);
				int index = sorted_key_arr_len-1;
				keyvals_t* kvals_p = (keyvals_t *)&(sorted_intermediate_keyvals_arr[index]);

				kvals_p->keySize = keySize_i;
				kvals_p->key = malloc(sizeof(char)*keySize_i);
				memcpy(kvals_p->key, key_i, keySize_i);

				kvals_p->vals = (val_t *)malloc(sizeof(val_t));
				kvals_p->val_arr_len = 1;

				kvals_p->vals[0].valSize = valSize_i;
				kvals_p->vals[0].val = (char *)malloc(sizeof(char)*valSize_i);
				memcpy(kvals_p->vals[0].val,val_i, valSize_i);

			}//if
		}//for j;
	}//for i;
	d_g_state->sorted_intermediate_keyvals_arr = sorted_intermediate_keyvals_arr;
	d_g_state->sorted_keyvals_arr_len = sorted_key_arr_len;
	ShowLog("total number of different intermediate records:%d",sorted_key_arr_len);

#endif
}


void Shuffle4GPUOutput(gpu_context* d_g_state){

	cudaThreadSynchronize();
	int *count_arr = (int *)malloc(sizeof(int) * d_g_state->num_input_record);
	checkCudaErrors(cudaMemcpy(count_arr, d_g_state->d_intermediate_keyval_total_count, sizeof(int)*d_g_state->num_input_record, cudaMemcpyDeviceToHost));

	int total_count = 0;
	for(int i=0;i<d_g_state->num_input_record;i++){
		total_count += count_arr[i];
	}//for
	free(count_arr);

	ShowLog("Total Count of Intermediate Records:%d",total_count);
	checkCudaErrors(cudaMalloc((void **)&(d_g_state->d_intermediate_keyval_arr),sizeof(keyval_t)*total_count));

	int num_blocks = (d_g_state->num_mappers + (NUM_THREADS)-1)/(NUM_THREADS);
	int numGPUCores = getGPUCoresNum();
	dim3 blocks(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
	int numBlocks = (numGPUCores*16+(blocks.x*blocks.y)-1)/(blocks.x*blocks.y);
	dim3 grids(numBlocks, 1);


	copyDataFromDevice2Host1<<<grids,blocks>>>(*d_g_state);
	cudaThreadSynchronize();

	//TODO intermediate keyval_arr use pos_arr
	keyval_t * h_keyval_arr = (keyval_t *)malloc(sizeof(keyval_t)*total_count);
	checkCudaErrors(cudaMemcpy(h_keyval_arr, d_g_state->d_intermediate_keyval_arr, sizeof(keyval_t)*total_count, cudaMemcpyDeviceToHost));
	d_g_state->h_intermediate_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*total_count);
	keyval_pos_t *h_intermediate_keyvals_pos_arr = d_g_state->h_intermediate_keyval_pos_arr;

	int totalKeySize = 0;
	int totalValSize = 0;

	for (int i=0;i<total_count;i++){
		h_intermediate_keyvals_pos_arr[i].valPos= totalValSize;
		h_intermediate_keyvals_pos_arr[i].keyPos = totalKeySize;

		h_intermediate_keyvals_pos_arr[i].keySize = h_keyval_arr[i].keySize;
		h_intermediate_keyvals_pos_arr[i].valSize = h_keyval_arr[i].valSize;

		totalKeySize += (h_keyval_arr[i].keySize+3)/4*4;
		totalValSize += (h_keyval_arr[i].valSize+3)/4*4;


		if (totalValSize<0)
			exit(0);
	}//for
	d_g_state->totalValSize = totalValSize;
	d_g_state->totalKeySize = totalKeySize;

	ShowLog("allocate memory for totalKeySize:%d KB totalValSize:%d KB number of intermediate records:%d ", totalKeySize/1024, totalValSize/1024, total_count);
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_intermediate_keys_shared_buff,totalKeySize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_intermediate_vals_shared_buff,totalValSize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_intermediate_keyval_pos_arr,sizeof(keyval_pos_t)*total_count));
	checkCudaErrors(cudaMemcpy(d_g_state->d_intermediate_keyval_pos_arr, h_intermediate_keyvals_pos_arr, sizeof(keyval_pos_t)*total_count, cudaMemcpyHostToDevice));

	cudaThreadSynchronize();
	copyDataFromDevice2Host2<<<grids,blocks>>>(*d_g_state);
	cudaThreadSynchronize();

	d_g_state->h_intermediate_keys_shared_buff = malloc(sizeof(char)*totalKeySize);
	d_g_state->h_intermediate_vals_shared_buff = malloc(sizeof(char)*totalValSize);

	checkCudaErrors(cudaMemcpy(d_g_state->h_intermediate_keys_shared_buff,d_g_state->d_intermediate_keys_shared_buff,sizeof(char)*totalKeySize,cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(d_g_state->h_intermediate_vals_shared_buff,d_g_state->d_intermediate_vals_shared_buff,sizeof(char)*totalValSize,cudaMemcpyDeviceToHost));

	//////////////////////////////////////////////
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_sorted_keys_shared_buff,totalKeySize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_sorted_vals_shared_buff,totalValSize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_keyval_pos_arr,sizeof(keyval_pos_t)*total_count));

	d_g_state->h_sorted_keys_shared_buff = malloc(sizeof(char)*totalKeySize);
	d_g_state->h_sorted_vals_shared_buff = malloc(sizeof(char)*totalValSize);

	char *sorted_keys_shared_buff = (char *)d_g_state->h_sorted_keys_shared_buff;
	char *sorted_vals_shared_buff = (char *)d_g_state->h_sorted_vals_shared_buff;

	char *intermediate_key_shared_buff = (char *)d_g_state->h_intermediate_keys_shared_buff;
	char *intermediate_val_shared_buff = (char *)d_g_state->h_intermediate_vals_shared_buff;

	memcpy(sorted_keys_shared_buff, intermediate_key_shared_buff, totalKeySize);
	memcpy(sorted_vals_shared_buff, intermediate_val_shared_buff, totalValSize);

	int sorted_key_arr_len = 0;

	///////////////////////////////////////////////////////////////////////////////////////////////////
	//transfer the d_sorted_keyval_pos_arr to h_sorted_keyval_pos_arr
	//ShowLog("transfer the d_sorted_keyval_pos_arr to h_sorted_keyval_pos_arr");

	sorted_keyval_pos_t * h_sorted_keyval_pos_arr = NULL;
	for (int i=0; i<total_count; i++){
		int iKeySize = h_intermediate_keyvals_pos_arr[i].keySize;

		int j = 0;
		for (; j<sorted_key_arr_len; j++){

			int jKeySize = h_sorted_keyval_pos_arr[j].keySize;
			char *key_i = (char *)(intermediate_key_shared_buff + h_intermediate_keyvals_pos_arr[i].keyPos);
			char *key_j = (char *)(sorted_keys_shared_buff + h_sorted_keyval_pos_arr[j].keyPos);
			if (cpu_compare(key_i,iKeySize,key_j,jKeySize)!=0)
				continue;

			//found the match
			int arr_len = h_sorted_keyval_pos_arr[j].val_arr_len;
			h_sorted_keyval_pos_arr[j].val_pos_arr = (val_pos_t *)realloc(h_sorted_keyval_pos_arr[j].val_pos_arr, sizeof(val_pos_t)*(arr_len+1));
			h_sorted_keyval_pos_arr[j].val_pos_arr[arr_len].valSize = h_intermediate_keyvals_pos_arr[i].valSize;
			h_sorted_keyval_pos_arr[j].val_pos_arr[arr_len].valPos = h_intermediate_keyvals_pos_arr[i].valPos;
			h_sorted_keyval_pos_arr[j].val_arr_len ++;
			break;
		}//for

		if(j==sorted_key_arr_len){
			sorted_key_arr_len++;
			h_sorted_keyval_pos_arr = (sorted_keyval_pos_t *)realloc(h_sorted_keyval_pos_arr,sorted_key_arr_len*sizeof(sorted_keyval_pos_t));
			sorted_keyval_pos_t *p = &(h_sorted_keyval_pos_arr[sorted_key_arr_len - 1]);
			p->keySize = iKeySize;
			p->keyPos = h_intermediate_keyvals_pos_arr[i].keyPos;

			p->val_arr_len = 1;
			p->val_pos_arr = (val_pos_t*)malloc(sizeof(val_pos_t));
			p->val_pos_arr[0].valSize = h_intermediate_keyvals_pos_arr[i].valSize;
			p->val_pos_arr[0].valPos = h_intermediate_keyvals_pos_arr[i].valPos;
		}//if
	}

	d_g_state->h_sorted_keyval_pos_arr = h_sorted_keyval_pos_arr;
	d_g_state->d_sorted_keyvals_arr_len = sorted_key_arr_len;

	keyval_pos_t *tmp_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*total_count);
	ShowLog("GPU_ID:[%d] #input_records:%d #intermediate_records:%lu #different_intermediate_records:%d totalKeySize:%d KB totalValSize:%d KB", 
		d_g_state->gpu_id, d_g_state->num_input_record, total_count, sorted_key_arr_len,totalKeySize/1024,totalValSize/1024);

	int *pos_arr_4_pos_arr = (int*)malloc(sizeof(int)*sorted_key_arr_len);
	memset(pos_arr_4_pos_arr,0,sizeof(int)*sorted_key_arr_len);

	int	index = 0;
	for (int i=0;i<sorted_key_arr_len;i++){
		sorted_keyval_pos_t *p = (sorted_keyval_pos_t *)&(h_sorted_keyval_pos_arr[i]);

		for (int j=0;j<p->val_arr_len;j++){
			tmp_keyval_pos_arr[index].keyPos = p->keyPos;
			tmp_keyval_pos_arr[index].keySize = p->keySize;
			tmp_keyval_pos_arr[index].valPos = p->val_pos_arr[j].valPos;
			tmp_keyval_pos_arr[index].valSize = p->val_pos_arr[j].valSize;
			index++;
		}//for
		pos_arr_4_pos_arr[i] = index;

	}

	checkCudaErrors(cudaMemcpy(d_g_state->d_keyval_pos_arr,tmp_keyval_pos_arr,sizeof(keyval_pos_t)*total_count,cudaMemcpyHostToDevice));
	d_g_state->d_sorted_keyvals_arr_len = sorted_key_arr_len;

	checkCudaErrors(cudaMalloc((void**)&d_g_state->d_pos_arr_4_sorted_keyval_pos_arr,sizeof(int)*sorted_key_arr_len));
	checkCudaErrors(cudaMemcpy(d_g_state->d_pos_arr_4_sorted_keyval_pos_arr,pos_arr_4_pos_arr,sizeof(int)*sorted_key_arr_len,cudaMemcpyHostToDevice));

}


//host function sort_CPU
//copy intermediate records from device memory to host memory and sort the intermediate records there. 
//The host API cannot copy from dynamically allocated addresses on device runtime heap, only device code can access them

void sort_CPU(gpu_context* d_g_state){

#ifdef REMOVE

	//start_row_id sorting
	//partition

#endif

}



void PandaShuffleMergeCPU(panda_context *d_g_state_0, cpu_context *d_g_state_1){
	ShowLog("PandaShuffleMergeCPU CPU_GROUP_ID:[%d]", d_g_state_1->cpu_group_id);

	keyvals_t * panda_sorted_intermediate_keyvals_arr = d_g_state_0->sorted_intermediate_keyvals_arr;

	keyvals_t * cpu_sorted_intermediate_keyvals_arr = d_g_state_1->sorted_intermediate_keyvals_arr;

	void *key_0, *key_1;
	int keySize_0, keySize_1;
	bool equal;	

	for (int i=0; i<d_g_state_1->sorted_keyvals_arr_len; i++){
		key_1 = cpu_sorted_intermediate_keyvals_arr[i].key;
		keySize_1 = cpu_sorted_intermediate_keyvals_arr[i].keySize;

		int j;
		for (j=0; j<d_g_state_0->sorted_keyvals_arr_len; j++){
			key_0 = panda_sorted_intermediate_keyvals_arr[j].key;
			keySize_0 = panda_sorted_intermediate_keyvals_arr[j].keySize;

			if(cpu_compare(key_0,keySize_0,key_1,keySize_1)!=0)
				continue;


			//copy values from cpu_contex to panda context
			int val_arr_len_1 = cpu_sorted_intermediate_keyvals_arr[i].val_arr_len;
			int index = panda_sorted_intermediate_keyvals_arr[j].val_arr_len;
			if (panda_sorted_intermediate_keyvals_arr[j].val_arr_len ==0)
				panda_sorted_intermediate_keyvals_arr[j].vals = NULL;
			panda_sorted_intermediate_keyvals_arr[j].val_arr_len += val_arr_len_1;

			val_t *vals = panda_sorted_intermediate_keyvals_arr[j].vals;
			panda_sorted_intermediate_keyvals_arr[j].vals = (val_t*)realloc(vals, sizeof(val_t)*(panda_sorted_intermediate_keyvals_arr[j].val_arr_len));

			for (int k=0;k<val_arr_len_1;k++){
				char *val_0 = (char *)(cpu_sorted_intermediate_keyvals_arr[i].vals[k].val);
				int valSize_0 = cpu_sorted_intermediate_keyvals_arr[i].vals[k].valSize;

				panda_sorted_intermediate_keyvals_arr[j].vals[index+k].val = malloc(sizeof(char)*valSize_0);
				panda_sorted_intermediate_keyvals_arr[j].vals[index+k].valSize = valSize_0;
				memcpy(panda_sorted_intermediate_keyvals_arr[j].vals[index+k].val, val_0, valSize_0);

			}//for
			break;
		}//for

		if (j == d_g_state_0->sorted_keyvals_arr_len){

			if (d_g_state_0->sorted_keyvals_arr_len == 0) panda_sorted_intermediate_keyvals_arr = NULL;

			val_t *vals = cpu_sorted_intermediate_keyvals_arr[i].vals;
			int val_arr_len = cpu_sorted_intermediate_keyvals_arr[i].val_arr_len;

			d_g_state_0->sorted_keyvals_arr_len++;
			panda_sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(panda_sorted_intermediate_keyvals_arr, 
				sizeof(keyvals_t)*(d_g_state_0->sorted_keyvals_arr_len));

			int index = d_g_state_0->sorted_keyvals_arr_len-1;
			keyvals_t* kvals_p = (keyvals_t *)&(panda_sorted_intermediate_keyvals_arr[index]);

			kvals_p->keySize = keySize_1;
			kvals_p->key = malloc(sizeof(char)*keySize_1);
			memcpy(kvals_p->key, key_1, keySize_1);

			kvals_p->vals = (val_t *)malloc(sizeof(val_t)*val_arr_len);
			kvals_p->val_arr_len = val_arr_len;

			for (int k=0; k < val_arr_len; k++){
				char *val_0 = (char *)(cpu_sorted_intermediate_keyvals_arr[i].vals[k].val);
				int valSize_0 = cpu_sorted_intermediate_keyvals_arr[i].vals[k].valSize;

				kvals_p->vals[k].valSize = valSize_0;
				kvals_p->vals[k].val = (char *)malloc(sizeof(char)*valSize_0);

				memcpy(kvals_p->vals[k].val,val_0, valSize_0);

			}//for
		}//if (j == sorted_key_arr_len){
	}//if

	d_g_state_0->sorted_intermediate_keyvals_arr = cpu_sorted_intermediate_keyvals_arr;
	ShowLog("CPU_GROUP_ID:[%d] DONE. Sorted len:%d",d_g_state_1->cpu_group_id, d_g_state_0->sorted_keyvals_arr_len);

}


void PandaShuffleMergeGPU(panda_context *d_g_state_panda, gpu_context *d_g_state_gpu){

	//ShowLog("PandaShuffleMergeGPU GPU_ID:[%d] d_g_state_panda->sorted_keyvals_arr_len:%d",d_g_state_gpu->gpu_id,d_g_state_panda->sorted_keyvals_arr_len);

	char *sorted_keys_shared_buff_0 = (char *)d_g_state_gpu->h_sorted_keys_shared_buff;
	char *sorted_vals_shared_buff_0 = (char *)d_g_state_gpu->h_sorted_vals_shared_buff;

	sorted_keyval_pos_t *keyval_pos_arr_0 = d_g_state_gpu->h_sorted_keyval_pos_arr;
	keyvals_t * sorted_intermediate_keyvals_arr = d_g_state_panda->sorted_intermediate_keyvals_arr;

	void *key_0, *key_1;
	int keySize_0, keySize_1;
	bool equal;	


	int new_count = 0;
	for (int i=0;i<d_g_state_gpu->d_sorted_keyvals_arr_len;i++){
		//ShowLog("keyPos:%d",keyval_pos_arr_0[i].keyPos);
		key_0 = sorted_keys_shared_buff_0 + keyval_pos_arr_0[i].keyPos;
		keySize_0 = keyval_pos_arr_0[i].keySize;

		int j = 0;
		for (; j<d_g_state_panda->sorted_keyvals_arr_len; j++){

			key_1 = sorted_intermediate_keyvals_arr[j].key;
			keySize_1 = sorted_intermediate_keyvals_arr[j].keySize;

			if(cpu_compare(key_0,keySize_0,key_1,keySize_1)!=0)
				continue;

			val_t *vals = sorted_intermediate_keyvals_arr[j].vals;
			//copy values from gpu to cpu context
			int val_arr_len_0 =keyval_pos_arr_0[i].val_arr_len;
			val_pos_t * val_pos_arr =keyval_pos_arr_0[i].val_pos_arr;

			int index = sorted_intermediate_keyvals_arr[j].val_arr_len;
			sorted_intermediate_keyvals_arr[j].val_arr_len += val_arr_len_0;
			sorted_intermediate_keyvals_arr[j].vals = (val_t*)realloc(vals, sizeof(val_t)*(sorted_intermediate_keyvals_arr[j].val_arr_len));

			for (int k=0;k<val_arr_len_0;k++){

				char *val_0 = sorted_vals_shared_buff_0 + val_pos_arr[k].valPos;
				int valSize_0 = val_pos_arr[k].valSize;

				sorted_intermediate_keyvals_arr[j].vals[index+k].val = malloc(sizeof(char)*valSize_0);
				sorted_intermediate_keyvals_arr[j].vals[index+k].valSize = valSize_0;
				memcpy(sorted_intermediate_keyvals_arr[j].vals[index+k].val, val_0, valSize_0);
			}//for
			break;
		}//for

		if (j == d_g_state_panda->sorted_keyvals_arr_len){

			if (d_g_state_panda->sorted_keyvals_arr_len == 0) sorted_intermediate_keyvals_arr = NULL;
			int val_arr_len =keyval_pos_arr_0[i].val_arr_len;
			val_pos_t * val_pos_arr =keyval_pos_arr_0[i].val_pos_arr;
			d_g_state_panda->sorted_keyvals_arr_len++;

			sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(sorted_intermediate_keyvals_arr, sizeof(keyvals_t)*(d_g_state_panda->sorted_keyvals_arr_len));
			int index = d_g_state_panda->sorted_keyvals_arr_len-1;
			keyvals_t* kvals_p = (keyvals_t *)&(sorted_intermediate_keyvals_arr[index]);

			kvals_p->keySize = keySize_0;
			kvals_p->key = malloc(sizeof(char)*keySize_0);
			memcpy(kvals_p->key, key_0, keySize_0);

			kvals_p->vals = (val_t *)malloc(sizeof(val_t)*val_arr_len);
			kvals_p->val_arr_len = val_arr_len;

			for (int k=0; k < val_arr_len; k++){
				char *val_0 = sorted_vals_shared_buff_0 + val_pos_arr[k].valPos;
				int valSize_0 = val_pos_arr[k].valSize;

				kvals_p->vals[k].valSize = valSize_0;
				kvals_p->vals[k].val = (char *)malloc(sizeof(char)*valSize_0);
				memcpy(kvals_p->vals[k].val,val_0, valSize_0);
			}//for
		}//if (j == sorted_key_arr_len){
	}//if

	d_g_state_panda->sorted_intermediate_keyvals_arr = sorted_intermediate_keyvals_arr;

	ShowLog("GPU_ID:[%d] Panda added keyval len:%d GPU keyval len:%d",
		d_g_state_gpu->gpu_id,d_g_state_panda->sorted_keyvals_arr_len, d_g_state_gpu->d_sorted_keyvals_arr_len);

}

void Panda_Shuffle_Merge(gpu_context *d_g_state_0, gpu_context *d_g_state_1){

	char *sorted_keys_shared_buff_0 = (char *)d_g_state_0->h_sorted_keys_shared_buff;
	char *sorted_vals_shared_buff_0 = (char *)d_g_state_0->h_sorted_vals_shared_buff;

	char *sorted_keys_shared_buff_1 = (char *)d_g_state_1->h_sorted_keys_shared_buff;
	char *sorted_vals_shared_buff_1 = (char *)d_g_state_1->h_sorted_vals_shared_buff;

	sorted_keyval_pos_t *keyval_pos_arr_0 = d_g_state_0->h_sorted_keyval_pos_arr;
	sorted_keyval_pos_t *keyval_pos_arr_1 = d_g_state_1->h_sorted_keyval_pos_arr;

	int totalValSize_1 = d_g_state_1->totalValSize;
	int totalKeySize_1 = d_g_state_1->totalKeySize;

	void *key_0,*key_1;
	int keySize_0,keySize_1;
	bool equal;	
	//ShowLog("len1:%d  len2:%d\n",d_g_state_0->d_sorted_keyvals_arr_len, d_g_state_1->d_sorted_keyvals_arr_len);
	for (int i=0;i<d_g_state_0->d_sorted_keyvals_arr_len;i++){
		key_0 = sorted_keys_shared_buff_0 + keyval_pos_arr_0[i].keyPos;
		keySize_0 = keyval_pos_arr_0[i].keySize;

		int j;
		for (j=0;j<d_g_state_1->d_sorted_keyvals_arr_len;j++){
			key_1 = sorted_keys_shared_buff_1 + keyval_pos_arr_1[j].keyPos;
			keySize_1 = keyval_pos_arr_1[j].keySize;

			if(cpu_compare(key_0,keySize_0,key_1,keySize_1)!=0)
				continue;

			//copy all vals in d_g_state_0->h_sorted_keyval_pos_arr[i] to d_g_state_1->h_sorted_keyval_pos_arr[j];
			int incValSize = 0;
			int len0 = keyval_pos_arr_0[i].val_arr_len;
			int len1 = keyval_pos_arr_1[j].val_arr_len;
			//ShowLog("i:%d j:%d compare: key_0:%s key_1:%s  true:%s len0:%d len1:%d\n", i, j, key_0,key_1,(equal ? "true":"false"),len0,len1);
			keyval_pos_arr_1[j].val_pos_arr = (val_pos_t*)realloc(keyval_pos_arr_1[j].val_pos_arr,sizeof(val_pos_t)*(len0+len1));
			keyval_pos_arr_1[j].val_arr_len = len0+len1;

			for (int k = len1; k < len1 + len0; k++){
				keyval_pos_arr_1[j].val_pos_arr[k].valSize = keyval_pos_arr_0[i].val_pos_arr[k-len1].valSize;
				keyval_pos_arr_1[j].val_pos_arr[k].valPos = keyval_pos_arr_0[i].val_pos_arr[k-len1].valPos;
				incValSize += keyval_pos_arr_0[i].val_pos_arr[k-len1].valSize;
			}//for

			sorted_vals_shared_buff_1 = (char*)realloc(sorted_vals_shared_buff_1, totalValSize_1 + incValSize);
			for (int k = len1; k < len1 + len0; k++){
				void *val_1 = sorted_vals_shared_buff_1 + totalValSize_1;
				void *val_0 = sorted_vals_shared_buff_0+keyval_pos_arr_0[i].val_pos_arr[k-len1].valPos;
				memcpy(val_1, val_0, keyval_pos_arr_0[i].val_pos_arr[k-len1].valSize);
				totalValSize_1 += keyval_pos_arr_0[i].val_pos_arr[k-len1].valSize;
			}//for
			break;
		}//for (int j = 0;

		//key_0 is not exist in d_g_state_1->h_sorted_keyval_pos_arr, create new keyval pair position there
		if(j==d_g_state_1->d_sorted_keyvals_arr_len){

			sorted_keys_shared_buff_1 = (char*)realloc(sorted_keys_shared_buff_1, (totalKeySize_1 + keySize_0));
			//assert(keySize_0 == keyval_pos_arr_0[i].keySize);

			void *key_0 = sorted_keys_shared_buff_0 + keyval_pos_arr_0[i].keyPos;
			void *key_1 = sorted_keys_shared_buff_1 + totalKeySize_1;

			memcpy(key_1, key_0, keySize_0);
			totalKeySize_1 += keySize_0;

			keyval_pos_arr_1 = (sorted_keyval_pos_t *)realloc(keyval_pos_arr_1, sizeof(sorted_keyval_pos_t)*(d_g_state_1->d_sorted_keyvals_arr_len+1));
			sorted_keyval_pos_t *new_p = &(keyval_pos_arr_1[d_g_state_1->d_sorted_keyvals_arr_len]);
			d_g_state_1->d_sorted_keyvals_arr_len += 1;

			new_p->keySize = keySize_0;
			new_p->keyPos = totalKeySize_1 - keySize_0;

			int len0 = keyval_pos_arr_0[i].val_arr_len;
			new_p->val_arr_len = len0;
			new_p->val_pos_arr = (val_pos_t *)malloc(sizeof(val_pos_t)*len0);

			int incValSize = 0;
			for (int k = 0; k < len0; k++){
				new_p->val_pos_arr[k].valSize = keyval_pos_arr_0[i].val_pos_arr[k].valSize;
				new_p->val_pos_arr[k].valPos = keyval_pos_arr_0[i].val_pos_arr[k].valPos;
				incValSize += keyval_pos_arr_0[i].val_pos_arr[k].valSize;
			}//for
			sorted_vals_shared_buff_1 = (char*)realloc(sorted_vals_shared_buff_1,(totalValSize_1 + incValSize));

			for (int k = 0; k < len0; k++){
				void *val_1 = sorted_vals_shared_buff_1 + totalValSize_1;
				void *val_0 = sorted_vals_shared_buff_0 + keyval_pos_arr_0[i].val_pos_arr[k].valPos;
				memcpy(val_1,val_0,keyval_pos_arr_0[i].val_pos_arr[k].valSize);
				totalValSize_1 += keyval_pos_arr_0[i].val_pos_arr[k].valSize;
			}//for
		}//if(j==arr_len)
	}//for (int i = 0;

	d_g_state_1->h_sorted_keyval_pos_arr = keyval_pos_arr_1;

	int total_count = 0;
	for (int i=0; i<d_g_state_1->d_sorted_keyvals_arr_len; i++){
		total_count += d_g_state_1->h_sorted_keyval_pos_arr[i].val_arr_len;
	}//for
	ShowLog("total number of intermeidate records on two GPU's:%d",total_count);
	keyval_pos_t *tmp_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*total_count);
	ShowLog("total number of different intermediate records on two GPU's:%d",d_g_state_1->d_sorted_keyvals_arr_len);

	int *pos_arr_4_pos_arr = (int*)malloc(sizeof(int)*d_g_state_1->d_sorted_keyvals_arr_len);
	memset(pos_arr_4_pos_arr,0,sizeof(int)*d_g_state_1->d_sorted_keyvals_arr_len);

	int	index = 0;
	for (int i=0; i<d_g_state_1->d_sorted_keyvals_arr_len; i++){
		sorted_keyval_pos_t *p = (sorted_keyval_pos_t *)&(d_g_state_1->h_sorted_keyval_pos_arr[i]);

		for (int j=0;j<p->val_arr_len;j++){
			tmp_keyval_pos_arr[index].keyPos = p->keyPos;
			tmp_keyval_pos_arr[index].keySize = p->keySize;
			tmp_keyval_pos_arr[index].valPos = p->val_pos_arr[j].valPos;
			tmp_keyval_pos_arr[index].valSize = p->val_pos_arr[j].valSize;
			//printf("tmp_keyval_pos_arr[%d].keyPos:%d  keySize:%d valPos:%d valSize:%d\n",
			//index,p->keyPos,p->keySize,p->val_pos_arr[j].valPos,p->val_pos_arr[j].valSize);
			//printf("key:%s val:%d\n",(char*)(sorted_keys_shared_buff_1+p->keyPos), *(int*)(sorted_vals_shared_buff_1+p->val_pos_arr[j].valPos));
			index++;
		}//for
		pos_arr_4_pos_arr[i] = index;
	}

	//printf("totalKeySize_1:%d  totalValSize_1:%d\n",totalKeySize_1,totalValSize_1);
	//printf("%s\n",sorted_keys_shared_buff_1);

	checkCudaErrors(cudaMalloc((void**)&d_g_state_1->d_keyval_pos_arr,sizeof(keyval_pos_t)*total_count));
	checkCudaErrors(cudaMemcpy(d_g_state_1->d_keyval_pos_arr,tmp_keyval_pos_arr,sizeof(keyval_pos_t)*total_count,cudaMemcpyHostToDevice));
	//d_g_state_1->d_sorted_keyvals_arr_len = d_g_state_1->d_sorted_keyvals_arr_len;
	checkCudaErrors(cudaMalloc((void**)&d_g_state_1->d_pos_arr_4_sorted_keyval_pos_arr,sizeof(int)*d_g_state_1->d_sorted_keyvals_arr_len));
	checkCudaErrors(cudaMemcpy(d_g_state_1->d_pos_arr_4_sorted_keyval_pos_arr,pos_arr_4_pos_arr,sizeof(int)*d_g_state_1->d_sorted_keyvals_arr_len,cudaMemcpyHostToDevice));

	//TODO release these buffer bebore allocate
	checkCudaErrors(cudaMalloc((void **)&d_g_state_1->d_sorted_keys_shared_buff,totalKeySize_1));
	checkCudaErrors(cudaMalloc((void **)&d_g_state_1->d_sorted_vals_shared_buff,totalValSize_1));

	checkCudaErrors(cudaMemcpy(d_g_state_1->d_sorted_keys_shared_buff,sorted_keys_shared_buff_1,totalKeySize_1,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_g_state_1->d_sorted_vals_shared_buff,sorted_vals_shared_buff_1,totalValSize_1,cudaMemcpyHostToDevice));

	//d_g_state_1->d_sorted_keys_shared_buff = sorted_keys_shared_buff_1; 
	//d_g_state_1->d_sorted_vals_shared_buff = sorted_vals_shared_buff_1;
	d_g_state_1->totalKeySize = totalKeySize_1;
	d_g_state_1->totalValSize = totalValSize_1;

}

#endif 
