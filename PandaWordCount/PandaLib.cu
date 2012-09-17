/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: PandaLib.cu 
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2012-09-02

	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */

#ifndef __PANDALIB_CU__
#define __PANDALIB_CU__

#include "Panda.h"
#include "map.cu"
#include "reduce.cu"

//----------------------------------------------
//Get default runtime configuration
//return: default spec
//----------------------------------------------
job_configuration *GetJobConf(){

	job_configuration *job_conf = (job_configuration *)malloc(sizeof(job_configuration));

	if (job_conf == NULL) exit(-1);
	memset(job_conf, 0, sizeof(job_configuration));
	job_conf->num_input_record = 0;
	job_conf->input_keyval_arr = NULL;
	job_conf->auto_tuning = false;
	
	job_conf->num_mappers = 0;
	job_conf->num_reducers = 0;
	job_conf->num_gpus = 0;
	job_conf->num_cpus_cores = 0;
	job_conf->num_cpus_groups = 0;

	return job_conf;
}//gpu_context


gpu_context *GetGPUContext(){
	gpu_context *d_g_state = (gpu_context*)malloc(sizeof(gpu_context));
	if (d_g_state == NULL) exit(-1);
	memset(d_g_state, 0, sizeof(gpu_context));
	d_g_state->configured = false;
	d_g_state->h_input_keyval_arr = NULL;
	d_g_state->num_mappers = 0;
	d_g_state->num_reducers = 0;
	return d_g_state;
}//gpu_context
			 
cpu_context *GetCPUContext(){
	cpu_context *d_g_state = (cpu_context*)malloc(sizeof(cpu_context));
	if (d_g_state == NULL) exit(-1);
	memset(d_g_state, 0, sizeof(cpu_context));
	d_g_state->configured = false;
	d_g_state->input_keyval_arr = NULL;
	return d_g_state;
}//gpu_context

panda_context *GetPandaContext(){
	
	panda_context *d_g_state = (panda_context*)malloc(sizeof(panda_context));
	
	if (d_g_state == NULL) exit(-1);
	
	d_g_state->input_keyval_arr = NULL;
	d_g_state->intermediate_keyval_arr_arr_p = NULL;
	d_g_state->sorted_intermediate_keyvals_arr = NULL;
	d_g_state->sorted_keyvals_arr_len = 0;
	d_g_state->num_gpus = 0;
	d_g_state->gpu_context = NULL;
	d_g_state->num_cpus_groups = 0;
	d_g_state->cpu_context = NULL;

	return d_g_state;

}//panda_context


//For version 0.3
void InitCPUMapReduce2(thread_info_t * thread_info){

	cpu_context *d_g_state = (cpu_context *)(thread_info->d_g_state);
	job_configuration *job_conf = (job_configuration *)(thread_info->job_conf);

	if (job_conf->num_input_record<=0) { DoError("Error: no any input keys"); exit(-1);}
	if (job_conf->input_keyval_arr == NULL) { DoError("Error: input_keyval_arr == NULL"); exit(-1);}
	if (d_g_state->num_cpus_cores <= 0) {	DoError("Error: d_g_state->num_cpus == 0"); exit(-1);}

	//DoLog("d_g_state->configured:%s  enable for iterative applications",d_g_state->configured? "true" : "false");
	//if (d_g_state->configured)
	//	return;
	
	int totalKeySize = 0;
	int totalValSize = 0;

	for(int i=0;i<job_conf->num_input_record;i++){
		totalKeySize += job_conf->input_keyval_arr[i].keySize;
		totalValSize += job_conf->input_keyval_arr[i].valSize;
	}//for

	DoLog("CPU_GROUP_ID:[%d] num_input_record:%d, totalKeySize:%d totalValSize:%d num_cpus:%d", 
		d_g_state->cpu_group_id, job_conf->num_input_record, totalKeySize, totalValSize, d_g_state->num_cpus_cores);

	//TODO determin num_cpus
	//d_g_state->num_cpus = 12;
	int num_cpus_cores = d_g_state->num_cpus_cores;

	d_g_state->panda_cpu_task = (pthread_t *)malloc(sizeof(pthread_t)*(num_cpus_cores));
	d_g_state->panda_cpu_task_info = (panda_cpu_task_info_t *)malloc(sizeof(panda_cpu_task_info_t)*(num_cpus_cores));

	d_g_state->intermediate_keyval_arr_arr_p = (keyval_arr_t *)malloc(sizeof(keyval_arr_t)*job_conf->num_input_record);
	memset(d_g_state->intermediate_keyval_arr_arr_p, 0, sizeof(keyval_arr_t)*job_conf->num_input_record);

	for (int i=0;i<num_cpus_cores;i++){
		d_g_state->panda_cpu_task_info[i].d_g_state = d_g_state;
		d_g_state->panda_cpu_task_info[i].cpu_job_conf = job_conf;
		d_g_state->panda_cpu_task_info[i].num_cpus_cores = num_cpus_cores;
		d_g_state->panda_cpu_task_info[i].start_row_idx = 0;
		d_g_state->panda_cpu_task_info[i].end_row_idx = 0;
	}//for
	
	d_g_state->configured = true;
	DoLog("CPU_GROUP_ID:[%d] DONE",d_g_state->cpu_group_id);

}

#ifdef DEV_MODE
//For Version 0.2 depressed
void InitCPUMapReduce(cpu_context* d_g_state)
{	
	if (d_g_state->num_input_record<=0) { DoError("Error: no any input keys"); exit(-1);}
	if (d_g_state->input_keyval_arr == NULL) { DoError("Error: input_keyval_arr == NULL"); exit(-1);}
	if (d_g_state->num_cpus_cores <= 0) {	DoError("Error: d_g_state->num_cpus == 0"); exit(-1);}

	//DoLog("d_g_state->configured:%s  enable for iterative applications",d_g_state->configured? "true" : "false");
	//if (d_g_state->configured)
	//	return;

	DoLog("d_g_state->num_input_record:%d",d_g_state->num_input_record);
	int totalKeySize = 0;
	int totalValSize = 0;

	for(int i=0;i<d_g_state->num_input_record;i++){
		totalKeySize += d_g_state->input_keyval_arr[i].keySize;
		totalValSize += d_g_state->input_keyval_arr[i].valSize;
	}//for
	DoLog("totalKeySize:%d totalValSize:%d num_cpus:%d", totalKeySize, totalValSize, d_g_state->num_cpus_cores);
	
	int num_cpus_cores = d_g_state->num_cpus_cores;
	d_g_state->panda_cpu_task = (pthread_t *)malloc(sizeof(pthread_t)*(num_cpus_cores));
	d_g_state->panda_cpu_task_info = (panda_cpu_task_info_t *)malloc(sizeof(panda_cpu_task_info_t)*(num_cpus_cores));

	d_g_state->intermediate_keyval_arr_arr_p = (keyval_arr_t *)malloc(sizeof(keyval_arr_t)*d_g_state->num_input_record);
	memset(d_g_state->intermediate_keyval_arr_arr_p, 0, sizeof(keyval_arr_t)*d_g_state->num_input_record);

	for (int i=0;i<num_cpus_cores;i++){
		d_g_state->panda_cpu_task_info[i].d_g_state = d_g_state;
		d_g_state->panda_cpu_task_info[i].num_cpus_cores = num_cpus_cores;
		d_g_state->panda_cpu_task_info[i].start_row_idx = 0;
		d_g_state->panda_cpu_task_info[i].end_idx = 0;
	}//for
	d_g_state->configured = true;
	DoLog("DONE");
}//void
#endif

#ifdef DEV_MODE
//For Version 0.3 test depressed
void InitGPUMapReduce4(thread_info_t* thread_info)
{	
	gpu_context *d_g_state = (gpu_context *)(thread_info->d_g_state);
	job_configuration* gpu_job_conf = (job_configuration*)(thread_info->job_conf);
	keyval_t * kv_p = gpu_job_conf->input_keyval_arr;

	DoLog("d_g_state->configured:%s  enable for iterative applications",d_g_state->configured? "true" : "false");
	//if (d_g_state->configured)
	//	return;
	DoLog("copy %d input records from Host to GPU memory",gpu_job_conf->num_input_record);
	//checkCudaErrors(cudaMalloc((void **)&d_g_state->d_input_keyval_arr,sizeof(keyval_t)*d_g_state->num_input_record));
	int totalKeySize = 0;
	int totalValSize = 0;
	for(int i=0;i<gpu_job_conf->num_input_record;i++){
		totalKeySize += kv_p[i].keySize;
		totalValSize += kv_p[i].valSize;
	}//for
	DoLog("totalKeySize:%d totalValSize:%d", totalKeySize, totalValSize);
	
	void *input_vals_shared_buff = malloc(totalValSize);
	void *input_keys_shared_buff = malloc(totalKeySize);
	keyval_pos_t *input_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*gpu_job_conf->num_input_record);
	
	int keyPos = 0;
	int valPos = 0;
	int keySize = 0;
	int valSize = 0;
	
	for(int i=0; i<gpu_job_conf->num_input_record; i++){
		
		keySize = kv_p[i].keySize;
		valSize = kv_p[i].valSize;
		
		memcpy((char *)input_keys_shared_buff + keyPos,(char *)(kv_p[i].key), keySize);
		memcpy((char *)input_vals_shared_buff + valPos,(char *)(kv_p[i].val), valSize);
		
		input_keyval_pos_arr[i].keySize = keySize;
		input_keyval_pos_arr[i].keyPos = keyPos;
		input_keyval_pos_arr[i].valPos = valPos;
		input_keyval_pos_arr[i].valSize = valSize;

		keyPos += keySize;	
		valPos += valSize;
	}//for

	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_input_keyval_pos_arr,sizeof(keyval_pos_t)*gpu_job_conf->num_input_record));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_input_keys_shared_buff, totalKeySize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_input_vals_shared_buff, totalValSize));

	checkCudaErrors(cudaMemcpy(d_g_state->d_input_keyval_pos_arr, input_keyval_pos_arr,sizeof(keyval_pos_t)*gpu_job_conf->num_input_record ,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_g_state->d_input_keys_shared_buff, input_keys_shared_buff,totalKeySize ,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_g_state->d_input_vals_shared_buff, input_vals_shared_buff,totalValSize ,cudaMemcpyHostToDevice));

	//checkCudaErrors(cudaMemcpy(d_g_state->d_input_keyval_arr,h_buff,sizeof(keyval_t)*d_g_state->num_input_record,cudaMemcpyHostToDevice));
	cudaThreadSynchronize(); 
	d_g_state->configured = true;
}//void
#endif


void InitGPUMapReduce3(gpu_context* d_g_state)
{	

	DoLog("d_g_state->configured:%s  enable for iterative applications",d_g_state->configured? "true" : "false");
	//if (d_g_state->configured)
	//	return;
	
	//checkCudaErrors(cudaMalloc((void **)&d_g_state->d_input_keyval_arr,sizeof(keyval_t)*d_g_state->num_input_record));
	int totalKeySize = 0;
	int totalValSize = 0;
	for(int i=0;i<d_g_state->num_input_record;i++){
		totalKeySize += d_g_state->h_input_keyval_arr[i].keySize;
		totalValSize += d_g_state->h_input_keyval_arr[i].valSize;
	}//for
	DoLog("copy %d input records from Host to GPU memory totalKeySize:%d totalValSize:%d",d_g_state->num_input_record, totalKeySize, totalValSize);
	
	void *input_vals_shared_buff = malloc(totalValSize);
	void *input_keys_shared_buff = malloc(totalKeySize);
	keyval_pos_t *input_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*d_g_state->num_input_record);
	
	int keyPos = 0;
	int valPos = 0;
	int keySize = 0;
	int valSize = 0;
	
	for(int i=0;i<d_g_state->num_input_record;i++){
		
		keySize = d_g_state->h_input_keyval_arr[i].keySize;
		valSize = d_g_state->h_input_keyval_arr[i].valSize;
		
		memcpy((char *)input_keys_shared_buff + keyPos,(char *)(d_g_state->h_input_keyval_arr[i].key), keySize);
		memcpy((char *)input_vals_shared_buff + valPos,(char *)(d_g_state->h_input_keyval_arr[i].val), valSize);
		
		input_keyval_pos_arr[i].keySize = keySize;
		input_keyval_pos_arr[i].keyPos = keyPos;
		input_keyval_pos_arr[i].valPos = valPos;
		input_keyval_pos_arr[i].valSize = valSize;

		keyPos += keySize;	
		valPos += valSize;
	}//for

	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_input_keyval_pos_arr,sizeof(keyval_pos_t)*d_g_state->num_input_record));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_input_keys_shared_buff, totalKeySize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_input_vals_shared_buff, totalValSize));

	checkCudaErrors(cudaMemcpy(d_g_state->d_input_keyval_pos_arr, input_keyval_pos_arr,sizeof(keyval_pos_t)*d_g_state->num_input_record ,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_g_state->d_input_keys_shared_buff, input_keys_shared_buff,totalKeySize ,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_g_state->d_input_vals_shared_buff, input_vals_shared_buff,totalValSize ,cudaMemcpyHostToDevice));

	//checkCudaErrors(cudaMemcpy(d_g_state->d_input_keyval_arr,h_buff,sizeof(keyval_t)*d_g_state->num_input_record,cudaMemcpyHostToDevice));
	cudaThreadSynchronize(); 
	d_g_state->configured = true;

}//void

#ifdef DEV_MODE
void InitGPUMapReduce2(gpu_context* d_g_state)
{	
	
	DoLog("d_g_state->num_input_record:%d",d_g_state->num_input_record);
	//checkCudaErrors(cudaMalloc((void **)&d_g_state->d_input_keyval_arr,sizeof(keyval_t)*d_g_state->num_input_record));

	int totalKeySize = 0;
	int totalValSize = 0;

	for(int i=0;i<d_g_state->num_input_record;i++){
		totalKeySize += d_g_state->h_input_keyval_arr[i].keySize;
		totalValSize += d_g_state->h_input_keyval_arr[i].valSize;
	}//for

	void *input_vals_shared_buff = malloc(totalValSize);
	void *input_keys_shared_buff = malloc(totalKeySize);
	keyval_pos_t *input_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*d_g_state->num_input_record);

	int keyPos = 0;
	int valPos = 0;
	int keySize = 0;
	int valSize = 0;

	for(int i=0;i<d_g_state->num_input_record;i++){
		
		keySize = d_g_state->h_input_keyval_arr[i].keySize;
		valSize = d_g_state->h_input_keyval_arr[i].valSize;
		
		memcpy((char *)input_keys_shared_buff + keyPos,(char *)(d_g_state->h_input_keyval_arr[i].key), keySize);
		memcpy((char *)input_vals_shared_buff + valPos,(char *)(d_g_state->h_input_keyval_arr[i].val), valSize);
		
		input_keyval_pos_arr[i].keySize = keySize;
		input_keyval_pos_arr[i].keyPos = keyPos;
		input_keyval_pos_arr[i].valPos = valPos;
		input_keyval_pos_arr[i].valSize = valSize;

		keyPos += keySize;	
		valPos += valSize;

	}//for

	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_input_keyval_pos_arr,sizeof(keyval_pos_t)*d_g_state->num_input_record));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_input_keys_shared_buff, totalKeySize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_input_vals_shared_buff, totalValSize));

	checkCudaErrors(cudaMemcpy(d_g_state->d_input_keyval_pos_arr, input_keyval_pos_arr,sizeof(keyval_pos_t)*d_g_state->num_input_record ,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_g_state->d_input_keys_shared_buff, input_keys_shared_buff,totalKeySize ,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_g_state->d_input_vals_shared_buff, input_vals_shared_buff,totalValSize ,cudaMemcpyHostToDevice));

	//checkCudaErrors(cudaMemcpy(d_g_state->d_input_keyval_arr,h_buff,sizeof(keyval_t)*d_g_state->num_input_record,cudaMemcpyHostToDevice));
	cudaThreadSynchronize(); 

}//void
#endif


void InitCPUDevice(thread_info_t*thread_info){

	//------------------------------------------
	//1, init CPU device
	//------------------------------------------
	cpu_context *d_g_state = (cpu_context *)(thread_info->d_g_state);
	if (d_g_state->num_cpus_cores<=0) d_g_state->num_cpus_cores = getCPUCoresNum();

	int tid = thread_info->tid;
	DoLog( "CPU_GROUP_ID:[%d] Init CPU Deivce",d_g_state->cpu_group_id);
	
}

void InitGPUDevice(thread_info_t*thread_info){
	
	//------------------------------------------
	//1, init device
	//------------------------------------------
	
	gpu_context *d_g_state = (gpu_context *)(thread_info->d_g_state);
	int tid = thread_info->tid;
	int assigned_gpu_id = d_g_state->gpu_id;
	int num_gpus = d_g_state->num_gpus;
	if (num_gpus == 0) {
		DoError("error num_gpus == 0");
		exit(-1);
	}//gpu_context
	
	int gpu_id;
	cudaGetDevice(&gpu_id);
	int gpu_count = 0;
	cudaGetDeviceCount(&gpu_count);

	cudaDeviceProp gpu_dev;
	cudaGetDeviceProperties(&gpu_dev, gpu_id);
	//DoLog("Configure Device ID:%d: Device Name:%s MultProcessorCount:%d sm_per_multiproc:%d", i, gpu_dev.name,gpu_dev.multiProcessorCount,sm_per_multiproc);

	DoLog("TID:[%d] check GPU ids: cur_gpu_id:[%d] assig_gpu_id:[%d] cudaGetDeviceCount:[%d] GPU name:%s", 
		tid, gpu_id, assigned_gpu_id,  gpu_count, gpu_dev.name);

	if ( gpu_id != assigned_gpu_id ){
		//DoLog("cudaSetDevice gpu_id %d == (tid num_gpus) %d ", gpu_id, tid%num_gpus);
		cudaSetDevice(assigned_gpu_id % num_gpus);  
	}//if

	//cudaGetDevice(&gpu_id);
	//d_g_state->gpu_id = gpu_id;
		
	size_t total_mem,avail_mem, heap_limit;
	checkCudaErrors(cudaMemGetInfo( &avail_mem, &total_mem ));
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, (int)(total_mem*0.2)); 
	cudaDeviceGetLimit(&heap_limit, cudaLimitMallocHeapSize);

	int numGPUCores = getGPUCoresNum();
	//DoLog("TID:[%d] num_gpus:%d gpu_id:%d ",tid,num_gpus,gpu_id);
	DoLog("GPU_ID:[%d] numGPUCores:%d cudaLimitMallocHeapSize:%d MB avail_mem:%d MB total_mem:%d MB",
		 gpu_id, numGPUCores,heap_limit/1024/1024, avail_mem/1024/1024,total_mem/1024/1024);

}




void AddPandaTask(job_configuration* job_conf,
						void*		key, 
						void*		val,
						int		keySize, 
						int		valSize){
	
	int len = job_conf->num_input_record;
	if (len<0) return;
	if (len == 0) job_conf->input_keyval_arr = NULL;

	job_conf->input_keyval_arr = (keyval_t *)realloc(job_conf->input_keyval_arr, sizeof(keyval_t)*(len+1));
	job_conf->input_keyval_arr[len].keySize = keySize;
	job_conf->input_keyval_arr[len].valSize = valSize;
	job_conf->input_keyval_arr[len].key = malloc(keySize);
	job_conf->input_keyval_arr[len].val = malloc(valSize);

	memcpy(job_conf->input_keyval_arr[len].key,key,keySize);
	memcpy(job_conf->input_keyval_arr[len].val,val,valSize);
	job_conf->num_input_record++;
	
}

void AddReduceInputRecordGPU(gpu_context* d_g_state, keyvals_t * sorted_intermediate_keyvals_arr, int start_row_id, int end_row_id){
	
	long total_count = 0;
	for(int i=start_row_id;i<end_row_id;i++){
		total_count += sorted_intermediate_keyvals_arr[i].val_arr_len;
	}//for
	
	int totalKeySize = 0;
	int totalValSize = 0;
	for(int i=start_row_id;i<end_row_id;i++){
		totalKeySize += sorted_intermediate_keyvals_arr[i].keySize;
		for (int j=0;j<sorted_intermediate_keyvals_arr[i].val_arr_len;j++)
		totalValSize += sorted_intermediate_keyvals_arr[i].vals[j].valSize;
	}//for
	DoLog("totalKeySize:%d totalValSize:%d ",totalKeySize,totalValSize);
		
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_sorted_keys_shared_buff,totalKeySize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_sorted_vals_shared_buff,totalValSize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_keyval_pos_arr,sizeof(keyval_pos_t)*total_count));
	
	d_g_state->h_sorted_keys_shared_buff = malloc(sizeof(char)*totalKeySize);
	d_g_state->h_sorted_vals_shared_buff = malloc(sizeof(char)*totalValSize);
	
	char *sorted_keys_shared_buff = (char *)d_g_state->h_sorted_keys_shared_buff;
	char *sorted_vals_shared_buff = (char *)d_g_state->h_sorted_vals_shared_buff;
	char *keyval_pos_arr = (char *)malloc(sizeof(keyval_pos_t)*total_count);
	
	int sorted_key_arr_len = (end_row_id-start_row_id);
	keyval_pos_t *tmp_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*total_count);
	DoLog("total number of different intermediate records:%d total records:%d", end_row_id - start_row_id, total_count);
	int *pos_arr_4_pos_arr = (int*)malloc(sizeof(int)*(sorted_key_arr_len));
	memset(pos_arr_4_pos_arr,0,sizeof(int)*sorted_key_arr_len);

	int index = 0;
	int keyPos = 0;
	int valPos = 0;
	for (int i=start_row_id;i<end_row_id;i++){
		keyvals_t* p = (keyvals_t*)&(sorted_intermediate_keyvals_arr[i]);
		memcpy(sorted_keys_shared_buff+keyPos,p->key, p->keySize);
		
		for (int j=0;j<p->val_arr_len;j++){
			tmp_keyval_pos_arr[index].keyPos = keyPos;
			tmp_keyval_pos_arr[index].keySize = p->keySize;
			tmp_keyval_pos_arr[index].valPos = valPos;
			tmp_keyval_pos_arr[index].valSize = p->vals[j].valSize;
			memcpy(sorted_vals_shared_buff + valPos,p->vals[j].val,p->vals[j].valSize);
			valPos += p->vals[j].valSize;
			index++;
		}//for
		keyPos += p->keySize;
		pos_arr_4_pos_arr[i-start_row_id] = index;
	}//

	d_g_state->d_sorted_keyvals_arr_len = end_row_id-start_row_id;
	checkCudaErrors(cudaMemcpy(d_g_state->d_keyval_pos_arr,tmp_keyval_pos_arr,sizeof(keyval_pos_t)*total_count,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_g_state->d_pos_arr_4_sorted_keyval_pos_arr,sizeof(int)*sorted_key_arr_len));
	checkCudaErrors(cudaMemcpy(d_g_state->d_pos_arr_4_sorted_keyval_pos_arr,pos_arr_4_pos_arr,sizeof(int)*sorted_key_arr_len,cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_g_state->d_sorted_keys_shared_buff, sorted_keys_shared_buff, sizeof(char)*totalKeySize,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_g_state->d_sorted_vals_shared_buff, sorted_vals_shared_buff, sizeof(char)*totalValSize,cudaMemcpyHostToDevice));

}


void AddMapInputRecordGPU(gpu_context* d_g_state,
						keyval_t *kv_p, int start_row_id, int end_row_id){

	if (end_row_id<=start_row_id) {	DoError("error! end_row_id<=start_row_id");		return;	}
	int len = d_g_state->num_input_record;
	if (len<0) {	DoError("error! len<0");		return;	}
	if (len == 0) d_g_state->h_input_keyval_arr = NULL;

	DoLog("GPU_ID:[%d] add map input record into gpu device current #input:%d added #input:%d",d_g_state->gpu_id, len, end_row_id - start_row_id);			

	d_g_state->h_input_keyval_arr = (keyval_t *)realloc(d_g_state->h_input_keyval_arr, sizeof(keyval_t)*(len + end_row_id - start_row_id));
	//assert(d_g_state->h_input_keyval_arr != NULL);
	for (int i=start_row_id;i<end_row_id;i++){
	d_g_state->h_input_keyval_arr[len].keySize = kv_p[i].keySize;
	d_g_state->h_input_keyval_arr[len].valSize = kv_p[i].valSize;
	d_g_state->h_input_keyval_arr[len].key = kv_p[i].key;
	d_g_state->h_input_keyval_arr[len].val = kv_p[i].val;
	//memcpy(d_g_state->h_input_keyval_arr[len].key,key,keySize);
	//memcpy(d_g_state->h_input_keyval_arr[len].val,val,valSize);
	d_g_state->num_input_record++;
	len++;
	}

}


void AddMapInputRecordCPU(cpu_context* d_g_state,
						keyval_t *kv_p, int start_row_id, int end_row_id){

	if (end_row_id<=start_row_id) {	DoError("error! end_row_id[%d] <= start_row_id[%d]",end_row_id, start_row_id);		return;	}	
	int len = d_g_state->num_input_record;
	if (len<0) {	DoError("error! len<0");		return;	}
	if (len == 0) d_g_state->input_keyval_arr = NULL;

	DoLog("CPU_GROUP_ID:[%d] add map input record for cpu device current #input:%d added #input:%d",d_g_state->cpu_group_id,len,end_row_id-start_row_id);			
	d_g_state->input_keyval_arr = (keyval_t *)realloc(d_g_state->input_keyval_arr, sizeof(keyval_t)*(len+end_row_id-start_row_id));

	for (int i=start_row_id;i<end_row_id;i++){
	
	d_g_state->input_keyval_arr[len].keySize = kv_p[i].keySize;
	d_g_state->input_keyval_arr[len].valSize = kv_p[i].valSize;
	d_g_state->input_keyval_arr[len].key = kv_p[i].key;
	d_g_state->input_keyval_arr[len].val = kv_p[i].val;
	
	d_g_state->num_input_record++;
	len++;
	}

}

void AddReduceInputRecordCPU(cpu_context* d_g_state,
						keyvals_t *kv_p, int start_row_id, int end_row_id){
							
    if (end_row_id<start_row_id){	DoError("error! end_row_id<=start_row_id");		return;	}
	int len = d_g_state->sorted_keyvals_arr_len;
	if (len<0) {	DoError("error! len<0");		return;	}
	if (len == 0) d_g_state->sorted_intermediate_keyvals_arr = NULL;

	//DoLog("start_row_id:%d, end_row_id:%d, len:%d\n",start_row_id,end_row_id,len);

	d_g_state->sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(d_g_state->sorted_intermediate_keyvals_arr, 
		sizeof(keyvals_t)*(len+end_row_id-start_row_id));

	for (int i = len; i< len+end_row_id-start_row_id; i++){
		int test = kv_p[start_row_id+i-len].keySize;
		d_g_state->sorted_intermediate_keyvals_arr[i].keySize = kv_p[start_row_id+i-len].keySize;
		d_g_state->sorted_intermediate_keyvals_arr[i].key = kv_p[start_row_id+i-len].key;
		d_g_state->sorted_intermediate_keyvals_arr[i].vals = kv_p[start_row_id+i-len].vals;
		d_g_state->sorted_intermediate_keyvals_arr[i].val_arr_len = kv_p[start_row_id+i-len].val_arr_len;
		d_g_state->sorted_keyvals_arr_len++;
	}//for

}


__device__ void GPUEmitReduceOuput  (void*		key, 
						void*		val, 
						int		keySize, 
						int		valSize,
						gpu_context *d_g_state){
						
			keyval_t *p = &(d_g_state->d_reduced_keyval_arr[TID]);
			p->keySize = keySize;
			p->key = malloc(keySize);
			memcpy(p->key,key,keySize);
			p->valSize = valSize;
			p->val = malloc(valSize);
			memcpy(p->val,val,valSize);
			printf("[gpu output]: key:%s  val:%d\n",key,*(int *)val);
						
}//__device__ 


void CPUEmitReduceOuput  (void*		key, 
						void*		val, 
						int		keySize, 
						int		valSize,
						cpu_context *d_g_state){
						
			/*keyval_t *p = &(d_g_state->d_reduced_keyval_arr[TID]);
			p->keySize = keySize;
			p->key = malloc(keySize);
			memcpy(p->key,key,keySize);
			p->valSize = valSize;
			p->val = malloc(valSize);
			memcpy(p->val,val,valSize);*/

			printf("[cpu output]: key:%s  val:%d\n",key,*(int *)val);
						
}//__device__ 



//Last update 9/1/2012
void CPUEmitMapOutput(void *key, void *val, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx){

	if(map_task_idx >= d_g_state->num_input_record) {	DoLog("error ! map_task_idx >= d_g_state->num_input_record");		return;	}

	keyval_arr_t *kv_arr_p = &(d_g_state->intermediate_keyval_arr_arr_p[map_task_idx]);
	if (kv_arr_p->arr_len==0) kv_arr_p->arr = NULL;

	kv_arr_p->arr = (keyval_t*)realloc(kv_arr_p->arr, sizeof(keyval_t)*(kv_arr_p->arr_len+1));
	
	//printf("remain buff:%d\n", kv_arr_p->buff_len - sizeof(keyval_t)*(kv_arr_p->arr_len+1) - kv_arr_p->buff_pos);
	int current_map_output_index = (kv_arr_p->arr_len);
	keyval_t *kv_p = &(kv_arr_p->arr[current_map_output_index]);
	kv_p->key = (char *)malloc(sizeof(keySize));
	memcpy(kv_p->key,key,keySize);
	kv_p->keySize = keySize;
	
	kv_p->val = (char *)malloc(sizeof(valSize));
	memcpy(kv_p->val,val,valSize);
	kv_p->valSize = valSize;
	kv_arr_p->arr_len++;

	//printf("CPU GPUEmitMapOuput map_task_id:%d, key:%s: keyval_arr_len:%d\n", map_task_idx, kv_p->key, kv_arr_p->arr_len);

}//__device__


//Last update 9/1/2012
__device__ void GPUEmitMapOuput(void *key, void *val, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){
	
	keyval_arr_t *kv_arr_p = d_g_state->d_intermediate_keyval_arr_arr_p[map_task_idx];
	void *buff = kv_arr_p->buff;
	
	//TODO Hui Li 8/6/2012
	if (!(kv_arr_p->buff_pos +keySize+valSize < kv_arr_p->buff_len - sizeof(keyval_t)*((*kv_arr_p->total_arr_len)+1))){
		//char *p = (char *)(kv_arr_p->buff);
		kv_arr_p->buff = malloc(sizeof(char)*(kv_arr_p->buff_len*2));
		
		memcpy(kv_arr_p->buff, buff, sizeof(char)*(kv_arr_p->buff_pos));
		memcpy(kv_arr_p->buff, ((char*)buff + kv_arr_p->buff_len - sizeof(keyval_t)*(*kv_arr_p->total_arr_len)),
			sizeof(keyval_t)*(*kv_arr_p->total_arr_len));
		
		kv_arr_p->buff_len = kv_arr_p->buff_len*2;
		free(buff);
		buff = kv_arr_p->buff;
		printf("There is not engough shared memory[%d]; realloc memory for intermediate data[%d]\n",kv_arr_p->buff_len/2, kv_arr_p->buff_len);		
		//return;
	}//
	

	//printf("remain buff:%d\n", kv_arr_p->buff_len - sizeof(keyval_t)*(kv_arr_p->arr_len+1) - kv_arr_p->buff_pos);
	//int current_map_output_index = (kv_arr_p->arr_len);
	//keyval_t *kv_p = &(kv_arr_p->arr[current_map_output_index]);

	keyval_t *kv_p = (keyval_t *)((char *)buff + kv_arr_p->buff_len - sizeof(keyval_t)*((*kv_arr_p->total_arr_len)+1));
	kv_arr_p->arr = kv_p;
	(*kv_arr_p->total_arr_len)++;


	kv_p->key = (char *)(buff)+kv_arr_p->buff_pos;
	kv_arr_p->buff_pos += keySize;
	memcpy(kv_p->key,key,keySize);
	kv_p->keySize = keySize;
	
	kv_p->val = (char *)(buff)+kv_arr_p->buff_pos;
	kv_arr_p->buff_pos += valSize;
	memcpy(kv_p->val,val,valSize);
	kv_p->valSize = valSize;

	kv_arr_p->arr_len++;
	d_g_state->d_intermediate_keyval_total_count[map_task_idx] = kv_arr_p->arr_len;
	//printf("GPUEmitMapOuput TID[%d] map_task_id:%d, key%s: keyval_arr_len:%d\n",TID, map_task_idx, kv_p->key, kv_arr_p->arr_len);
}//__device__

//-------------------------------------------------
//called by user defined map function
//-------------------------------------------------

__global__ void GPUMapPartitioner(gpu_context d_g_state)
{	
	
	//DoLog("gridDim.x:%d gridDim.y:%d gridDim.z:%d blockDim.x:%d blockDim.y:%d blockDim.z:%d blockIdx.x:%d blockIdx.y:%d blockIdx.z:%d\n",
	//  gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z,blockIdx.x,blockIdx.y,blockIdx.z);
	int num_records_per_thread = (d_g_state.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_row_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_row_idx = block_start_row_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	//DoLog("num_records_per_thread:%d block_start_row_idx:%d gridDim.x:%d gridDim.y:%d gridDim.z:%d blockDim.x:%d blockDim.y:%d blockDim.z:%d\n",num_records_per_thread, block_start_row_idx, gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z);
	
	int thread_end_idx = thread_start_row_idx + num_records_per_thread*STRIDE;
	if (thread_end_idx > d_g_state.num_input_record)
		thread_end_idx = d_g_state.num_input_record;

	if (thread_start_row_idx >= thread_end_idx)
		return;

	int buff_len = 1024*100;

	char * buff = (char *)malloc(sizeof(char)*buff_len);
	int * total_arr_len = (int*)malloc(sizeof(int));
	(*total_arr_len) = 0;
	
	keyval_arr_t *kv_arr_t_arr = (keyval_arr_t *)malloc(sizeof(keyval_arr_t)*(thread_end_idx-thread_start_row_idx+STRIDE-1)/STRIDE);
	int index = 0;

	//printf("Mapper TID:%d, thread_start_row_idx:%d  thread_end_idx:%d runed tasks:%d totalTasks:%d\n",
	//	TID, thread_start_row_idx,thread_end_idx,(thread_end_idx - thread_start_row_idx)/STRIDE,d_g_state.num_input_record);

	for(int map_task_idx = thread_start_row_idx ; map_task_idx < thread_end_idx; map_task_idx += STRIDE){
		
		keyval_arr_t *kv_arr_t = (keyval_arr_t *)&(kv_arr_t_arr[index]);
		index++;

		kv_arr_t->buff = buff;
		kv_arr_t->total_arr_len = total_arr_len;

		kv_arr_t->buff_len = buff_len;
		kv_arr_t->buff_pos = 0;
		kv_arr_t->arr = NULL;
		kv_arr_t->arr_len = 0;
		
		//keyval_arr_t *kv_arr_p = &(d_g_state.d_intermediate_keyval_arr_arr[map_task_idx]);
		d_g_state.d_intermediate_keyval_arr_arr_p[map_task_idx] = kv_arr_t;
		
		/*//void *val = d_g_state.d_input_keyval_arr[map_task_idx].val;
		char *key = (char *)(d_g_state.d_input_keys_shared_buff) + d_g_state.d_input_keyval_pos_arr[map_task_idx].keyPos;
		char *val = (char *)(d_g_state.d_input_vals_shared_buff) + d_g_state.d_input_keyval_pos_arr[map_task_idx].valPos;

		int valSize = d_g_state.d_input_keyval_pos_arr[map_task_idx].valSize;
		int keySize = d_g_state.d_input_keyval_pos_arr[map_task_idx].keySize;

		//printf("map_task_idx:%d keySize:%d  valSize:%d\n",map_task_idx, keySize, valSize);
		//printf("key:%d  keySize:%d\n",*(int *)key, keySize);
		//TODO calculate the key val pair here directly. 
		///////////////////////////////////////////////////////////
		gpu_map(key, val, keySize, valSize, &d_g_state, map_task_idx);
		///////////////////////////////////////////////////////////	*/

	}//for
	//printf("task id:%d\n",TID);
	//__syncthreads();
}//GPUMapPartitioner

__global__ void RunGPUMapTasks(gpu_context d_g_state, int curIter, int totalIter)
{	
	
	//DoLog("gridDim.x:%d gridDim.y:%d gridDim.z:%d blockDim.x:%d blockDim.y:%d blockDim.z:%d blockIdx.x:%d blockIdx.y:%d blockIdx.z:%d\n",
	//  gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z,blockIdx.x,blockIdx.y,blockIdx.z);
	int num_records_per_thread = (d_g_state.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_row_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_row_idx = block_start_row_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	//DoLog("num_records_per_thread:%d block_start_row_idx:%d gridDim.x:%d gridDim.y:%d gridDim.z:%d blockDim.x:%d blockDim.y:%d blockDim.z:%d",num_records_per_thread, block_start_row_idx, gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z);
	
	int thread_end_idx = thread_start_row_idx + num_records_per_thread*STRIDE;
	if (thread_end_idx > d_g_state.num_input_record)
		thread_end_idx = d_g_state.num_input_record;

	if (thread_start_row_idx + curIter*STRIDE >= thread_end_idx)
		return;

	//printf("Mapper TID:%d, thread_start_row_idx:%d  thread_end_idx:%d runed tasks:%d totalTasks:%d\n",
	//	TID, thread_start_row_idx,thread_end_idx,(thread_end_idx - thread_start_row_idx)/STRIDE,d_g_state.num_input_record);

	for(int map_task_idx = thread_start_row_idx + curIter*STRIDE; map_task_idx < thread_end_idx; map_task_idx += totalIter*STRIDE){
		
		//void *val = d_g_state.d_input_keyval_arr[map_task_idx].val;
		char *key = (char *)(d_g_state.d_input_keys_shared_buff) + d_g_state.d_input_keyval_pos_arr[map_task_idx].keyPos;
		char *val = (char *)(d_g_state.d_input_vals_shared_buff) + d_g_state.d_input_keyval_pos_arr[map_task_idx].valPos;

		int valSize = d_g_state.d_input_keyval_pos_arr[map_task_idx].valSize;
		int keySize = d_g_state.d_input_keyval_pos_arr[map_task_idx].keySize;
		
		///////////////////////////////////////////////////////////
		gpu_map(key, val, keySize, valSize, &d_g_state, map_task_idx);
		///////////////////////////////////////////////////////////
	}//for
	//printf("task id:%d\n",TID);
	__syncthreads();
}//GPUMapPartitioner



int StartCPUMap2(thread_info_t* thread_info)
{		
	cpu_context *d_g_state = (cpu_context*)(thread_info->d_g_state);
	job_configuration *cpu_job_conf = (job_configuration*)(thread_info->job_conf);

	//DoLog("there are %d input records for map tasks.",cpu_job_conf->num_input_record);
	if (cpu_job_conf->num_input_record<=0) { DoError("Error: no any input keys"); exit(-1);}
	if (cpu_job_conf->input_keyval_arr == NULL) { DoError("Error: input_keyval_arr == NULL"); exit(-1);}
	if (d_g_state->num_cpus_cores <= 0) {	DoError("Error: d_g_state->num_cpus == 0"); exit(-1);}

	//-------------------------------------------------------
	//1, prepare buffer to store intermediate results
	//-------------------------------------------------------
	
	//DoLog("prepare buffer to store intermediate results");
	
	keyval_arr_t *d_keyval_arr_p;
	int *count = NULL;
	
	//---------------------------------------------
	//3, determine the number of threads to run
	//---------------------------------------------

	DoLog("CPU_GROUP_ID:[%d] the number of cpus used in computation:%d",d_g_state->cpu_group_id, d_g_state->num_cpus_cores);
	
	//--------------------------------------------------
	//4, start_row_id map
	//--------------------------------------------------
	
	int num_threads = d_g_state->num_cpus_cores;
	//DoLog("start_row_id CPUMapPartitioner num_threads:%d  num_input_record:%d",num_threads, cpu_job_conf->num_input_record);
	int num_records_per_thread = (cpu_job_conf->num_input_record+num_threads-1)/(num_threads);
	
	int start_row_idx = 0;
	int end_row_idx = 0;

	//pthread_t  *cpu_threads;
	//thread_info_t *cpu_threads_info;
	for (int tid = 0;tid<num_threads;tid++){
	
		end_row_idx = start_row_idx + num_records_per_thread;
		if (tid < (cpu_job_conf->num_input_record % num_threads) )
			end_row_idx++;
			
		d_g_state->panda_cpu_task_info[tid].start_row_idx = start_row_idx;
		if (end_row_idx > cpu_job_conf->num_input_record)
			end_row_idx = cpu_job_conf->num_input_record;
		d_g_state->panda_cpu_task_info[tid].end_row_idx = end_row_idx;
		
		if (pthread_create(&(d_g_state->panda_cpu_task[tid]),NULL,RunPandaCPUMapThread,(char *)&(d_g_state->panda_cpu_task_info[tid]))!=0) 
			perror("Thread creation failed!\n");
		start_row_idx = end_row_idx;
	}//for
	
	for (int tid = 0;tid<num_threads;tid++){
		void *exitstat;
		if (pthread_join(d_g_state->panda_cpu_task[tid],&exitstat)!=0) perror("joining failed");
	}//for
	
	DoLog("CPU_GROUP_ID:[%d] DONE", d_g_state->cpu_group_id);
	return 0;
}//int 



int StartCPUMap(cpu_context *d_g_state)
{		
#ifdef DEV_MODE

	DoLog("there are %d map tasks.",d_g_state->num_input_record);
	if (d_g_state->num_input_record<=0) { DoError("Error: no any input keys"); exit(-1);}
	if (d_g_state->input_keyval_arr == NULL) { DoError("Error: input_keyval_arr == NULL"); exit(-1);}
	if (d_g_state->num_cpus_cores <= 0) {	DoError("Error: d_g_state->num_cpus == 0"); exit(-1);}

	//-------------------------------------------------------
	//1, prepare buffer to store intermediate results
	//-------------------------------------------------------
	
	DoLog("prepare buffer to store intermediate results");
	
	keyval_arr_t *d_keyval_arr_p;
	int *count = NULL;
	
	//---------------------------------------------
	//3, determine the number of threads to run
	//---------------------------------------------

	DoLog("the number of cpus used in computation:%d",d_g_state->num_cpus_cores);
	
	//--------------------------------------------------
	//4, start_row_id map
	//--------------------------------------------------
	
	int num_threads = d_g_state->num_cpus_cores;

	DoLog("start_row_id CPUMapPartitioner num_threads:%d  num_input_record:%d",num_threads, d_g_state->num_input_record);
	int num_records_per_thread = (d_g_state->num_input_record+num_threads-1)/(num_threads);
	
	int start_row_idx = 0;
	int end_idx = 0;

	//pthread_t  *cpu_threads;
	//thread_info_t *cpu_threads_info;

	for (int tid = 0;tid<num_threads;tid++){
	
		end_idx = start_row_idx + num_records_per_thread;
		if (tid < (d_g_state->num_input_record % num_threads) )
			end_idx++;
			
		d_g_state->panda_cpu_task_info[tid].start_row_idx = start_row_idx;
		if (end_idx > d_g_state->num_input_record)
			end_idx = d_g_state->num_input_record;
		d_g_state->panda_cpu_task_info[tid].end_idx = end_idx;
		
		//pthread_t  *panda_cpu_task;
		//panda_cpu_task_info_t *panda_cpu_task_info;
		DoLog("tests");
		if (pthread_create(&(d_g_state->panda_cpu_task[tid]),NULL,RunPandaCPUMapThread,(char *)&(d_g_state->panda_cpu_task_info[tid]))!=0) 
			perror("Thread creation failed!\n");
		
		start_row_idx = end_idx;
	}//for

	
	for (int tid = 0;tid<num_threads;tid++){
		void *exitstat;
		if (pthread_join(d_g_state->panda_cpu_task[tid],&exitstat)!=0) perror("joining failed");
	}//for
	
	//DoLog("DONE :%d tasks current intermediate len:%d",panda_cpu_task_info->end_idx - panda_cpu_task_info->start_row_idx, d_g_state->intermediate_keyval_arr_arr_p[0].arr_len);
	DoLog("DONE");
#endif
	return 0;

}//int 

//--------------------------------------------------
//StartGPUMap
//Last Update 9/2/2012
//--------------------------------------------------

int StartGPUMap(gpu_context *d_g_state)
{		

	//-------------------------------------------------------
	//0, Check status of d_g_state;
	//-------------------------------------------------------

	DoLog("GPU_ID:[%d]  num_input_record %d", d_g_state->gpu_id, d_g_state->num_input_record);
	if (d_g_state->num_input_record<0) { DoLog("Error: no any input keys"); exit(-1);}
	if (d_g_state->h_input_keyval_arr == NULL) { DoLog("Error: h_input_keyval_arr == NULL"); exit(-1);}
	if (d_g_state->num_mappers<=0) {d_g_state->num_mappers = (NUM_BLOCKS)*(NUM_THREADS);}
	if (d_g_state->num_reducers<=0) {d_g_state->num_reducers = (NUM_BLOCKS)*(NUM_THREADS);}

	//-------------------------------------------------------
	//1, prepare buffer to store intermediate results
	//-------------------------------------------------------
	//DoLog("prepare buffer to store intermediate results");

	keyval_arr_t *h_keyval_arr_arr = (keyval_arr_t *)malloc(sizeof(keyval_arr_t)*d_g_state->num_input_record);
	keyval_arr_t *d_keyval_arr_arr;
	checkCudaErrors(cudaMalloc((void**)&(d_keyval_arr_arr),d_g_state->num_input_record*sizeof(keyval_arr_t)));
	
	for (int i=0; i<d_g_state->num_input_record;i++){
		h_keyval_arr_arr[i].arr = NULL;
		h_keyval_arr_arr[i].arr_len = 0;
	}//for
	//checkCudaErrors(cudaMemcpy(d_keyval_arr_arr, h_keyval_arr_arr, sizeof(keyval_arr_t)*d_g_state->num_input_record,cudaMemcpyHostToDevice));
	//d_g_state->d_intermediate_keyval_arr_arr = d_keyval_arr_arr;

	keyval_arr_t **d_keyval_arr_arr_p;
	checkCudaErrors(cudaMalloc((void***)&(d_keyval_arr_arr_p),d_g_state->num_input_record*sizeof(keyval_arr_t*)));
	d_g_state->d_intermediate_keyval_arr_arr_p = d_keyval_arr_arr_p;
	
	int *count = NULL;
	checkCudaErrors(cudaMalloc((void**)&(count),d_g_state->num_input_record*sizeof(int)));
	d_g_state->d_intermediate_keyval_total_count = count;
	checkCudaErrors(cudaMemset(d_g_state->d_intermediate_keyval_total_count,0,d_g_state->num_input_record*sizeof(int)));

	//----------------------------------------------
	//3, determine the number of threads to run
	//----------------------------------------------
	
	//--------------------------------------------------
	//4, start_row_id map
	//Note: DO *NOT* set large number of threads within block (512), which lead to too many invocation of malloc in the kernel. 
	//--------------------------------------------------

	cudaThreadSynchronize();
	
	int numGPUCores = getGPUCoresNum();
	dim3 blocks(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
	int numBlocks = (numGPUCores*16+(blocks.x*blocks.y)-1)/(blocks.x*blocks.y);
    dim3 grids(numBlocks, 1);
	
	//int num_blocks = (d_g_state->num_mappers+(NUM_THREADS)-1)/(NUM_THREADS);
	//GPUMapPartitioner<<<num_blocks,NUM_THREADS>>>(*d_g_state);
	int total_gpu_threads = (grids.x*grids.y*blocks.x*blocks.y);
	DoLog("GridDim.X:%d GridDim.Y:%d BlockDim.X:%d BlockDim.Y:%d TotalGPUThreads:%d",grids.x,grids.y,blocks.x,blocks.y,total_gpu_threads);

	cudaDeviceSynchronize();

	double t1 = PandaTimer();
	GPUMapPartitioner<<<grids,blocks>>>(*d_g_state);

	cudaThreadSynchronize();
	
	double t2 = PandaTimer();
	int num_records_per_thread = (d_g_state->num_input_record + (total_gpu_threads)-1)/(total_gpu_threads);
	int totalIter = num_records_per_thread;
	DoLog("GPUMapPartitioner:%f totalIter:%d",t2-t1, totalIter);
	//TODO STREAM
	for (int iter = 0; iter< totalIter; iter++){
		RunGPUMapTasks<<<grids,blocks>>>(*d_g_state, iter, totalIter);
		cudaThreadSynchronize();
		double t3 = PandaTimer();
		size_t total_mem,avail_mem;
		checkCudaErrors(cudaMemGetInfo( &avail_mem, &total_mem ));
		cudaDeviceSynchronize();
		DoLog("GPU_ID:[%d] RunGPUMapTasks take %f sec at iter [%d/%d] remain %d MB GPU mem",d_g_state->gpu_id, t3-t2,iter,totalIter, avail_mem/1024/1024);
		t2 = t3;
	}//for
	DoLog("GPU_ID:[%d] Done %d Tasks",d_g_state->gpu_id,d_g_state->num_input_record);
	return 0;
}//int 




void DestroyDGlobalState(gpu_context * d_g_state){
	
}//void 


void StartGPUShuffle(gpu_context * state){
	DoLog("GPU_ID:[%d] GPU Shuffle", state->gpu_id);
	gpu_context* d_g_state = state;

	Shuffle4GPUOutput(d_g_state);

	//DoLog("DONE");
}//void

void *RunPandaCPUMapThread(void *ptr){
		
	panda_cpu_task_info_t *panda_cpu_task_info = (panda_cpu_task_info_t *)ptr;
	cpu_context *d_g_state = (cpu_context *)(panda_cpu_task_info->d_g_state); 
	job_configuration *cpu_job_conf = (job_configuration *)(panda_cpu_task_info->cpu_job_conf); 

	//DoLog("panda_cpu_task_info_t start_row_idx:%d end_idx:%d",panda_cpu_task_info->start_row_idx, panda_cpu_task_info->end_idx);
	for (int map_idx = panda_cpu_task_info->start_row_idx; map_idx < panda_cpu_task_info->end_row_idx; map_idx++){
		keyval_t *kv_p = (keyval_t *)(&(cpu_job_conf->input_keyval_arr[map_idx]));
		//void cpu_map(void *KEY, void*VAL, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx){
		cpu_map(kv_p->key,kv_p->val,kv_p->keySize,kv_p->valSize,d_g_state,map_idx);
		//DoLog("finished map_task:%d at tid:%d",d_g_state->end_idx - d_g_state->start_row_idx, d_g_state->tid);
	}//for
	
	DoLog("CPU_GROUP_ID:[%d] Done :%d tasks",d_g_state->cpu_group_id, panda_cpu_task_info->end_row_idx - panda_cpu_task_info->start_row_idx);
	return NULL;
}

//Use Pthread to process Panda_Reduce
void * Panda_Reduce(void *ptr){
		
	thread_info_t *thread_info = (thread_info_t *)ptr;
		
	if(thread_info->device_type == GPU_ACC){
		
		int tid = thread_info->tid;
		int num_gpus = 1;//thread_info->num_gpus;
		
		if (num_gpus == 0){
			DoLog("thread_info->num_gpus == 0 return");
			return NULL;
		}//if
		
		cudaSetDevice(tid % num_gpus);        // "% num_gpus" allows more CPU threads than GPU devices
		int gpu_id;
		cudaGetDevice(&gpu_id);
		gpu_context *d_g_state = (gpu_context *)(thread_info->d_g_state);
		//DoLog( "start_row_id reduce tasks on GPU:%d tid:%d",gpu_id, tid);
		StartGPUReduce(d_g_state);
		}//if
		
	if(thread_info->device_type == CPU_ACC){
		
		cpu_context *d_g_state = (cpu_context *)(thread_info->d_g_state);
		DoLog("Start CPU Reduce Tasks");
		
		if (d_g_state->num_cpus_cores == 0){
			DoLog("d_g_state->num_cpus == 0 return");
			return NULL;
		}

		//StartCPUReduce(d_g_state);
		//panda_cpu_task_info_t *panda_cpu_task_info = (panda_cpu_task_info_t *)ptr;
		//cpu_context *d_g_state = (cpu_context *)(panda_cpu_task_info->d_g_state); 
		//DoLog("panda_cpu_task_info_t start_row_idx:%d end_idx:%d",panda_cpu_task_info->start_row_idx, panda_cpu_task_info->end_idx);

		for (int map_idx = 0; map_idx < d_g_state->sorted_keyvals_arr_len; map_idx++){

		keyvals_t *kv_p = (keyvals_t *)(&(d_g_state->sorted_intermediate_keyvals_arr[map_idx]));
		cpu_reduce(kv_p->key, kv_p->vals, kv_p->keySize, kv_p->val_arr_len, d_g_state);

		}//for
		DoLog("DONE");
		
	}//if	
	
	//cudaFree(d_filebuf);
	//handle the buffer different
	//free(h_filebuf);
	//handle the buffer different
	return NULL;
}//void


__device__ void *GetVal(void *vals, int4* interOffsetSizes, int keyIndex, int valStartIndex)
{
}

__device__ void *GetKey(void *key, int4* interOffsetSizes, int keyIndex, int valStartIndex)
{
}

//-------------------------------------------------------
//Reducer
//-------------------------------------------------------



__global__ void ReducePartitioner(gpu_context d_g_state)
{
	int num_records_per_thread = (d_g_state.d_sorted_keyvals_arr_len+(gridDim.x*blockDim.x)-1)/(gridDim.x*blockDim.x);
	int block_start_row_idx = num_records_per_thread*blockIdx.x*blockDim.x;

	int thread_start_row_idx = block_start_row_idx 
		+ (threadIdx.x/STRIDE)*num_records_per_thread*STRIDE
		+ (threadIdx.x%STRIDE);

	int thread_end_idx = thread_start_row_idx+num_records_per_thread*STRIDE;
	if(thread_end_idx>d_g_state.d_sorted_keyvals_arr_len)
		thread_end_idx = d_g_state.d_sorted_keyvals_arr_len;

	//printf("ReducePartitioner: TID:%d  start_row_idx:%d  end_idx:%d d_sorted_keyvals_arr_len:%d\n",TID,thread_start_row_idx,thread_end_idx,d_g_state.d_sorted_keyvals_arr_len);

	int start_row_id, end;
	for(int reduce_task_idx=thread_start_row_idx; reduce_task_idx < thread_end_idx; reduce_task_idx+=STRIDE){
		if (reduce_task_idx==0)
			start_row_id = 0;
		else
			start_row_id = d_g_state.d_pos_arr_4_sorted_keyval_pos_arr[reduce_task_idx-1];
		end = d_g_state.d_pos_arr_4_sorted_keyval_pos_arr[reduce_task_idx];

		val_t *val_t_arr = (val_t*)malloc(sizeof(val_t)*(end-start_row_id));
		//assert(val_t_arr!=NULL);
		int keySize = d_g_state.d_keyval_pos_arr[start_row_id].keySize;
		int keyPos = d_g_state.d_keyval_pos_arr[start_row_id].keyPos;
		void *key = (char*)d_g_state.d_sorted_keys_shared_buff+keyPos;
		//printf("keySize;%d keyPos:%d key:%s\n",keySize,keyPos,key);
		//printf("reduce_task_idx:%d		keyPos:%d,  keySize:%d, key:% start_row_id:%d end:%d\n",reduce_task_idx,keyPos,keySize,start_row_id,end);
		//printf("start_row_id:%d end:%d\n",start_row_id,end);
		
		
		for (int index = start_row_id;index<end;index++){
			int valSize = d_g_state.d_keyval_pos_arr[index].valSize;
			int valPos = d_g_state.d_keyval_pos_arr[index].valPos;
			val_t_arr[index-start_row_id].valSize = valSize;
			val_t_arr[index-start_row_id].val = (char*)d_g_state.d_sorted_vals_shared_buff + valPos;
		}   //for
		gpu_reduce(key, val_t_arr, keySize, end-start_row_id, d_g_state);

	}//for
}



//----------------------------------------------
//start_row_id reduce
//
//1, if there is not a reduce phase, just return
//   then user uses spec->interKeys/spec->intervals 
//   for further processing
//2, get reduce input data on host
//3, upload reduce input data onto device memory
//4, determine the number of threads to run
//5, calculate output data keys'buf size 
//	 and values' buf size
//6, do prefix sum on--
//	 i)		d_outputKeysSizePerTask
//	 ii)	d_outputValsSizePerTask
//	 iii)	d_outputCountPerTask
//7, allocate output memory on device memory
//8, start_row_id reduce
//9, copy output data to Spect_t structure
//10,free allocated memory
//----------------------------------------------
		
void StartGPUReduce(gpu_context *d_g_state)
{	
	cudaThreadSynchronize(); 
	d_g_state->d_reduced_keyval_arr_len = d_g_state->d_sorted_keyvals_arr_len;
	checkCudaErrors(cudaMalloc((void **)&(d_g_state->d_reduced_keyval_arr), sizeof(keyval_t)*d_g_state->d_reduced_keyval_arr_len));
	
	DoLog("number of reduce tasks:%d",d_g_state->d_sorted_keyvals_arr_len);
	cudaThreadSynchronize(); 
	

	int num_blocks = (d_g_state->num_reducers+(NUM_THREADS)-1)/(NUM_THREADS);
	DoLog("num_blocks:%d NUM_THREADS:%d",num_blocks,NUM_THREADS);
	ReducePartitioner<<<num_blocks,NUM_THREADS>>>(*d_g_state);
	cudaThreadSynchronize(); 

	DoLog("DONE");
}//void


void* Panda_Map(void *ptr){
		
	//DoLog("panda_map");
	thread_info_t *thread_info = (thread_info_t *)ptr;
		
	if(thread_info->device_type == GPU_ACC){
		
		gpu_context *d_g_state = (gpu_context *)(thread_info->d_g_state);
		//DoLog("gpu_id in current Panda_Map:%d",gpu_id);
		InitGPUDevice(thread_info);
		
		DoLog("GPU_ID:[%d] Init GPU MapReduce Load Data From Host to GPU memory",d_g_state->gpu_id);
		InitGPUMapReduce3(d_g_state);
		

		DoLog("GPU_ID:[%d] Start GPU Map Tasks",d_g_state->gpu_id);
		StartGPUMap(d_g_state);
		
		//d_g_state->d_intermediate_keyval_total_count;
		//checkCudaErrors(cudaMemset(d_g_state->d_intermediate_keyval_total_count,0,d_g_state->num_input_record*sizeof(int)));
		
		StartGPUShuffle(d_g_state);
		
	}//if
		
	if(thread_info->device_type == CPU_ACC){

		//DoLog("CPU_ACC");
		cpu_context *d_g_state = (cpu_context *)(thread_info->d_g_state);
		DoLog("CPU_GROUP_ID:[%d] Init CPU Device",d_g_state->cpu_group_id);
		InitCPUDevice(thread_info);
		
		//DoLog("Init CPU MapReduce");
		InitCPUMapReduce2(thread_info);

		DoLog("CPU_GROUP_ID:[%d] Start CPU Map Tasks",d_g_state->cpu_group_id);
		StartCPUMap2(thread_info);
		//d_g_state->d_intermediate_keyval_total_count;
		//checkCudaErrors(cudaMemset(d_g_state->d_intermediate_keyval_total_count,0,d_g_state->num_input_record*sizeof(int)));
		StartCPUShuffle2(thread_info);
		
	}	
			
	return NULL;
}//FinishMapReduce2(d_g_state);


void FinishMapReduce(Spec_t* spec)
{
	DoLog( "=====finish panda mapreduce=====");
}//void


void FinishMapReduce2(gpu_context* state)
{

	size_t total_mem,avail_mem, heap_limit;
	checkCudaErrors(cudaMemGetInfo( &avail_mem, &total_mem ));
	DoLog("avail_mem:%d",avail_mem);

}//void


#endif //__PANDALIB_CU__