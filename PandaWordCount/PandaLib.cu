/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: PandaLib.cu 
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2012-09-016

	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */

#ifndef __PANDALIB_CU__
#define __PANDALIB_CU__

#include "Panda.h"
#include "UserAPI.cu"


//----------------------------------------------
//Get default job configuration
//----------------------------------------------
job_configuration *CreateJobConf(){

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


gpu_context *CreateGPUContext(){
	gpu_context *d_g_state = (gpu_context*)malloc(sizeof(gpu_context));
	if (d_g_state == NULL) exit(-1);
	memset(d_g_state, 0, sizeof(gpu_context));
	d_g_state->configured = false;
	d_g_state->h_input_keyval_arr = NULL;
	d_g_state->num_mappers = 0;
	d_g_state->num_reducers = 0;
	return d_g_state;
}//gpu_context
			 
cpu_context *CreateCPUContext(){
	cpu_context *d_g_state = (cpu_context*)malloc(sizeof(cpu_context));
	if (d_g_state == NULL) exit(-1);
	memset(d_g_state, 0, sizeof(cpu_context));
	d_g_state->configured = false;
	d_g_state->input_keyval_arr = NULL;
	return d_g_state;
}//gpu_context

panda_context *CreatePandaContext(){
	
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

	if (job_conf->num_input_record<=0) { ShowError("Error: no any input keys"); exit(-1);}
	if (job_conf->input_keyval_arr == NULL) { ShowError("Error: input_keyval_arr == NULL"); exit(-1);}
	if (d_g_state->num_cpus_cores <= 0) {	ShowError("Error: d_g_state->num_cpus == 0"); exit(-1);}

	//ShowLog("d_g_state->configured:%s  enable for iterative applications",d_g_state->configured? "true" : "false");
	//if (d_g_state->configured)
	//	return;
	
	int totalKeySize = 0;
	int totalValSize = 0;

	for(int i=0;i<job_conf->num_input_record;i++){
		totalKeySize += job_conf->input_keyval_arr[i].keySize;
		totalValSize += job_conf->input_keyval_arr[i].valSize;
	}//for

	ShowLog("CPU_GROUP_ID:[%d] num_input_record:%d, totalKeySize:%d totalValSize:%d num_cpus:%d", 
		d_g_state->cpu_group_id, job_conf->num_input_record, totalKeySize, totalValSize, d_g_state->num_cpus_cores);

	//TODO determin num_cpus
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
	ShowLog("CPU_GROUP_ID:[%d] DONE",d_g_state->cpu_group_id);

}


#ifdef DEV_MODE
//For Version 0.3 test depressed
void InitGPUMapReduce4(thread_info_t* thread_info)
{	
	gpu_context *d_g_state = (gpu_context *)(thread_info->d_g_state);
	job_configuration* gpu_job_conf = (job_configuration*)(thread_info->job_conf);
	keyval_t * kv_p = gpu_job_conf->input_keyval_arr;

	ShowLog("d_g_state->configured:%s  enable for iterative applications",d_g_state->configured? "true" : "false");
	//if (d_g_state->configured)
	//	return;
	ShowLog("copy %d input records from Host to GPU memory",gpu_job_conf->num_input_record);
	//checkCudaErrors(cudaMalloc((void **)&d_g_state->d_input_keyval_arr,sizeof(keyval_t)*d_g_state->num_input_record));
	int totalKeySize = 0;
	int totalValSize = 0;
	for(int i=0;i<gpu_job_conf->num_input_record;i++){
		totalKeySize += kv_p[i].keySize;
		totalValSize += kv_p[i].valSize;
	}//for
	ShowLog("totalKeySize:%d totalValSize:%d", totalKeySize, totalValSize);
	
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

	ShowLog("d_g_state->configured:%s  enable for iterative applications",d_g_state->configured? "true" : "false");
	//if (d_g_state->configured)
	//	return;
	
	int totalKeySize = 0;
	int totalValSize = 0;
	for(int i=0;i<d_g_state->num_input_record;i++){
		totalKeySize += d_g_state->h_input_keyval_arr[i].keySize;
		totalValSize += d_g_state->h_input_keyval_arr[i].valSize;
	}//for
	ShowLog("GPU_ID:[%d] copy %d input records from Host to GPU memory totalKeySize:%d totalValSize:%d",d_g_state->gpu_id, d_g_state->num_input_record, totalKeySize, totalValSize);
	
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
	
	ShowLog("d_g_state->num_input_record:%d",d_g_state->num_input_record);
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
	ShowLog( "CPU_GROUP_ID:[%d] Init CPU Deivce",d_g_state->cpu_group_id);
	
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
		ShowError("error num_gpus == 0");
		exit(-1);
	}//gpu_context
	
	int gpu_id;
	cudaGetDevice(&gpu_id);
	int gpu_count = 0;
	cudaGetDeviceCount(&gpu_count);

	cudaDeviceProp gpu_dev;
	cudaGetDeviceProperties(&gpu_dev, gpu_id);

	ShowLog("TID:[%d] check GPU ids: cur_gpu_id:[%d] assig_gpu_id:[%d] cudaGetDeviceCount:[%d] GPU name:%s", 
		tid, gpu_id, assigned_gpu_id,  gpu_count, gpu_dev.name);

	if ( gpu_id != assigned_gpu_id ){
		//ShowLog("cudaSetDevice gpu_id %d == (tid num_gpus) %d ", gpu_id, tid%num_gpus);
		cudaSetDevice(assigned_gpu_id % num_gpus);  
	}//if
		
	size_t total_mem,avail_mem, heap_limit;
	checkCudaErrors(cudaMemGetInfo( &avail_mem, &total_mem ));
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, (int)(total_mem*0.8)); 
	cudaDeviceGetLimit(&heap_limit, cudaLimitMallocHeapSize);

	int numGPUCores = getGPUCoresNum();
	ShowLog("GPU_ID:[%d] numGPUCores:%d cudaLimitMallocHeapSize:%d MB avail_mem:%d MB total_mem:%d MB",
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
	
	int total_count = 0;
	for(int i=start_row_id;i<end_row_id;i++){
		total_count += sorted_intermediate_keyvals_arr[i].val_arr_len;
	}//for
	
	int totalKeySize = 0;
	int totalValSize = 0;
	for(int i=start_row_id;i<end_row_id;i++){
		totalKeySize += (sorted_intermediate_keyvals_arr[i].keySize+3)/4*4;
		for (int j=0;j<sorted_intermediate_keyvals_arr[i].val_arr_len;j++)
		totalValSize += (sorted_intermediate_keyvals_arr[i].vals[j].valSize+3)/4*4;
	}//for
		
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
	
	ShowLog("GPU_ID:[%d] total #different intermediate records:%d total records:%d totalKeySize:%d KB totalValSize:%d KB", 
		d_g_state->gpu_id, end_row_id - start_row_id, total_count, totalKeySize/1024, totalValSize/1024);

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
			valPos += (p->vals[j].valSize+3)/4*4;
			index++;
		}//for
		keyPos += (p->keySize+3)/4*4;
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

	if (end_row_id<=start_row_id) {	ShowError("error! end_row_id<=start_row_id");		return;	}
	int len = d_g_state->num_input_record;
	if (len<0) {	ShowError("error! len<0");		return;	}
	if (len == 0) d_g_state->h_input_keyval_arr = NULL;

	ShowLog("GPU_ID:[%d] add map tasks into gpu; #total input:%d #added input:%d",d_g_state->gpu_id, len, end_row_id-start_row_id);			

	d_g_state->h_input_keyval_arr = (keyval_t *)realloc(d_g_state->h_input_keyval_arr, sizeof(keyval_t)*(len + end_row_id - start_row_id));
	//assert(d_g_state->h_input_keyval_arr != NULL);
	for (int i=start_row_id;i<end_row_id;i++){

	d_g_state->h_input_keyval_arr[len].keySize = kv_p[i].keySize;
	d_g_state->h_input_keyval_arr[len].valSize = kv_p[i].valSize;
	d_g_state->h_input_keyval_arr[len].key = kv_p[i].key;
	d_g_state->h_input_keyval_arr[len].val = kv_p[i].val;
	d_g_state->num_input_record++;
	len++;

	}
}


void AddMapInputRecordCPU(cpu_context* d_g_state,
						keyval_t *kv_p, int start_row_id, int end_row_id){

	if (end_row_id<=start_row_id) {	ShowError("error! end_row_id[%d] <= start_row_id[%d]",end_row_id, start_row_id);		return;	}	
	int len = d_g_state->num_input_record;
	if (len<0) {	ShowError("error! len<0");		return;	}
	if (len == 0) d_g_state->input_keyval_arr = NULL;

	ShowLog("CPU_GROUP_ID:[%d] add map input record for cpu device current #input:%d added #input:%d",d_g_state->cpu_group_id,len,end_row_id-start_row_id);			
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
							
    if (end_row_id<start_row_id){	ShowError("error! end_row_id<=start_row_id");		return;	}
	int len = d_g_state->sorted_keyvals_arr_len;
	if (len<0) {	ShowError("error! len<0");		return;	}
	if (len == 0) d_g_state->sorted_intermediate_keyvals_arr = NULL;

	d_g_state->sorted_intermediate_keyvals_arr = (keyvals_t *)malloc(sizeof(keyvals_t)*(len+end_row_id-start_row_id));
	
	for (int i = len; i< len+end_row_id-start_row_id; i++){
		d_g_state->sorted_intermediate_keyvals_arr[i].keySize = kv_p[start_row_id+i-len].keySize;
		d_g_state->sorted_intermediate_keyvals_arr[i].key = kv_p[start_row_id+i-len].key;
		d_g_state->sorted_intermediate_keyvals_arr[i].vals = kv_p[start_row_id+i-len].vals;
		d_g_state->sorted_intermediate_keyvals_arr[i].val_arr_len = kv_p[start_row_id+i-len].val_arr_len;

	}//for
	d_g_state->sorted_keyvals_arr_len = end_row_id-start_row_id;
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


void CPUEmitReduceOutput  (void*		key, 
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

	if(map_task_idx >= d_g_state->num_input_record) {	ShowLog("error ! map_task_idx >= d_g_state->num_input_record");		return;	}

	keyval_arr_t *kv_arr_p = &(d_g_state->intermediate_keyval_arr_arr_p[map_task_idx]);
	if (kv_arr_p->arr_len==0) kv_arr_p->arr = NULL;

	kv_arr_p->cpu_arr = (keyval_t*)realloc(kv_arr_p->arr, sizeof(keyval_t)*(kv_arr_p->arr_len+1));
	
	int current_map_output_index = (kv_arr_p->arr_len);
	keyval_t *kv_p = &(kv_arr_p->cpu_arr[current_map_output_index]);
	kv_p->key = (char *)malloc(sizeof(keySize));
	memcpy(kv_p->key,key,keySize);
	kv_p->keySize = keySize;
	
	kv_p->val = (char *)malloc(sizeof(valSize));
	memcpy(kv_p->val,val,valSize);
	kv_p->valSize = valSize;
	kv_arr_p->arr_len++;

}//__device__


__device__ void GPUEmitCombinerOuput(void *key, void *val, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){
			
	keyval_arr_t *kv_arr_p = d_g_state->d_intermediate_keyval_arr_arr_p[map_task_idx];
	void *shared_buff = kv_arr_p->shared_buff;	
	int shared_buff_len = *kv_arr_p->shared_buff_len;
	int shared_arr_len = *kv_arr_p->shared_arr_len;
	int shared_buff_pos = *kv_arr_p->shared_buff_pos;
		
	int required_mem_len = (shared_buff_pos) + keySize + valSize + sizeof(keyval_pos_t)*(shared_arr_len+1);
	if (required_mem_len> shared_buff_len){

	ShowWarn("Warning! no enough memory in GPU task:%d need:%d KB KeySize:%d ValSize:%d shared_arr_len:%d shared_buff_pos:%d shared_buff_len:%d",
			map_task_idx, required_mem_len/1024,keySize,valSize,shared_arr_len,shared_buff_pos,shared_buff_len);
		
	char *new_buff = (char*)malloc(sizeof(char)*((*kv_arr_p->shared_buff_len)*2));
	if(new_buff==NULL)ShowLog("Error! There is not enough memory to allocat!\n");

	memcpy(new_buff, shared_buff, sizeof(char)*(*kv_arr_p->shared_buff_pos));
	memcpy(new_buff + (*kv_arr_p->shared_buff_len)*2 - sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len), 
			(char*)shared_buff + (*kv_arr_p->shared_buff_len) - sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len),
												sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len));
		
	shared_buff_len = 2*(*kv_arr_p->shared_buff_len);
	(*kv_arr_p->shared_buff_len) = shared_buff_len;	
		
	for(int  idx = 0; idx < (kv_arr_p->shared_buddy_len); idx++){

		int cur_map_task_idx = kv_arr_p->shared_buddy[idx];  //the buddy relationship won't be changed 
		keyval_arr_t *cur_kv_arr_p = d_g_state->d_intermediate_keyval_arr_arr_p[cur_map_task_idx];
		cur_kv_arr_p->shared_buff = new_buff;
		
	}//for
	free(shared_buff);
	shared_buff = new_buff;
	
	}//if

	keyval_pos_t *kv_p = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len+1));
	kv_p->keySize = keySize;
	kv_p->valSize = valSize;
	kv_p->task_idx = map_task_idx;
	kv_p->next_idx = -2;			//merged results

	memcpy( (char*)shared_buff + *kv_arr_p->shared_buff_pos, key, keySize);
	kv_p->keyPos = *kv_arr_p->shared_buff_pos;
	*kv_arr_p->shared_buff_pos += (keySize+3)/4*4;

	memcpy( (char*)shared_buff + *kv_arr_p->shared_buff_pos, val, valSize);
	kv_p->valPos = *kv_arr_p->shared_buff_pos;
	*kv_arr_p->shared_buff_pos += (valSize+3)/4*4;

	(*kv_arr_p->shared_arr_len)++;
			
}//__device__


//Last update 9/16/2012
__device__ void GPUEmitMapOutput(void *key, void *val, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){

	keyval_arr_t *kv_arr_p = d_g_state->d_intermediate_keyval_arr_arr_p[map_task_idx];
	char *buff = (char*)(kv_arr_p->shared_buff);
	
	if (!((*kv_arr_p->shared_buff_pos) + keySize + valSize < (*kv_arr_p->shared_buff_len) - sizeof(keyval_pos_t)*((*kv_arr_p->shared_arr_len)+1))){
		
		ShowWarn("Warning! not enough memory at GPU task:%d *kv_arr_p->shared_arr_len:%d current buff_size:%d KB",
			map_task_idx,*kv_arr_p->shared_arr_len,(*kv_arr_p->shared_buff_len)/1024);
		
		char *new_buff = (char*)malloc(sizeof(char)*((*kv_arr_p->shared_buff_len)*2));
		if(new_buff==NULL){ ShowError("Error ! There is not enough memory to allocat!"); return; }

		memcpy(new_buff, buff, sizeof(char)*(*kv_arr_p->shared_buff_pos));
		memcpy(new_buff + (*kv_arr_p->shared_buff_len)*2 - sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len), 
			(char*)buff + (*kv_arr_p->shared_buff_len) - sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len),
														sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len));
		
		(*kv_arr_p->shared_buff_len) = 2*(*kv_arr_p->shared_buff_len);
			
		for(int  idx = 0; idx < (kv_arr_p->shared_buddy_len); idx++){

			int cur_map_task_idx = kv_arr_p->shared_buddy[idx];  //the buddy relationship won't be changed 
			keyval_arr_t *cur_kv_arr_p = d_g_state->d_intermediate_keyval_arr_arr_p[cur_map_task_idx];
			cur_kv_arr_p->shared_buff = new_buff;
	
		}//for
		free(buff);//?????
		buff = new_buff;
	}//if
	
	keyval_pos_t *kv_p = (keyval_pos_t *)((char *)buff + *kv_arr_p->shared_buff_len - sizeof(keyval_pos_t)*((*kv_arr_p->shared_arr_len)+1));
	(*kv_arr_p->shared_arr_len)++;
	kv_p->task_idx = map_task_idx;
	kv_p->next_idx = -1;

	kv_p->keyPos = (*kv_arr_p->shared_buff_pos);
	*kv_arr_p->shared_buff_pos += ((keySize+3)/4)*4;		//alignment 4 bytes for reading and writing
	memcpy((char *)(buff) + kv_p->keyPos,key,keySize);
	kv_p->keySize = keySize;
	
	kv_p->valPos = (*kv_arr_p->shared_buff_pos);
	*kv_arr_p->shared_buff_pos += ((valSize+3)/4)*4;
	char *val_p = (char *)(buff) + kv_p->valPos;
	memcpy((char *)(buff) + kv_p->valPos, val, valSize);
	kv_p->valSize = valSize;

	(kv_arr_p->arr) = kv_p;
	//kv_arr_p->arr_len++;
	//d_g_state->d_intermediate_keyval_total_count[map_task_idx] = kv_arr_p->arr_len;
	
}//__device__

//-------------------------------------------------
//called by user defined map function
//-------------------------------------------------
//need update copydata1<<<???
//TODO  9/11/2012  merge threads and blocks code into the same place. 

__global__ void GPUMapPartitioner(gpu_context d_g_state)
{	
	
	//ShowLog("gridDim.x:%d gridDim.y:%d gridDim.z:%d blockDim.x:%d blockDim.y:%d blockDim.z:%d blockIdx.x:%d blockIdx.y:%d blockIdx.z:%d\n",
	//  gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z,blockIdx.x,blockIdx.y,blockIdx.z);
	int num_records_per_thread = (d_g_state.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx + num_records_per_thread*STRIDE;
	if (thread_end_idx > d_g_state.num_input_record)
		thread_end_idx = d_g_state.num_input_record;

	if (thread_start_idx >= thread_end_idx)
		return;
		
	
	int buddy_arr_len = num_records_per_thread;
	int * int_arr = (int*)malloc((4+buddy_arr_len)*sizeof(int));
	if(int_arr==NULL){ ShowError("there is not enough GPU memory\n"); return;}

	int *shared_arr_len = int_arr;
	int *shared_buff_len = int_arr+1;
	int *shared_buff_pos = int_arr+2;
	int *num_buddy = int_arr+3;
	int *buddy = int_arr+4;
			
	(*shared_buff_len) = SHARED_BUFF_LEN;
	(*shared_arr_len) = 0;
	(*shared_buff_pos) = 0;

	char * buff = (char *)malloc(sizeof(char)*(*shared_buff_len));
	keyval_arr_t *kv_arr_t_arr = (keyval_arr_t *)malloc(sizeof(keyval_arr_t)*(thread_end_idx-thread_start_idx+STRIDE-1)/STRIDE);
	int index = 0;
	
	for(int idx = thread_start_idx; idx < thread_end_idx; idx += STRIDE){
			buddy[index] = idx;
			index ++;
	}//for
	index = 0;

	for(int map_task_idx = thread_start_idx; map_task_idx < thread_end_idx; map_task_idx += STRIDE){

		keyval_arr_t *kv_arr_t = (keyval_arr_t *)&(kv_arr_t_arr[index]);
		index++;
		kv_arr_t->shared_buff = buff;
		kv_arr_t->shared_arr_len = shared_arr_len;
		kv_arr_t->shared_buff_len = shared_buff_len;
		kv_arr_t->shared_buff_pos = shared_buff_pos;
		kv_arr_t->shared_buddy = buddy;
		kv_arr_t->shared_buddy_len = buddy_arr_len;
		kv_arr_t->arr = NULL;
		kv_arr_t->arr_len = 0;
		
		d_g_state.d_intermediate_keyval_arr_arr_p[map_task_idx] = kv_arr_t;

	}//for

}//GPUMapPartitioner

__global__ void RunGPUMapTasks(gpu_context d_g_state, int curIter, int totalIter)
{	
		
	//ShowLog("gridDim.x:%d gridDim.y:%d gridDim.z:%d blockDim.x:%d blockDim.y:%d blockDim.z:%d blockIdx.x:%d blockIdx.y:%d blockIdx.z:%d\n",
	//  gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z,blockIdx.x,blockIdx.y,blockIdx.z);
	int num_records_per_thread = (d_g_state.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	//ShowLog("num_records_per_thread:%d block_start_idx:%d gridDim.x:%d gridDim.y:%d gridDim.z:%d blockDim.x:%d blockDim.y:%d blockDim.z:%d",num_records_per_thread, block_start_idx, gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z);
	int thread_end_idx = thread_start_idx + num_records_per_thread*STRIDE;
	if (thread_end_idx > d_g_state.num_input_record)
		thread_end_idx = d_g_state.num_input_record;
		
	if (thread_start_idx + curIter*STRIDE >= thread_end_idx)
		return;
	
	for(int map_task_idx = thread_start_idx + curIter*STRIDE; map_task_idx < thread_end_idx; map_task_idx += totalIter*STRIDE){
		
		char *key = (char *)(d_g_state.d_input_keys_shared_buff) + d_g_state.d_input_keyval_pos_arr[map_task_idx].keyPos;
		char *val = (char *)(d_g_state.d_input_vals_shared_buff) + d_g_state.d_input_keyval_pos_arr[map_task_idx].valPos;

		int valSize = d_g_state.d_input_keyval_pos_arr[map_task_idx].valSize;
		int keySize = d_g_state.d_input_keyval_pos_arr[map_task_idx].keySize;
		
		////////////////////////////////////////////////////////////////
		gpu_map(key, val, keySize, valSize, &d_g_state, map_task_idx);//
		////////////////////////////////////////////////////////////////

	}//for

	keyval_arr_t *kv_arr_p = d_g_state.d_intermediate_keyval_arr_arr_p[thread_start_idx];
	//char *shared_buff = (char *)(kv_arr_p->shared_buff);
	//int shared_arr_len = *kv_arr_p->shared_arr_len;
	//int shared_buff_len = *kv_arr_p->shared_buff_len;
	d_g_state.d_intermediate_keyval_total_count[thread_start_idx] = *kv_arr_p->shared_arr_len;

	//__syncthreads();
}//GPUMapPartitioner


//NOTE: gpu_combiner will affect the later program results
//Last update 9/16/2012

__global__ void GPUCombiner(gpu_context d_g_state)
{	

	//ShowLog("gridDim.x:%d gridDim.y:%d gridDim.z:%d blockDim.x:%d blockDim.y:%d blockDim.z:%d blockIdx.x:%d blockIdx.y:%d blockIdx.z:%d\n",
	//  gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z,blockIdx.x,blockIdx.y,blockIdx.z);
	
	int num_records_per_thread = (d_g_state.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx + num_records_per_thread*STRIDE;
	if (thread_end_idx > d_g_state.num_input_record)
		thread_end_idx = d_g_state.num_input_record;

	if (thread_start_idx >= thread_end_idx)
		return;

	keyval_arr_t *kv_arr_p = d_g_state.d_intermediate_keyval_arr_arr_p[thread_start_idx];

	int *buddy = kv_arr_p->shared_buddy;
	
	//TODO use host function set 
	/*for (int idx=0;idx<kv_arr_p->shared_buddy_len;idx++){
		d_g_state.d_intermediate_keyval_total_count[idx] = 0;
	}*/

	int unmerged_shared_arr_len = *kv_arr_p->shared_arr_len;

	int num_keyval_pairs_after_combiner = 0;
	for (int i=0; i<unmerged_shared_arr_len;i++){
		
		void *shared_buff = kv_arr_p->shared_buff;	
		int shared_buff_len = *kv_arr_p->shared_buff_len;

		keyval_pos_t *head_kv_p = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(unmerged_shared_arr_len-i));
		keyval_pos_t *first_kv_p = head_kv_p;

		if (first_kv_p->next_idx != -1)
			continue;

		int iKeySize = first_kv_p->keySize;
		char *iKey = (char*)shared_buff + first_kv_p->keyPos;
		char *iVal = (char*)shared_buff + first_kv_p->valPos;

		if((first_kv_p->keyPos%4!=0)||(first_kv_p->valPos%4!=0)){
			ShowError("keyPos or valPos is not aligned with 4 bytes, results could be wrong\n");
		}

		int valCount = 1;
		for (int j=i+1;j<unmerged_shared_arr_len;j++){
			keyval_pos_t *next_kv_p = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(unmerged_shared_arr_len-j));
			char *jKey = (char *)shared_buff+next_kv_p->keyPos;
			int jKeySize = next_kv_p->keySize;
		
			if (gpu_compare(iKey,iKeySize,jKey,jKeySize)!=0){
				continue;
			}
			valCount++;
			first_kv_p->next_idx = j;
			first_kv_p = next_kv_p;
		}//for
				
		val_t *val_t_arr = (val_t *)malloc(sizeof(val_t)*valCount);
		int index = 0;
		first_kv_p = head_kv_p;

		(val_t_arr[index]).valSize = first_kv_p->valSize;
		(val_t_arr[index]).val = (char*)shared_buff + first_kv_p->valPos;

		for (int j=i+1;j<unmerged_shared_arr_len;j++){

			keyval_pos_t *next_kv_p = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(unmerged_shared_arr_len-j));
			char *jKey = (char *)shared_buff+next_kv_p->keyPos;
			int jKeySize = next_kv_p->keySize;
		
			if (gpu_compare(iKey,iKeySize,jKey,jKeySize)!=0){
				//printf("not match iKey:%s jKey:%s\n",iKey,jKey);
				continue;
			}
			index++;
			(val_t_arr[index]).valSize = next_kv_p->valSize;
			(val_t_arr[index]).val = (char*)shared_buff + next_kv_p->valPos;
		}

		//do {
			//(val_t_arr[index]).valSize = first_kv_p->valSize;
			//(val_t_arr[index]).val = (char*)shared_buff + first_kv_p->valPos;
			//char *key = (char*)shared_buff + first_kv_p->keyPos;
			//char *val = (char*)shared_buff + first_kv_p->valPos;
			//char* val_p = (char*)((val_t_arr[index]).val);
			//if (thread_start_idx == 95)
			//int next_idx = first_kv_p->next_idx;
			//first_kv_p = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(shared_arr_len-next_idx));
			//printf("-map_task_idx:%d ->key:%s val:%d valPos:%d valSize:%d next_idx:%d shared_arr_len:%d [%d/%d]\n",
			//		first_kv_p->task_idx, key, *(int*)((char*)shared_buff + first_kv_p->valPos), first_kv_p->valPos,(val_t_arr[index]).valSize, next_idx, shared_arr_len, index, valCount);
			//printf("->compare key:%s next_key:%s\n", key, (char *)shared_buff + first_kv_p->keyPos);
			//index++;
		//}while(first_kv_p->next_idx != -1);

		if((index+1)!=valCount){
			ShowError("Error! thread_idx:%d  index:%d != valCount:%d  records are missing.\n",thread_start_idx,(index+1),valCount);
			return;
		}

		if(valCount>1)
		gpu_combiner(iKey,val_t_arr,iKeySize,(valCount),d_g_state,thread_start_idx);
		else{
			first_kv_p->next_idx = -2;
			first_kv_p->task_idx = thread_start_idx;
		}
		num_keyval_pairs_after_combiner++;
	}//for
	d_g_state.d_intermediate_keyval_total_count[thread_start_idx] = num_keyval_pairs_after_combiner;
	////////////////////////////////////////////////////////////////////
	__syncthreads();

}//GPUMapPartitioner
		
int StartCPUMap2(thread_info_t* thread_info)
{		
	cpu_context *d_g_state = (cpu_context*)(thread_info->d_g_state);
	job_configuration *cpu_job_conf = (job_configuration*)(thread_info->job_conf);

	if (cpu_job_conf->num_input_record<=0) { ShowError("Error: no any input keys"); exit(-1);}
	if (cpu_job_conf->input_keyval_arr == NULL) { ShowError("Error: input_keyval_arr == NULL"); exit(-1);}
	if (d_g_state->num_cpus_cores <= 0) {	ShowError("Error: d_g_state->num_cpus == 0"); exit(-1);}

	//-------------------------------------------------------
	//1, prepare buffer to store intermediate results
	//-------------------------------------------------------
	
	keyval_arr_t *d_keyval_arr_p;
	int *count = NULL;
	
	//---------------------------------------------
	//3, determine the number of threads to run
	//---------------------------------------------

	ShowLog("CPU_GROUP_ID:[%d] the number of cpus used in computation:%d",d_g_state->cpu_group_id, d_g_state->num_cpus_cores);
	
	//--------------------------------------------------
	//4, start_row_id map
	//--------------------------------------------------
	
	int num_threads = d_g_state->num_cpus_cores;
	int num_records_per_thread = (cpu_job_conf->num_input_record+num_threads-1)/(num_threads);
	
	int start_row_idx = 0;
	int end_row_idx = 0;

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
	
	ShowLog("CPU_GROUP_ID:[%d] DONE", d_g_state->cpu_group_id);
	return 0;
}//int 



int StartCPUMap(cpu_context *d_g_state)
{		
#ifdef DEV_MODE

	ShowLog("there are %d map tasks.",d_g_state->num_input_record);
	if (d_g_state->num_input_record<=0) { ShowError("Error: no any input keys"); exit(-1);}
	if (d_g_state->input_keyval_arr == NULL) { ShowError("Error: input_keyval_arr == NULL"); exit(-1);}
	if (d_g_state->num_cpus_cores <= 0) {	ShowError("Error: d_g_state->num_cpus == 0"); exit(-1);}

	//-------------------------------------------------------
	//1, prepare buffer to store intermediate results
	//-------------------------------------------------------
	
	ShowLog("prepare buffer to store intermediate results");
	
	keyval_arr_t *d_keyval_arr_p;
	int *count = NULL;
	//---------------------------------------------
	//3, determine the number of threads to run
	//---------------------------------------------
	ShowLog("the number of cpus used in computation:%d",d_g_state->num_cpus_cores);
	
	//--------------------------------------------------
	//4, start_row_id map
	//--------------------------------------------------
	int num_threads = d_g_state->num_cpus_cores;

	ShowLog("start_row_id CPUMapPartitioner num_threads:%d  num_input_record:%d",num_threads, d_g_state->num_input_record);
	int num_records_per_thread = (d_g_state->num_input_record+num_threads-1)/(num_threads);
	
	int start_row_idx = 0;
	int end_idx = 0;

	for (int tid = 0;tid<num_threads;tid++){
	
		end_idx = start_row_idx + num_records_per_thread;
		if (tid < (d_g_state->num_input_record % num_threads) )
			end_idx++;
		d_g_state->panda_cpu_task_info[tid].start_row_idx = start_row_idx;
		if (end_idx > d_g_state->num_input_record)
			end_idx = d_g_state->num_input_record;
		d_g_state->panda_cpu_task_info[tid].end_idx = end_idx;
		if (pthread_create(&(d_g_state->panda_cpu_task[tid]),NULL,RunPandaCPUMapThread,(char *)&(d_g_state->panda_cpu_task_info[tid]))!=0) 
			perror("Thread creation failed!\n");
		start_row_idx = end_idx;
	}//for
	
	for (int tid = 0;tid<num_threads;tid++){
		void *exitstat;
		if (pthread_join(d_g_state->panda_cpu_task[tid],&exitstat)!=0) perror("joining failed");
	}//for
	ShowLog("DONE");
#endif
	return 0;

}//int 

//--------------------------------------------------
//	StartGPUMap
//	Last Update 9/2/2012
//--------------------------------------------------

int StartGPUMap(gpu_context *d_g_state)
{		
	//-------------------------------------------------------
	//0, Check status of d_g_state;
	//-------------------------------------------------------
	ShowLog("GPU_ID:[%d]  num_input_record %d", d_g_state->gpu_id, d_g_state->num_input_record);
	if (d_g_state->num_input_record<0) { ShowLog("Error: no any input keys"); exit(-1);}
	if (d_g_state->h_input_keyval_arr == NULL) { ShowLog("Error: h_input_keyval_arr == NULL"); exit(-1);}
	if (d_g_state->num_mappers<=0) {d_g_state->num_mappers = (NUM_BLOCKS)*(NUM_THREADS);}
	if (d_g_state->num_reducers<=0) {d_g_state->num_reducers = (NUM_BLOCKS)*(NUM_THREADS);}

	//-------------------------------------------------------
	//1, prepare buffer to store intermediate results
	//-------------------------------------------------------

	keyval_arr_t *h_keyval_arr_arr = (keyval_arr_t *)malloc(sizeof(keyval_arr_t)*d_g_state->num_input_record);
	keyval_arr_t *d_keyval_arr_arr;
	checkCudaErrors(cudaMalloc((void**)&(d_keyval_arr_arr),d_g_state->num_input_record*sizeof(keyval_arr_t)));
	
	for (int i=0; i<d_g_state->num_input_record;i++){
		h_keyval_arr_arr[i].arr = NULL;
		h_keyval_arr_arr[i].arr_len = 0;
	}//for

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
	
	int total_gpu_threads = (grids.x*grids.y*blocks.x*blocks.y);
	ShowLog("GridDim.X:%d GridDim.Y:%d BlockDim.X:%d BlockDim.Y:%d TotalGPUThreads:%d",grids.x,grids.y,blocks.x,blocks.y,total_gpu_threads);

	cudaDeviceSynchronize();
	double t1 = PandaTimer();
	GPUMapPartitioner<<<grids,blocks>>>(*d_g_state);
	cudaThreadSynchronize();
	double t2 = PandaTimer();

	int num_records_per_thread = (d_g_state->num_input_record + (total_gpu_threads)-1)/(total_gpu_threads);
	int totalIter = num_records_per_thread;
	ShowLog("GPUMapPartitioner:%f totalIter:%d",t2-t1, totalIter);

	for (int iter = 0; iter< totalIter; iter++){

		double t3 = PandaTimer();
		RunGPUMapTasks<<<grids,blocks>>>(*d_g_state, totalIter -1 - iter, totalIter);
		cudaThreadSynchronize();
		double t4 = PandaTimer();
		size_t total_mem,avail_mem;
		checkCudaErrors(cudaMemGetInfo( &avail_mem, &total_mem ));

		ShowLog("GPU_ID:[%d] RunGPUMapTasks take %f sec at iter [%d/%d] remain %d mb GPU mem processed",
			d_g_state->gpu_id, t4-t3,iter,totalIter, avail_mem/1024/1024);
		
	}//for
	ShowLog("GPU_ID:[%d] Done %d Tasks",d_g_state->gpu_id,d_g_state->num_input_record);

	//Local combiner
	t1 = PandaTimer();
	checkCudaErrors(cudaMemset(d_g_state->d_intermediate_keyval_total_count,0,d_g_state->num_input_record*sizeof(int)));
	GPUCombiner<<<grids,blocks>>>(*d_g_state);
	cudaThreadSynchronize();
	t2 = PandaTimer();
	ShowLog("GPU_ID: GPUCombiner Take:%f",t2-t1);

	return 0;
}//int 




void DestroyDGlobalState(gpu_context * d_g_state){
	
}//void 


void StartGPUShuffle(gpu_context * state){
	
	gpu_context* d_g_state = state;
	double t1 = PandaTimer();
	Shuffle4GPUOutput(d_g_state);
	double t2 = PandaTimer();
	ShowLog("GPU_ID:[%d] GPU Shuffle take %f sec", state->gpu_id,t2-t1);

	//ShowLog("DONE");
}//void

void *RunPandaCPUMapThread(void *ptr){
		
	panda_cpu_task_info_t *panda_cpu_task_info = (panda_cpu_task_info_t *)ptr;
	cpu_context *d_g_state = (cpu_context *)(panda_cpu_task_info->d_g_state); 
	job_configuration *cpu_job_conf = (job_configuration *)(panda_cpu_task_info->cpu_job_conf); 

	for (int map_idx = panda_cpu_task_info->start_row_idx; map_idx < panda_cpu_task_info->end_row_idx; map_idx++){
		keyval_t *kv_p = (keyval_t *)(&(cpu_job_conf->input_keyval_arr[map_idx]));
		cpu_map(kv_p->key,kv_p->val,kv_p->keySize,kv_p->valSize,d_g_state,map_idx);
	}//for
	
	ShowLog("CPU_GROUP_ID:[%d] Done :%d tasks",d_g_state->cpu_group_id, panda_cpu_task_info->end_row_idx - panda_cpu_task_info->start_row_idx);
	return NULL;
}

//Use Pthread to process Panda_Reduce
void * Panda_Reduce(void *ptr){
		
	thread_info_t *thread_info = (thread_info_t *)ptr;
		
	if(thread_info->device_type == GPU_ACC){
		
		//TODO Multiple GPU support
		
		gpu_context *d_g_state = (gpu_context *)(thread_info->d_g_state);
		int num_gpus = d_g_state->num_gpus;
		if ( num_gpus <= 0){
			ShowLog("Error! num_gpus == 0 return");
			return NULL;
		}//if

		int tid = thread_info->tid;
		cudaSetDevice(tid % num_gpus);        // "% num_gpus" allows more CPU threads than GPU devices
		int gpu_id;
		cudaGetDevice(&gpu_id);

		ShowLog("Start GPU Reduce Tasks.  Number of Reduce Tasks:%d",d_g_state->d_sorted_keyvals_arr_len);

		StartGPUReduce(d_g_state);

	}//if
		
	if(thread_info->device_type == CPU_ACC){
		
		cpu_context *d_g_state = (cpu_context *)(thread_info->d_g_state);
		
		if (d_g_state->num_cpus_cores == 0){
			ShowLog("Error! d_g_state->num_cpus == 0 return");
			return NULL;
		}
		
		ShowLog("Start CPU Reduce Tasks.  Number of Reduce Tasks:%d",d_g_state->sorted_keyvals_arr_len);
		
		for (int map_idx = 0; map_idx < d_g_state->sorted_keyvals_arr_len; map_idx++){

			keyvals_t *kv_p = (keyvals_t *)(&(d_g_state->sorted_intermediate_keyvals_arr[map_idx]));
			cpu_reduce(kv_p->key, kv_p->vals, kv_p->keySize, kv_p->val_arr_len, d_g_state);

		}//for
		
	}//if	
	
	//cudaFree(d_filebuf);
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

	int num_records_per_thread = (d_g_state.d_sorted_keyvals_arr_len + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx + num_records_per_thread*STRIDE;
	if (thread_end_idx > d_g_state.d_sorted_keyvals_arr_len)
		thread_end_idx = d_g_state.d_sorted_keyvals_arr_len;
	if (thread_start_idx >= thread_end_idx)
		return;

	int start_idx, end_idx;
	for(int reduce_task_idx=thread_start_idx; reduce_task_idx < thread_end_idx; reduce_task_idx+=STRIDE){

		if (reduce_task_idx==0)
			start_idx = 0;
		else
			start_idx = d_g_state.d_pos_arr_4_sorted_keyval_pos_arr[reduce_task_idx-1];
		end_idx = d_g_state.d_pos_arr_4_sorted_keyval_pos_arr[reduce_task_idx];
		val_t *val_t_arr = (val_t*)malloc(sizeof(val_t)*(end_idx-start_idx));
		int keySize = d_g_state.d_keyval_pos_arr[start_idx].keySize;
		int keyPos = d_g_state.d_keyval_pos_arr[start_idx].keyPos;
		void *key = (char*)d_g_state.d_sorted_keys_shared_buff+keyPos;
				
		for (int index = start_idx;index<end_idx;index++){
			int valSize = d_g_state.d_keyval_pos_arr[index].valSize;
			int valPos = d_g_state.d_keyval_pos_arr[index].valPos;
			val_t_arr[index-start_idx].valSize = valSize;
			val_t_arr[index-start_idx].val = (char*)d_g_state.d_sorted_vals_shared_buff + valPos;
		}   //for

		gpu_reduce(key, val_t_arr, keySize, end_idx-start_idx, d_g_state);

	}//for
}


		
void StartGPUReduce(gpu_context *d_g_state)
{	
	cudaThreadSynchronize(); 
	d_g_state->d_reduced_keyval_arr_len = d_g_state->d_sorted_keyvals_arr_len;
	checkCudaErrors(cudaMalloc((void **)&(d_g_state->d_reduced_keyval_arr), sizeof(keyval_t)*d_g_state->d_reduced_keyval_arr_len));


	cudaThreadSynchronize(); 
	int numGPUCores = getGPUCoresNum();
	dim3 blocks(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
	int numBlocks = (numGPUCores*16+(blocks.x*blocks.y)-1)/(blocks.x*blocks.y);
    dim3 grids(numBlocks, 1);

	int total_gpu_threads = (grids.x*grids.y*blocks.x*blocks.y);
	ShowLog("number of reduce tasks:%d total gpu threads:%d",d_g_state->d_sorted_keyvals_arr_len, total_gpu_threads);
	ReducePartitioner<<<grids,blocks>>>(*d_g_state);
	cudaThreadSynchronize(); 

}//void


void* Panda_Map(void *ptr){
		
	thread_info_t *thread_info = (thread_info_t *)ptr;
		
	if(thread_info->device_type == GPU_ACC){
		
		gpu_context *d_g_state = (gpu_context *)(thread_info->d_g_state);
		InitGPUDevice(thread_info);
		
		//ShowLog("GPU_ID:[%d] Init GPU MapReduce Load Data From Host to GPU memory",d_g_state->gpu_id);
		InitGPUMapReduce3(d_g_state);

		//ShowLog("GPU_ID:[%d] Start GPU Map Tasks",d_g_state->gpu_id);
		StartGPUMap(d_g_state);
		
		StartGPUShuffle(d_g_state);
		
	}//if
		
	if(thread_info->device_type == CPU_ACC){

		cpu_context *d_g_state = (cpu_context *)(thread_info->d_g_state);
		//ShowLog("CPU_GROUP_ID:[%d] Init CPU Device",d_g_state->cpu_group_id);
		InitCPUDevice(thread_info);
		
		//ShowLog("Init CPU MapReduce");
		InitCPUMapReduce2(thread_info);

		ShowLog("CPU_GROUP_ID:[%d] Start CPU Map Tasks",d_g_state->cpu_group_id);
		StartCPUMap2(thread_info);
		
		StartCPUShuffle2(thread_info);
		
	}	
			
	return NULL;
}//FinishMapReduce2(d_g_state);


void FinishMapReduce(Spec_t* spec)
{
	ShowLog( "=====finish panda mapreduce=====");
}//void


void FinishMapReduce2(gpu_context* state)
{

	size_t total_mem,avail_mem, heap_limit;
	checkCudaErrors(cudaMemGetInfo( &avail_mem, &total_mem ));
	ShowLog("avail_mem:%d",avail_mem);

}//void


#endif //__PANDALIB_CU__