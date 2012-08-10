/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 0.20
	
	File: PandaLib.cu 
	First Version:	2012-07-01 V0.1
	Github: https://github.com/cyberaide/biostatistics/tree/master/GPUMapReduce			

	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */

#ifndef __PANDALIB_CU__
#define __PANDALIB_CU__

#include "Panda.h"
#include "stdlib.h"
#include "map.cu"
#include "reduce.cu"

//----------------------------------------------
//Get default runtime configuration
//return: default spec
//----------------------------------------------

gpu_context *GetGPUContext(){
	gpu_context *d_g_state = (gpu_context*)malloc(sizeof(gpu_context));
	if (d_g_state == NULL) exit(-1);
	memset(d_g_state, 0, sizeof(gpu_context));
	d_g_state->configured = false;
	d_g_state->h_input_keyval_arr = NULL;
	return d_g_state;
}//gpu_context
			 
cpu_context *GetCPUContext(){
	cpu_context *d_g_state = (cpu_context*)malloc(sizeof(cpu_context));
	if (d_g_state == NULL) exit(-1);
	memset(d_g_state, 0, sizeof(gpu_context));
	d_g_state->configured = false;
	d_g_state->input_keyval_arr = NULL;
	return d_g_state;
}//gpu_context


void InitCPUMapReduce(cpu_context* d_g_state)
{	
			
	DoLog("d_g_state->configured:%s  enable for iterative applications",d_g_state->configured? "true" : "false");
	//if (d_g_state->configured)
	//	return;
	DoLog("d_g_state->num_input_record:%d",d_g_state->num_input_record);
	int totalKeySize = 0;
	int totalValSize = 0;
	for(int i=0;i<d_g_state->num_input_record;i++){
		totalKeySize += d_g_state->input_keyval_arr[i].keySize;
		totalValSize += d_g_state->input_keyval_arr[i].valSize;
	}//for
	DoLog("totalKeySize:%d totalValSize:%d num_cpus:%d", totalKeySize, totalValSize, d_g_state->num_cpus);
		
	//TODO determin num_cpus
	
	//d_g_state->num_cpus = 12;

	int num_cpus = d_g_state->num_cpus;
	d_g_state->panda_cpu_task = (pthread_t *)malloc(sizeof(pthread_t)*(num_cpus));
	d_g_state->panda_cpu_task_info = (panda_cpu_task_info_t *)malloc(sizeof(panda_cpu_task_info_t)*(num_cpus));

	d_g_state->intermediate_keyval_arr_arr_p = (keyval_arr_t *)malloc(sizeof(keyval_arr_t)*d_g_state->num_input_record);
	memset(d_g_state->intermediate_keyval_arr_arr_p, 0, sizeof(keyval_arr_t)*d_g_state->num_input_record);

	for (int i=0;i<num_cpus;i++){
		d_g_state->panda_cpu_task_info[i].d_g_state = d_g_state;
		d_g_state->panda_cpu_task_info[i].num_cpus = num_cpus;
		d_g_state->panda_cpu_task_info[i].start_idx = 0;
		d_g_state->panda_cpu_task_info[i].end_idx = 0;
	}//for
	d_g_state->configured = true;
}//void
	

void InitGPUMapReduce3(gpu_context* d_g_state)
{	

	DoLog("d_g_state->configured:%s  enable for iterative applications",d_g_state->configured? "true" : "false");
	//if (d_g_state->configured)
	//	return;
	
	DoLog("d_g_state->num_input_record:%d",d_g_state->num_input_record);
	//checkCudaErrors(cudaMalloc((void **)&d_g_state->d_input_keyval_arr,sizeof(keyval_t)*d_g_state->num_input_record));
	int totalKeySize = 0;
	int totalValSize = 0;
	for(int i=0;i<d_g_state->num_input_record;i++){
		totalKeySize += d_g_state->h_input_keyval_arr[i].keySize;
		totalValSize += d_g_state->h_input_keyval_arr[i].valSize;
	}//for
	
	DoLog("totalKeySize:%d totalValSize:%d", totalKeySize, totalValSize);
	
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

#if 0
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




/*
void InitMapReduce(Spec_t* spec)
{
	if (g_spec->dimBlockMap <= 0)
		g_spec->dimBlockMap = DEFAULT_DIMBLOCK;
	if (g_spec->dimBlockReduce <= 0)
		g_spec->dimBlockReduce = DEFAULT_DIMBLOCK;
	if (g_spec->numRecTaskReduce <= 0)
		g_spec->numRecTaskReduce = DEFAULT_NUMTASK;
	if (g_spec->numRecTaskMap <= 0)
		g_spec->numRecTaskMap = DEFAULT_NUMTASK;
	if (g_spec->workflow <= 0)
		g_spec->workflow = MAP_ONLY;
}*/

void InitCPUDevice(thread_info_t*thread_info){

	//------------------------------------------
	//1, init CPU device
	//------------------------------------------
	
	cpu_context *d_g_state = (cpu_context *)(thread_info->d_g_state);
	int tid = thread_info->tid;
	DoLog( "Init CPU Deivce tid:%d",tid);
	//char *fn = thread_info->file_name;
	//"% num_gpus" allows more CPU threads than GPU devices
	
}


void InitGPUDevice(thread_info_t*thread_info){

	//------------------------------------------
	//1, init device
	//------------------------------------------
	DoLog( "Init GPU Deivce");
	gpu_context *d_g_state = (gpu_context *)(thread_info->d_g_state);

	int tid = thread_info->tid;
	//char *fn = thread_info->file_name;
	int num_gpus = thread_info->num_gpus;
	cudaSetDevice(tid % num_gpus);        
	//"% num_gpus" allows more CPU threads than GPU devices
	int gpu_id;
	cudaGetDevice(&gpu_id);
	d_g_state->gpu_id = gpu_id;
		
	//CUT_DEVICE_INIT();
	size_t total_mem,avail_mem, heap_limit;
	checkCudaErrors(cudaMemGetInfo( &avail_mem, &total_mem ));
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, (int)(total_mem*0.2)); 
	cudaDeviceGetLimit(&heap_limit, cudaLimitMallocHeapSize);

	DoLog("tid:%d num_gpus:%d gpu_id:%d device_name:%s",tid,num_gpus,gpu_id,thread_info->device_name);
	DoLog("cudaLimitMallocHeapSize:%d KB avail_mem:%d KB total_mem:%d KB",heap_limit/1024, avail_mem/1024,total_mem/1024);

}



//--------------------------------------------------
//  Add a map input record
//--------------------------------------------------

void Start_Panda_Job(job_configuration *job_conf){
		
	int num_gpus = job_conf->num_gpus;
	int num_cpus = job_conf->num_cpus;
	int num_cpus_group = job_conf->num_cpus_groups;
		
	pthread_t *no_threads = (pthread_t*)malloc(sizeof(pthread_t)*(num_gpus + num_cpus_group));
	thread_info_t *thread_info = (thread_info_t*)malloc(sizeof(thread_info_t)*(num_gpus + num_cpus_group));
		
	for (int i=0; i<num_gpus; i++){
		thread_info[i].tid = i;
		//thread_info[i].file_name = argv[i+1];
		thread_info[i].num_gpus = num_gpus;
		thread_info[i].device_type = GPU_ACC;
		
		cudaDeviceProp gpu_dev;
		cudaGetDeviceProperties(&gpu_dev, i);
		DoLog("Configure Device ID:%d: Device Name:%s", i, gpu_dev.name);
		thread_info[i].device_name = gpu_dev.name;
		gpu_context *d_g_state = GetGPUContext();
		thread_info[i].d_g_state = d_g_state;
	}//for num_gpus
		
	for (int i=num_gpus;i<num_gpus+num_cpus_group;i++){
		thread_info[i].tid = i;
		thread_info[i].device_type = CPU_ACC;
		cpu_context *d_g_state = GetCPUContext();
		d_g_state->num_cpus = num_cpus;
		
		thread_info[i].d_g_state = d_g_state;
	}//for

	int *split = (int*)malloc(sizeof(int)*(num_gpus+num_cpus_group));
	//int split[2];
	//"400/400/10/"
	printf("num_gpus:%d num_cpus_group:%d num_input_record:%d\n", num_gpus, num_cpus_group,job_conf->num_input_record);
	int num_input_record = job_conf->num_input_record;
	int records_per_device = num_input_record/(num_gpus*10+num_cpus_group);
	for (int i=0;i<num_gpus;i++){
			split[i] = records_per_device*10*(i+1);
	}//for
	for (int i=num_gpus;i<num_gpus+num_cpus_group;i++){
			split[i] = records_per_device*10*(num_gpus)+(i+1)*records_per_device;
	}//for
	split[num_gpus+num_cpus_group-1] = num_input_record;

	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_group); dev_id++){
		int start_id = 0;
		int end_id = split[dev_id];
		if (dev_id>0)
			start_id = split[dev_id-1];
		
		//printf("start_id:%d end_id:%d\n",start_id, end_id);

			if (thread_info[dev_id].device_type == GPU_ACC){
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);

				for (int i=start_id;i<end_id;i++){
				//printf(":%s  keySize:%d",job_conf->input_keyval_arr[i].val, job_conf->input_keyval_arr[i].valSize);

				AddMapInputRecordGPU(d_g_state,&(job_conf->input_keyval_arr[i]));
				}

			}//if

			if (thread_info[dev_id].device_type == CPU_ACC){
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);

				for (int i=start_id;i<end_id;i++){
				AddMapInputRecordCPU(d_g_state,&(job_conf->input_keyval_arr[i]));
				}

			}//if
		//}//for
	}//for

	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_group); dev_id++){
		//if (thread_info[dev_id].device_type == GPU_ACC){
		if (pthread_create(&(no_threads[dev_id]),NULL,Panda_Map,(char *)&(thread_info[dev_id]))!=0) 
			perror("Thread creation failed!\n");
		//}//if
		/*if(thread_info[dev_id].device_type == CPU_ACC){
			if (pthread_create(&(no_threads[dev_id]),NULL,Panda_Map,(char *)&(thread_info[dev_id]))!=0) 
			perror("Thread creation failed!\n");
		}//if
		*/
	}//for
		
	for (int i=0; i<num_gpus+num_cpus_group; i++){
		void *exitstat;
		if (pthread_join(no_threads[i],&exitstat)!=0) perror("joining failed");
	}//for

	DoLog("start to merge!");
	
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
			Panda_Shuffle_Merge(&dummy_d_g_state, (gpu_context*)(thread_info[0].d_g_state));
	}//if

	//merge the thread_info results;
	for (int i=1; i<num_gpus; i++){
		Panda_Shuffle_Merge((gpu_context*)(thread_info[i-1].d_g_state), (gpu_context*)(thread_info[i].d_g_state));
	}//for

	for (int i = num_gpus; i < num_gpus+num_cpus_group;i++){
		
	}//for

	cudaThreadSynchronize();
	Panda_Reduce(&thread_info[num_gpus-1]);
	
}

void AddPandaTask(job_configuration* job_conf,
						void*		key, 
						void*		val,
						int		keySize, 
						int		valSize){
	
	int len = job_conf->num_input_record;
	if (len<0) return;

	job_conf->input_keyval_arr = (keyval_t *)realloc(job_conf->input_keyval_arr, sizeof(keyval_t)*(len+1));
	job_conf->input_keyval_arr[len].keySize = keySize;
	job_conf->input_keyval_arr[len].valSize = valSize;
	job_conf->input_keyval_arr[len].key = malloc(keySize);
	job_conf->input_keyval_arr[len].val = malloc(valSize);

	memcpy(job_conf->input_keyval_arr[len].key,key,keySize);
	memcpy(job_conf->input_keyval_arr[len].val,val,valSize);
	job_conf->num_input_record++;
	
}

void AddMapInputRecordGPU(gpu_context* d_g_state,
						keyval_t *kv_p){
		
	int len = d_g_state->num_input_record;
	//DoLog("len:%d realloc:%d",len,sizeof(keyval_t)*(len+1));
	if (len == 0)
		d_g_state->h_input_keyval_arr = NULL;

	if (len<0) return;
	
	d_g_state->h_input_keyval_arr = (keyval_t *)realloc(d_g_state->h_input_keyval_arr, sizeof(keyval_t)*(len+1));

	//assert(d_g_state->h_input_keyval_arr != NULL);

	
	d_g_state->h_input_keyval_arr[len].keySize = kv_p->keySize;
	d_g_state->h_input_keyval_arr[len].valSize = kv_p->valSize;
	d_g_state->h_input_keyval_arr[len].key = kv_p->key;
	d_g_state->h_input_keyval_arr[len].val = kv_p->val;
	//memcpy(d_g_state->h_input_keyval_arr[len].key,key,keySize);
	//memcpy(d_g_state->h_input_keyval_arr[len].val,val,valSize);
	

	d_g_state->num_input_record++;
			
}


void AddMapInputRecordCPU(cpu_context* d_g_state,
						keyval_t *kv_p){
			
	int len = d_g_state->num_input_record;
	if (len<0) return;
	//DoLog("len:%d size:%d",len,sizeof(keyval_t)*(len+1));
	d_g_state->input_keyval_arr = (keyval_t *)realloc(d_g_state->input_keyval_arr, sizeof(keyval_t)*(len+1));
	d_g_state->input_keyval_arr[len].keySize = kv_p->keySize;
	d_g_state->input_keyval_arr[len].valSize = kv_p->valSize;
	d_g_state->input_keyval_arr[len].key = kv_p->key;
	d_g_state->input_keyval_arr[len].val = kv_p->val;
	
	d_g_state->num_input_record++;
}

__device__ void Emit2  (void*		key, 
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
			printf("[output]: key:%s  val:%d\n",key,*(int *)val);
						
}//__device__ 


void CPUEmitIntermediate(void *key, void *val, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx){
	
	//printf(":%s   :%d\n",key, *(int*)val);
	keyval_arr_t *kv_arr_p = &(d_g_state->intermediate_keyval_arr_arr_p[map_task_idx]);
	//keyval_t *p = (keyval_t *)kv_arr_p->arr;
	//void *buff = kv_arr_p->buff;
	//&(kv_arr_p->arr[len]) = (keyval_t*)((char *)buff - sizeof(keyval_t));
	//keyval_t *kv_p = (keyval_t *)((char *)buff + kv_arr_p->buff_len - sizeof(keyval_t)*((*kv_arr_p->total_arr_len)+1));

	if (kv_arr_p->arr_len==0)
		kv_arr_p->arr = NULL;

	kv_arr_p->arr = (keyval_t*)realloc(kv_arr_p->arr, sizeof(keyval_t)*(kv_arr_p->arr_len+1));
	//(*kv_arr_p->total_arr_len)++;

	/*
	if (!(kv_arr_p->buff_pos +keySize+valSize < kv_arr_p->buff_len - sizeof(keyval_t)*((*kv_arr_p->total_arr_len)+1))){
		printf("!!!!!!!error there is not engough shared memory\n");
		return;
	}*/

	//printf("remain buff:%d\n", kv_arr_p->buff_len - sizeof(keyval_t)*(kv_arr_p->arr_len+1) - kv_arr_p->buff_pos);
	int current_map_output_index = (kv_arr_p->arr_len);
	keyval_t *kv_p = &(kv_arr_p->arr[current_map_output_index]);
	kv_p->key = (char *)malloc(sizeof(keySize));
	//kv_arr_p->buff_pos += keySize;
	memcpy(kv_p->key,key,keySize);
	kv_p->keySize = keySize;
	
	kv_p->val = (char *)malloc(sizeof(valSize));
	//kv_arr_p->buff_pos += valSize;
	memcpy(kv_p->val,val,valSize);
	kv_p->valSize = valSize;

	kv_arr_p->arr_len++;

	//d_g_state->d_intermediate_keyval_total_count[map_task_idx] = kv_arr_p->arr_len;
	/*
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
	*/

	//printf("CPU EmitInterMediate2 map_task_id:%d, key:%s: keyval_arr_len:%d\n", map_task_idx, kv_p->key, kv_arr_p->arr_len);

}//__device__


__device__ void EmitIntermediate2(void *key, void *val, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){
	
	//printf(":%s   :%d\n",key, *(int*)val);
	keyval_arr_t *kv_arr_p = d_g_state->d_intermediate_keyval_arr_arr_p[map_task_idx];
	//keyval_t *p = (keyval_t *)kv_arr_p->arr;
	void *buff = kv_arr_p->buff;
	//&(kv_arr_p->arr[len]) = (keyval_t*)((char *)buff - sizeof(keyval_t));
	
	keyval_t *kv_p = (keyval_t *)((char *)buff + kv_arr_p->buff_len - sizeof(keyval_t)*((*kv_arr_p->total_arr_len)+1));
	kv_arr_p->arr = kv_p;
	(*kv_arr_p->total_arr_len)++;

	//TODO Hui Li 8/6/2012
	if (!(kv_arr_p->buff_pos +keySize+valSize < kv_arr_p->buff_len - sizeof(keyval_t)*((*kv_arr_p->total_arr_len)+1))){
		printf("!!!!!!!error there is not engough shared memory\n");
		return;
	}//
	//printf("remain buff:%d\n", kv_arr_p->buff_len - sizeof(keyval_t)*(kv_arr_p->arr_len+1) - kv_arr_p->buff_pos);
	//int current_map_output_index = (kv_arr_p->arr_len);
	//keyval_t *kv_p = &(kv_arr_p->arr[current_map_output_index]);

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
	//printf("EmitInterMediate2 TID[%d] map_task_id:%d, key%s: keyval_arr_len:%d\n",TID, map_task_idx, kv_p->key, kv_arr_p->arr_len);
}//__device__

//-------------------------------------------------
//called by user defined map function
//-------------------------------------------------

__global__ void GPUMapPartitioner(gpu_context d_g_state)
{	
	/*int index = TID;
	int bid = BLOCK_ID;
	int tid = THREAD_ID;*/
	
	int num_records_per_thread = (d_g_state.num_input_record+(gridDim.x*blockDim.x)-1)/(gridDim.x*blockDim.x);
	int block_start_idx = num_records_per_thread*blockIdx.x*blockDim.x;
	int thread_start_idx = block_start_idx 
		+ (threadIdx.x/STRIDE)*num_records_per_thread*STRIDE
		+ (threadIdx.x%STRIDE);
	int thread_end_idx = thread_start_idx+num_records_per_thread*STRIDE;
			
	if (thread_end_idx>d_g_state.num_input_record)
		thread_end_idx = d_g_state.num_input_record;
	//printf("Mapper TID:%d, thread_start_idx:%d  thread_end_idx:%d totalThreads:%d\n",TID, thread_start_idx,thread_end_idx,gridDim.x*blockDim.x);

	if (thread_start_idx >= thread_end_idx)
		return;

	char * buff = (char *)malloc(sizeof(char)*1024*100);
	int * total_arr_len = (int*)malloc(sizeof(int));
	(*total_arr_len) = 0;
   
	for(int map_task_idx=thread_start_idx; map_task_idx < thread_end_idx; map_task_idx+=STRIDE){
		
		keyval_arr_t *kv_arr_t = (keyval_arr_t *)malloc(sizeof(keyval_arr_t));
		kv_arr_t->buff = buff;
		kv_arr_t->total_arr_len = total_arr_len;

		kv_arr_t->buff_len = 1024*100;
		kv_arr_t->buff_pos = 0;
		kv_arr_t->arr = NULL;
		kv_arr_t->arr_len = 0;
		
		//keyval_arr_t *kv_arr_p = &(d_g_state.d_intermediate_keyval_arr_arr[map_task_idx]);
		d_g_state.d_intermediate_keyval_arr_arr_p[map_task_idx] = kv_arr_t;
		
		/*(d_g_state.d_intermediate_keyval_arr_arr[map_task_idx]).buff = buff;
		(d_g_state.d_intermediate_keyval_arr_arr[map_task_idx]).buff_len = 1024*1024;
		(d_g_state.d_intermediate_keyval_arr_arr[map_task_idx]).buff_pos = 0;*/

		//void *val = d_g_state.d_input_keyval_arr[map_task_idx].val;
		char *key = (char *)(d_g_state.d_input_keys_shared_buff) + d_g_state.d_input_keyval_pos_arr[map_task_idx].keyPos;
		char *val = (char *)(d_g_state.d_input_vals_shared_buff) + d_g_state.d_input_keyval_pos_arr[map_task_idx].valPos;

		int valSize = d_g_state.d_input_keyval_pos_arr[map_task_idx].valSize;
		int keySize = d_g_state.d_input_keyval_pos_arr[map_task_idx].keySize;

		//printf("map_task_idx:%d keySize:%d  valSize:%d\n",map_task_idx, keySize, valSize);
		//printf("key:%d  keySize:%d\n",*(int *)key, keySize);
		//TODO calculate the key val pair here directly. 

		///////////////////////////////////////////////////////////
		map2(key, val, keySize, valSize, &d_g_state, map_task_idx);
		///////////////////////////////////////////////////////////

	}//for
	//__syncthreads();
}//GPUMapPartitioner


int StartCPUMap(cpu_context *d_g_state)
{		
	
	DoLog("check parameters of for map tasks:%d",d_g_state->num_input_record);
	if (d_g_state->num_input_record<0) { DoLog("Error: no any input keys"); exit(-1);}
	if (d_g_state->input_keyval_arr == NULL) { DoLog("Error: input_keyval_arr == NULL"); exit(-1);}
		
	//-------------------------------------------------------
	//1, prepare buffer to store intermediate results
	//-------------------------------------------------------
	
	DoLog("prepare buffer to store intermediate results");
	
	keyval_arr_t *d_keyval_arr_p;
	int *count = NULL;
	
	//---------------------------------------------
	//3, determine the number of threads to run
	//---------------------------------------------
	DoLog("the number of cpus used in computation:%d",d_g_state->num_cpus);
	
	//--------------------------------------------------
	//4, start map
	//--------------------------------------------------
	
	int num_threads = d_g_state->num_cpus;

	DoLog("start CPUMapPartitioner num_threads:%d  num_input_record:%d",num_threads, d_g_state->num_input_record);
	int num_records_per_thread = (d_g_state->num_input_record+num_threads-1)/(num_threads);
	
	int start_idx = 0;
	int end_idx = 0;

	//pthread_t  *cpu_threads;
	//thread_info_t *cpu_threads_info;

	for (int tid = 0;tid<num_threads;tid++){
	
		end_idx = start_idx + num_records_per_thread;
		if (tid < (d_g_state->num_input_record % num_threads) )
			end_idx++;
			
		d_g_state->panda_cpu_task_info[tid].start_idx = start_idx;
		if (end_idx > d_g_state->num_input_record)
			end_idx = d_g_state->num_input_record;
		d_g_state->panda_cpu_task_info[tid].end_idx = end_idx;
		
		//pthread_t  *panda_cpu_task;
		//panda_cpu_task_info_t *panda_cpu_task_info;

		if (pthread_create(&(d_g_state->panda_cpu_task[tid]),NULL,RunPandaCPUMapThread,(char *)&(d_g_state->panda_cpu_task_info[tid]))!=0) 
			perror("Thread creation failed!\n");
		
		start_idx = end_idx;
	}//for

	
	for (int tid = 0;tid<num_threads;tid++){
		void *exitstat;
		if (pthread_join(d_g_state->panda_cpu_task[tid],&exitstat)!=0) perror("joining failed");
	}//for
	
	DoLog("DONE");

	return 0;
}//int 

//--------------------------------------------------
//StartGPUMap
//
//7/1/2012
//--------------------------------------------------
 

int StartGPUMap(gpu_context *d_g_state)
{		

	//-------------------------------------------------------
	//0, Check status of d_g_state;
	//-------------------------------------------------------

	DoLog("check parameters of for map tasks:%d",d_g_state->num_input_record);
	if (d_g_state->num_input_record<0) { DoLog("Error: no any input keys"); exit(-1);}
	if (d_g_state->h_input_keyval_arr == NULL) { DoLog("Error: h_input_keyval_arr == NULL"); exit(-1);}
		
	//-------------------------------------------------------
	//1, prepare buffer to store intermediate results
	//-------------------------------------------------------
	DoLog("prepare buffer to store intermediate results");

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
	//TODO determine the number of threads to run
	DoLog("determine the number of threads (NUM_BLOCKS, NUM_THREADS) to run");
	//int num_threads = d_g_state->num_input_record;
	//calculate NUM_BLOCKS, NUM_THREADS
	
	//--------------------------------------------------
	//4, start map
	//--------------------------------------------------
	/*dim3 h_dimBlock(512,1,1);
    dim3 h_dimGrid(4,1,1);
	dim3 h_dimThread(1,1,1);
	int sizeSmem = 128;*/
	DoLog("start GPUMapPartitioner");
	cudaThreadSynchronize();
	
	//printf("avail_mem:%d \n",avail_mem);
	
	GPUMapPartitioner<<<NUM_BLOCKS,NUM_THREADS>>>(*d_g_state);
	cudaThreadSynchronize();

	//size_t total_mem,avail_mem, heap_limit;
	//checkCudaErrors(cudaMemGetInfo( &avail_mem, &total_mem ));
	//printf("avail_mem:%d \n",avail_mem);
	
	DoLog("DONE");

	return 0;
}//int 


//--------------------------------------------------
//start map
//
//1, get map input data on host
//2, upload map input data to device memory
//	 (keys, vals, keyOffsets, valOffsets, keySizes, valSizes)
//3, determine the number of threads to run 
//4, calculate intermediate data keys'buf size 
//	 and values' buf size
//5, do prefix sum on--
//	 i)		d_interKeysSizePerTask
//	 ii)	d_interValsSizePerTask
//	 iii)	d_interCountPerTask
//6, allocate intermediate memory on device memory
//7, start map
//8, free allocated memory
//--------------------------------------------------

/*
int startMap(Spec_t* spec, gpu_context *d_g_state)
{
	    
	Spec_t* g_spec = spec;

	if (g_spec->inputKeys == NULL) { DoLog("Error: no any input keys"); exit(0);}
	if (g_spec->inputVals == NULL) { DoLog("Error: no any input values"); exit(0); }
	if (g_spec->inputOffsetSizes == NULL) { DoLog( "Error: no any input pointer info"); exit(0); }
	if (g_spec->inputRecordCount == 0) {DoLog( "Error: invalid input record count"); exit(0);}
	
	//-------------------------------------------------------
	//1, get map input data on host
	//-------------------------------------------------------
	return 0;
}//return 0;
*/

void DestroyDGlobalState(gpu_context * d_g_state){
	
}//void 


void StartGPUShuffle(gpu_context * state){
	gpu_context* d_g_state = state;
	DoLog("GPU Shuffle");
	
	Shuffle4GPUOutput(d_g_state);

	DoLog("DONE");
}//void

void *RunPandaCPUMapThread(void *ptr){
		
	panda_cpu_task_info_t *panda_cpu_task_info = (panda_cpu_task_info_t *)ptr;
	cpu_context *d_g_state = (cpu_context *)(panda_cpu_task_info->d_g_state); 

	DoLog("panda_cpu_task_info_t start_idx:%d end_idx:%d",panda_cpu_task_info->start_idx, panda_cpu_task_info->end_idx);
	for (int map_idx = panda_cpu_task_info->start_idx; map_idx < panda_cpu_task_info->end_idx; map_idx++){
		keyval_t *kv_p = (keyval_t *)(&(d_g_state->input_keyval_arr[map_idx]));
		//void cpu_map(void *KEY, void*VAL, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx){
		cpu_map(kv_p->key,kv_p->val,kv_p->keySize,kv_p->valSize,d_g_state,map_idx);
	}//for
	
	return NULL;

}

//Use Pthread to process Panda_Reduce
void * Panda_Reduce(void *ptr){

	thread_info_t *thread_info = (thread_info_t *)ptr;
	int tid = thread_info->tid;
	int num_gpus = thread_info->num_gpus;
	//cudaSetDevice(tid % num_gpus);        // "% num_gpus" allows more CPU threads than GPU devices
	int gpu_id;
    cudaGetDevice(&gpu_id);
	
	//printf("tid:%d num_gpus:%d gpu_id:%d\n",tid,num_gpus,gpu_id);
	
	//configuration for Panda Reduce
	gpu_context *d_g_state = (gpu_context *)(thread_info->d_g_state);
	DoLog( "start reduce tasks on GPU:%d tid:%d",gpu_id, tid);
	//TimeVal_t reduceTimer;
	//startTimer(&reduceTimer);

	StartGPUReduce(d_g_state);
	//endTimer("Reduce", &reduceTimer);

	FinishMapReduce2(d_g_state);
	//cudaFree(d_filebuf);
	//handle the buffer different
	//free(h_filebuf);
	//handle the buffer different
	return NULL;
}

//--------------------------------------------------------
//get a value from value list of the same key
//
//param	: vals
//param	: interOffsetSizes
//param	: index
//return: the wanted value
//--------------------------------------------------------

__device__ void *GetVal(void *vals, int4* interOffsetSizes, int keyIndex, int valStartIndex)
{
	int4 offset = interOffsetSizes[valStartIndex];
	return (void*)((char*)vals + keyIndex * offset.w);
}

__device__ void *GetKey(void *key, int4* interOffsetSizes, int keyIndex, int valStartIndex)
{
	int4 offset = interOffsetSizes[valStartIndex];
	return (void*)((char*)key + keyIndex * offset.y);
}

//-------------------------------------------------------
//Reducer
//-------------------------------------------------------

__global__ void ReducePartitioner(gpu_context d_g_state)
{
	int num_records_per_thread = (d_g_state.d_sorted_keyvals_arr_len+(gridDim.x*blockDim.x)-1)/(gridDim.x*blockDim.x);
	int block_start_idx = num_records_per_thread*blockIdx.x*blockDim.x;

	int thread_start_idx = block_start_idx 
		+ (threadIdx.x/STRIDE)*num_records_per_thread*STRIDE
		+ (threadIdx.x%STRIDE);

	int thread_end_idx = thread_start_idx+num_records_per_thread*STRIDE;
	if(thread_end_idx>d_g_state.d_sorted_keyvals_arr_len)
		thread_end_idx = d_g_state.d_sorted_keyvals_arr_len;

	//printf("ReducePartitioner: TID:%d  start_idx:%d  end_idx:%d d_sorted_keyvals_arr_len:%d\n",TID,thread_start_idx,thread_end_idx,d_g_state.d_sorted_keyvals_arr_len);

	int start, end;
	for(int reduce_task_idx=thread_start_idx; reduce_task_idx < thread_end_idx; reduce_task_idx+=STRIDE){
		if (reduce_task_idx==0)
			start = 0;
		else
			start = d_g_state.d_pos_arr_4_sorted_keyval_pos_arr[reduce_task_idx-1];
		end = d_g_state.d_pos_arr_4_sorted_keyval_pos_arr[reduce_task_idx];

		val_t *val_t_arr = (val_t*)malloc(sizeof(val_t)*(end-start));
		//assert(val_t_arr!=NULL);
		int keySize = d_g_state.d_keyval_pos_arr[start].keySize;
		int keyPos = d_g_state.d_keyval_pos_arr[start].keyPos;
		void *key = (char*)d_g_state.d_sorted_keys_shared_buff+keyPos;
		//printf("keySize;%d keyPos:%d key:%s\n",keySize,keyPos,key);
		//printf("reduce_task_idx:%d		keyPos:%d,  keySize:%d, key:% start:%d end:%d\n",reduce_task_idx,keyPos,keySize,start,end);
		//printf("start:%d end:%d\n",start,end);
		
		
		for (int index = start;index<end;index++){
			int valSize = d_g_state.d_keyval_pos_arr[index].valSize;
			int valPos = d_g_state.d_keyval_pos_arr[index].valPos;
			val_t_arr[index-start].valSize = valSize;
			val_t_arr[index-start].val = (char*)d_g_state.d_sorted_vals_shared_buff + valPos;
		}   //for
		reduce2(key, val_t_arr, keySize, end-start, d_g_state);

	}//for
}



//----------------------------------------------
//start reduce
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
//8, start reduce
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
	//printData3<<<NUM_BLOCKS,NUM_THREADS>>>(*d_g_state);
	ReducePartitioner<<<NUM_BLOCKS,NUM_THREADS>>>(*d_g_state);
	cudaThreadSynchronize(); 

	DoLog("DONE");
}//void

/*void *Panda_CPU_Map(void *ptr){
return NULL;
}*/

void * Panda_Map(void *ptr){
		
	thread_info_t *thread_info = (thread_info_t *)ptr;
		
	if(thread_info->device_type == GPU_ACC){
		
		gpu_context *d_g_state = (gpu_context *)(thread_info->d_g_state);
		
		DoLog("Init GPU Device");
		InitGPUDevice(thread_info);
				
		//int tid = thread_info->tid;
		//cudaSetDevice(tid % num_gpus);        // "% num_gpus" allows more CPU threads than GPU devices

		int gpu_id;
		cudaGetDevice(&gpu_id);
		DoLog("gpu_id in current Panda_Map:%d",gpu_id);
		
		DoLog("Init GPU MapReduce");
		InitGPUMapReduce3(d_g_state);
		
		DoLog("Start GPU Map Tasks");
		StartGPUMap(d_g_state);
		
		//d_g_state->d_intermediate_keyval_total_count;
		//checkCudaErrors(cudaMemset(d_g_state->d_intermediate_keyval_total_count,0,d_g_state->num_input_record*sizeof(int)));

		StartGPUShuffle(d_g_state);
		
	}//if
		
	if(thread_info->device_type == CPU_ACC){

		cpu_context *d_g_state = (cpu_context *)(thread_info->d_g_state);
		DoLog("Init CPU Device");
		InitCPUDevice(thread_info);
		
		DoLog("Init CPU MapReduce");
		InitCPUMapReduce(d_g_state);

		DoLog("Start CPU Map Tasks");
		StartCPUMap(d_g_state);
		//d_g_state->d_intermediate_keyval_total_count;
		//checkCudaErrors(cudaMemset(d_g_state->d_intermediate_keyval_total_count,0,d_g_state->num_input_record*sizeof(int)));
		StartCPUShuffle(d_g_state);
		
	}	
			
	return NULL;
}//FinishMapReduce2(d_g_state);


void FinishMapReduce(Spec_t* spec)
{
	DoLog( "=====finish map/reduce=====");
}//void


void FinishMapReduce2(gpu_context* state)
{
	
	size_t total_mem,avail_mem, heap_limit;
	checkCudaErrors(cudaMemGetInfo( &avail_mem, &total_mem ));
	DoLog("avail_mem:%d",avail_mem);
	DoLog( "=====finish map/reduce=====");
	
}//void





#endif //__PANDALIB_CU__