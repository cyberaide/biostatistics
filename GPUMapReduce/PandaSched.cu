/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: PandaSched.cu 
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2012-09-02

	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */

#ifndef _PANDASCHED_CU_
#define _PANDASCHED_CU_

// includes, kernels
#include "Panda.h"


//--------------------------------------------------
//  PandaMetaScheduler
//--------------------------------------------------
/*
 * 1) input a set of panda worker (thread)
 * 2) each panda worker consist of one panda job and pand device
 * 3) copy input data from pand job to pand device 
 */



//For version 0.3
void PandaMetaScheduler(thread_info_t *thread_info, panda_context *panda){

	int num_gpus = panda->num_gpus;
	int num_cpus_groups = panda->num_cpus_groups;
	float ratio = panda->ratio;
	
	pthread_t *no_threads = (pthread_t*)malloc(sizeof(pthread_t)*(num_gpus + num_cpus_groups));
	
	int assigned_gpu_id = 0;
	int assigned_cpu_group_id = 0;
	for (int dev_id=0; dev_id<(num_gpus + num_cpus_groups); dev_id++){

		if (thread_info[dev_id].device_type == GPU_ACC){
			
			job_configuration* gpu_job_conf = (job_configuration*)(thread_info[dev_id].job_conf);
			gpu_context *d_g_state = GetGPUContext();
			d_g_state->num_mappers = gpu_job_conf->num_mappers;
			d_g_state->num_reducers = gpu_job_conf->num_reducers;
			d_g_state->num_gpus = num_gpus;
			d_g_state->gpu_id = assigned_gpu_id;

			thread_info[dev_id].tid = dev_id;
			thread_info[dev_id].d_g_state = d_g_state;

			DoLog("Assigned Dev_ID:[%d] GPU_ACC TID:%d",assigned_gpu_id,thread_info[dev_id].tid);
			assigned_gpu_id++;
		}//if

		if (thread_info[dev_id].device_type == CPU_ACC){
			
			cpu_context *d_g_state = GetCPUContext();
			d_g_state->cpu_group_id = assigned_cpu_group_id;
			thread_info[dev_id].tid = dev_id;
			thread_info[dev_id].d_g_state = d_g_state;

			DoLog("Assigned Dev_ID:[%d] CPU_ACC TID:%d",dev_id,thread_info[dev_id].tid);
			assigned_cpu_group_id++;
		}//if
	}//for
	
	///////////////////////////////////////////////////
		
	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){

		if (thread_info[dev_id].device_type == GPU_ACC){

				job_configuration *gpu_job_conf = (job_configuration *)(thread_info[dev_id].job_conf);
				int start_task_id = 0;
				int end_task_id = gpu_job_conf->num_input_record;
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);

				AddMapInputRecordGPU(d_g_state,(gpu_job_conf->input_keyval_arr), start_task_id,end_task_id);
				
		}//if
	
		if (thread_info[dev_id].device_type == CPU_ACC){

				job_configuration *cpu_job_conf = (job_configuration *)(thread_info[dev_id].job_conf);
				int start_task_id = 0;
				int end_task_id = cpu_job_conf->num_input_record;
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);
				
				AddMapInputRecordCPU(d_g_state,(cpu_job_conf->input_keyval_arr),start_task_id, end_task_id);
				
		}//if
	}//for
	
	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){
		if (pthread_create(&(no_threads[dev_id]), NULL, Panda_Map, (char *)&(thread_info[dev_id])) != 0) 
			perror("Thread creation failed!\n");
	}//for

	for (int i = 0; i < num_gpus + num_cpus_groups; i++){
		void *exitstat;
		if (pthread_join(no_threads[i],&exitstat)!=0) perror("joining failed");
	}//for

	//DoLog("start to merge results of GPU's and CPU's device to Panda scheduler");
	for (int i = 0; i < num_gpus+num_cpus_groups; i++){

		if (thread_info[i].device_type == CPU_ACC)
			PandaShuffleMergeCPU((panda_context*)panda, (cpu_context*)(thread_info[i].d_g_state));

		if (thread_info[i].device_type == GPU_ACC)
			PandaShuffleMergeGPU((panda_context*)panda, (gpu_context*)(thread_info[i].d_g_state));
			
	}//for
	
	//TODO reduce task ratio 
	int num_sorted_intermediate_record = panda->sorted_keyvals_arr_len;
	int records_per_device = num_sorted_intermediate_record/(num_gpus + num_cpus_groups*ratio);
	
	int *split = (int*)malloc(sizeof(int)*(num_gpus+num_cpus_groups));
	
	for (int i=0; i<num_gpus; i++){
	
				if (i==0) 
				split[0] = records_per_device;
				else
				split[i] = split[i-1] + records_per_device;
				
	}//for
	
	for (int i=num_gpus; i<num_gpus+num_cpus_groups; i++){
	
				if (i==0) 
				split[0] = records_per_device*ratio;
				else 
				split[i] = split[i-1] + records_per_device*ratio;
								
	}//for
	split[num_gpus + num_cpus_groups-1] = num_sorted_intermediate_record;

	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){
	
		int start_row_id = 0;
		if (dev_id>0) start_row_id = split[dev_id-1];
		int end_row_id = split[dev_id];
				
		if (thread_info[dev_id].device_type == GPU_ACC){
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);
				AddReduceInputRecordGPU(d_g_state,(panda->sorted_intermediate_keyvals_arr), start_row_id, end_row_id);
		}//if

		if (thread_info[dev_id].device_type == CPU_ACC){
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);
				AddReduceInputRecordCPU(d_g_state,(panda->sorted_intermediate_keyvals_arr), start_row_id, end_row_id);
		}//if
	}//for

	for (int dev_id = 0; dev_id < (num_gpus+num_cpus_groups); dev_id++){
		if (pthread_create(&(no_threads[dev_id]),NULL,Panda_Reduce,(char *)&(thread_info[dev_id]))!=0) 
			perror("Thread creation failed!\n");
	}//for
		
	for (int i=0; i < num_gpus + num_cpus_groups; i++){
		void *exitstat;
		if (pthread_join(no_threads[i],&exitstat)!=0) perror("joining failed");
	}//for

	//TODO Reduce Merge
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	//Panda_Reduce_Merge(&thread_info[num_gpus-1]);															 //
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	
	int total_output_records = 0;
	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){
	
		if (thread_info[dev_id].device_type == GPU_ACC){
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);
				total_output_records += d_g_state->d_reduced_keyval_arr_len;
		}//if

		if (thread_info[dev_id].device_type == CPU_ACC){
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);
				total_output_records += d_g_state->sorted_keyvals_arr_len;
		}//if
		
	}//for
	DoLog("number of reduce output:%d\n",total_output_records);
	DoLog("=====panda mapreduce job finished=====");

}//PandaMetaScheduler


//Scheduler for version 0.2 depressed
void Start_Panda_Job(job_configuration *job_conf){
#ifdef DEV_MODE	
	int num_gpus = job_conf->num_gpus;
	int num_cpus_cores = job_conf->num_cpus_cores;
	int num_cpus_group = job_conf->num_cpus_groups;

	panda_context *panda = GetPandaContext();
	
	panda->num_gpus = num_gpus;
	panda->num_cpus_groups = num_cpus_group;
	DoLog("Start num_gpus:%d  num_cpus_groups:%d", num_gpus, num_cpus_group);
	pthread_t *no_threads = (pthread_t*)malloc(sizeof(pthread_t)*(num_gpus + num_cpus_group));
	thread_info_t *thread_info = (thread_info_t*)malloc(sizeof(thread_info_t)*(num_gpus + num_cpus_group));
	
	for (int i=0; i<num_gpus; i++){
		thread_info[i].tid = i;
		//thread_info[i].file_name = argv[i+1];
		thread_info[i].num_gpus = num_gpus;
		thread_info[i].device_type = GPU_ACC;
		
		cudaDeviceProp gpu_dev;
		cudaGetDeviceProperties(&gpu_dev, i);
		
		//DoLog("Configure Device ID:%d: Device Name:%s MultProcessorCount:%d sm_per_multiproc:%d", i, gpu_dev.name,gpu_dev.multiProcessorCount,sm_per_multiproc);

		thread_info[i].device_name = gpu_dev.name;
		gpu_context *d_g_state = GetGPUContext();
		d_g_state->matrix_size = job_conf->matrix_size;
		d_g_state->num_mappers = job_conf->num_mappers;
		d_g_state->num_reducers = job_conf->num_reducers;
		thread_info[i].d_g_state = d_g_state;
	}//for num_gpus
	
	
	for (int i=num_gpus; i<num_gpus+num_cpus_group; i++){
		thread_info[i].tid = i;
		thread_info[i].device_type = CPU_ACC;
		cpu_context *d_g_state = GetCPUContext();
		d_g_state->num_cpus_cores = num_cpus_cores;
		thread_info[i].d_g_state = d_g_state;
	}//for


	///////////////////////////////////////////////////
	double ratio = 10.0;
	ratio = (double)(job_conf->ratio);
	if (job_conf->auto_tuning){
		ratio = (AutoTuningScheduler(job_conf));
		job_conf->ratio = ratio;
	}//if
	
	//////////////////////////////////////////////////
	DoLog("num_gpus:%d num_cpus_group:%d num_input_record:%d sizeof(int):%d  ratio:%f\n", num_gpus, num_cpus_group,job_conf->num_input_record,sizeof(int),ratio);

	int *split = NULL;
	split = (int *)malloc(sizeof(int)*(num_gpus+num_cpus_group));
	
	int num_input_record = job_conf->num_input_record;
	int records_per_device = (int)(num_input_record/(num_gpus*ratio+num_cpus_group));
	for (int i=0;i<num_gpus;i++){
			split[i] = (int)(records_per_device*ratio*(i+1));
	}//for

	for (int i=num_gpus;i<num_gpus+num_cpus_group;i++){
			split[i] = (int)(records_per_device*ratio*(num_gpus)+(i+1)*records_per_device);
	}//for
	split[num_gpus+num_cpus_group-1] = num_input_record;

	printf("---	split:num_input_record:%d records_per_device:%d  ",num_input_record, records_per_device);
	for (int i=0;i<num_gpus+num_cpus_group;i++)
		printf("%d\t",split[i]);
	printf("\n");

	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_group); dev_id++){
		int start_row_id = 0;
		if (dev_id>0) start_row_id = split[dev_id-1];

		int end_id = split[dev_id];
		
		if (thread_info[dev_id].device_type == GPU_ACC){
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);
				//for (int i=start_row_id; i<end_id; i++){
				//printf(":%s  keySize:%d",job_conf->input_keyval_arr[i].val, job_conf->input_keyval_arr[i].valSize);
				AddMapInputRecordGPU(d_g_state,(job_conf->input_keyval_arr), start_row_id,end_id);
				//}//for
			}//if

		if (thread_info[dev_id].device_type == CPU_ACC){
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);
				//for (int i=start_row_id;i<end_id;i++){
				AddMapInputRecordCPU(d_g_state,(job_conf->input_keyval_arr),start_row_id, end_id);
				//}//for
		}//if
	}//for

	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_group); dev_id++){
		//if (thread_info[dev_id].device_type == GPU_ACC){
		if (pthread_create(&(no_threads[dev_id]), NULL, Panda_Map, (char *)&(thread_info[dev_id])) != 0) 
			perror("Thread creation failed!\n");
	}//for

	for (int i=0; i < num_gpus + num_cpus_group; i++){
		void *exitstat;
		if (pthread_join(no_threads[i],&exitstat)!=0) perror("joining failed");
	}//for

	DoLog("start_row_id to merge!");
	for (int i = 0; i<num_gpus; i++){
		PandaShuffleMergeGPU((panda_context*)panda, (gpu_context*)(thread_info[i].d_g_state));
	}//for

	for (int i = num_gpus; i < num_gpus+num_cpus_group; i++){
		PandaShuffleMergeCPU((panda_context*)panda, (cpu_context*)(thread_info[i].d_g_state));
	}//for

	DoLog("totoal number of different intermediate records:%d",panda->sorted_keyvals_arr_len);
	//TOD smart job for reduce ratio

	//cudaThreadSynchronize();
	//static scheduling -- split the workload between devices 
	int num_sorted_intermediate_record = panda->sorted_keyvals_arr_len;
	records_per_device = num_sorted_intermediate_record/(num_gpus*10+num_cpus_group);
	for (int i=0;i<num_gpus;i++){
			split[i] = records_per_device*10*(i+1);
	}//for
	for (int i=num_gpus;i<num_gpus+num_cpus_group;i++){
			split[i] = records_per_device*10*(num_gpus)+(i+1)*records_per_device;
	}//for
	split[num_gpus + num_cpus_group-1] = num_sorted_intermediate_record;

	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_group); dev_id++){
		int start_row_id = 0;
		if (dev_id>0) start_row_id = split[dev_id-1];
		int end_id = split[dev_id];
				
		if (thread_info[dev_id].device_type == GPU_ACC){
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);
				//for (int i=start_row_id; i<end_id; i++){
				AddReduceInputRecordGPU(d_g_state,(panda->sorted_intermediate_keyvals_arr),start_row_id, end_id);
				//}//for

		}//if

		if (thread_info[dev_id].device_type == CPU_ACC){
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);
				//for (int i=start_row_id;i<end_id;i++){
				AddReduceInputRecordCPU(d_g_state,(panda->sorted_intermediate_keyvals_arr),start_row_id, end_id);
				//}//for
		}//if
		

	}//for

	for (int dev_id = 0; dev_id < (num_gpus+num_cpus_group); dev_id++){
		if (pthread_create(&(no_threads[dev_id]),NULL,Panda_Reduce,(char *)&(thread_info[dev_id]))!=0) 
			perror("Thread creation failed!\n");
	}//for
		
	for (int i=0; i < num_gpus + num_cpus_group; i++){
		void *exitstat;
		if (pthread_join(no_threads[i],&exitstat)!=0) perror("joining failed");
	}//for
	
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	//Panda_Reduce(&thread_info[num_gpus-1]);
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	DoLog("Finishing Panda MapReduce Job...");
	int total_output_records = 0;
	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_group); dev_id++){
	
		if (thread_info[dev_id].device_type == GPU_ACC){
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);
				total_output_records += d_g_state->d_reduced_keyval_arr_len;
		}//if

		if (thread_info[dev_id].device_type == CPU_ACC){
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);
				total_output_records += d_g_state->sorted_keyvals_arr_len;
		}//if
	}//for
	DoLog("there are :%d output records\n",total_output_records);
	DoLog("=====finish map/reduce=====");
#endif
}		

//Ratio = Tcpu/Tgpu
//Tcpu = (execution time on CPU cores for sampled tasks)/(#sampled tasks)
//Tgpu = (execution time on 1 GPU for sampled tasks)/(#sampled tasks)
//smart scheduler for auto tuning; measure the performance of sample data  

float AutoTuningScheduler(thread_info_t *thread_info, panda_context *panda){
	
	DoLog("AutoTuningScheduler");
	int num_gpus = panda->num_gpus;
	int num_cpus_cores = getCPUCoresNum();//job_conf->num_cpus;
	int num_cpus_groups = panda->num_cpus_groups;
	
	num_gpus = 1;
	num_cpus_groups = 1;

	pthread_t *no_threads = (pthread_t*)malloc(sizeof(pthread_t)*(num_gpus + num_cpus_groups));
	
	int cpu_sampled_tasks_num = 0;
	int gpu_sampled_tasks_num = 0;

	int start_row_id = 0;
	int end_row_id = 0;//job_conf->num_cpus_cores*2; //(job_conf->num_input_record/100);

	int cpu_index = -1;
	int gpu_index = -1;

	for (int tid=0; tid<num_gpus+num_cpus_groups; tid++){
	
		if (thread_info[tid].device_type == GPU_ACC){
			if (gpu_index>=0)
				continue;
			gpu_index = tid;

			gpu_context *d_g_state = GetGPUContext();
			d_g_state->num_gpus = num_gpus;
			thread_info[tid].d_g_state = d_g_state;

			job_configuration *gpu_job_conf = (job_configuration *)(thread_info[tid].job_conf);
			gpu_sampled_tasks_num = gpu_job_conf->num_input_record;
			start_row_id = 0;
			end_row_id = gpu_job_conf->num_input_record;
			AddMapInputRecordGPU(d_g_state,(gpu_job_conf->input_keyval_arr), start_row_id, end_row_id);
			
		}//if
		
		if (thread_info[tid].device_type == CPU_ACC){
			if (cpu_index>=0)
				continue;
			cpu_index = tid;

			cpu_context *d_g_state = GetCPUContext();
			//d_g_state->num_cpus_groups = num_cpus_groups;
			thread_info[tid].d_g_state = d_g_state;

			job_configuration *cpu_job_conf = (job_configuration *)(thread_info[tid].job_conf);
			cpu_sampled_tasks_num = cpu_job_conf->num_input_record;
			start_row_id = 0;
			end_row_id = cpu_job_conf->num_input_record;
			AddMapInputRecordCPU(d_g_state,(cpu_job_conf->input_keyval_arr), start_row_id, end_row_id);

		}//if
	}//for 
	
	//cpu_sampled_tasks_num = num_cpus_cores*job_conf->auto_tuning_sample_rate;
	//gpu_sampled_tasks_num = getGPUCoresNum()*job_conf->auto_tuning_sample_rate;
	//if (cpu_sampled_tasks_num>job_conf->num_input_record)
	//if (gpu_sampled_tasks_num>job_conf->num_input_record)
		
	double t1 = PandaTimer();
	Panda_Map((void *)&(thread_info[gpu_index]));
	double t2 = PandaTimer();
	//start_row_id cpu 
	Panda_Map((void *)&(thread_info[cpu_index]));
	double t3 = PandaTimer();
	
	double t_cpu = (t3-t2);///cpu_sampled_tasks_num;
	double t_gpu = (t2-t1);///gpu_sampled_tasks_num;

	if (t_gpu<0.0001)
		t_gpu=0.0001;
	
	//double ratio = (t_cpu*gpu_sampled_tasks_num)/(t_gpu*cpu_sampled_tasks_num);
	
	double ratio = (t_cpu)/(t_gpu);
	DoLog("cpu time:%f gpu time:%f ratio:%f", (t_cpu), (t_gpu), ratio);
	/*
	char log[128];
	sprintf(log,"	cpu_sampled_tasks:%d cpu time:%f cpu time per task:%f", cpu_sampled_tasks_num, t_cpu, t_cpu/(cpu_sampled_tasks_num));
	DoDiskLog(log);
	sprintf(log,"	gpu_sampled_tasks:%d gpu time:%f gpu time per task:%f	ratio:%f", gpu_sampled_tasks_num, t_gpu, t_gpu/(gpu_sampled_tasks_num), ratio);
	DoDiskLog(log);
	*/
	
	return (ratio);
	
}//void

void PandaDynamicMetaScheduler(thread_info_t *thread_info, panda_context *panda){

	int num_gpus = panda->num_gpus;
	int num_cpus_groups = panda->num_cpus_groups;
	float ratio = panda->ratio;
	
	pthread_t *no_threads = (pthread_t*)malloc(sizeof(pthread_t)*(num_gpus + num_cpus_groups));
	
	for (int dev_id=0; dev_id<(num_gpus + num_cpus_groups); dev_id++){

		int assigned_gpu_id = 0;
		if (thread_info[dev_id].device_type == GPU_ACC){
			
			job_configuration* gpu_job_conf = (job_configuration*)(thread_info[dev_id].job_conf);
			gpu_context *d_g_state = GetGPUContext();
			d_g_state->num_mappers = gpu_job_conf->num_mappers;
			d_g_state->num_reducers = gpu_job_conf->num_reducers;
			d_g_state->num_gpus = num_gpus;
			d_g_state->gpu_id = assigned_gpu_id;

			thread_info[dev_id].tid = dev_id;
			thread_info[dev_id].d_g_state = d_g_state;

			DoLog("Assigned Dev_ID:[%d] GPU_ACC TID:%d",assigned_gpu_id,thread_info[dev_id].tid);
			assigned_gpu_id++;
		}//if

		int cpu_group_id = 0;
		if (thread_info[dev_id].device_type == CPU_ACC){
			
			cpu_context *d_g_state = GetCPUContext();
			d_g_state->cpu_group_id = cpu_group_id;
			thread_info[dev_id].tid = dev_id;
			thread_info[dev_id].d_g_state = d_g_state;

			DoLog("Assigned Dev_ID:[%d] CPU_ACC TID:%d",dev_id,thread_info[dev_id].tid);
			cpu_group_id++;
		}//if
	}//for

	///////////////////////////////////////////////////
	
	
	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){

		if (thread_info[dev_id].device_type == GPU_ACC){

				job_configuration *gpu_job_conf = (job_configuration *)(thread_info[dev_id].job_conf);
				int start_row_id = 0;
				int end_id = gpu_job_conf->num_input_record;
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);

				AddMapInputRecordGPU(d_g_state,(gpu_job_conf->input_keyval_arr), start_row_id,end_id);
				
		}//if
	
		if (thread_info[dev_id].device_type == CPU_ACC){

				job_configuration *cpu_job_conf = (job_configuration *)(thread_info[dev_id].job_conf);
				int start_row_id = 0;
				int end_id = cpu_job_conf->num_input_record;
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);
				
				AddMapInputRecordCPU(d_g_state,(cpu_job_conf->input_keyval_arr),start_row_id, end_id);
				
		}//if
	}//for
	
	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){
		if (pthread_create(&(no_threads[dev_id]), NULL, Panda_Map, (char *)&(thread_info[dev_id])) != 0) 
			perror("Thread creation failed!\n");
	}//for

	for (int i = 0; i < num_gpus + num_cpus_groups; i++){
		void *exitstat;
		if (pthread_join(no_threads[i],&exitstat)!=0) perror("joining failed");
	}//for

	//DoLog("start to merge results of GPU's and CPU's device to Panda scheduler");
	for (int i = 0; i < num_gpus+num_cpus_groups; i++){

		if (thread_info[i].device_type == CPU_ACC)
			PandaShuffleMergeCPU((panda_context*)panda, (cpu_context*)(thread_info[i].d_g_state));

		if (thread_info[i].device_type == GPU_ACC)
			PandaShuffleMergeGPU((panda_context*)panda, (gpu_context*)(thread_info[i].d_g_state));
			
	}//for
	
	//TODO reduce task ratio 
	int num_sorted_intermediate_record = panda->sorted_keyvals_arr_len;
	int records_per_device = num_sorted_intermediate_record/(num_gpus + num_cpus_groups*ratio);
	
	int *split = (int*)malloc(sizeof(int)*(num_gpus+num_cpus_groups));
	
	for (int i=0; i<num_gpus; i++){
	
				if (i==0) 
				split[0] = records_per_device;
				else
				split[i] = split[i-1] + records_per_device;
				
	}//for
	
	for (int i=num_gpus; i<num_gpus+num_cpus_groups; i++){
	
				if (i==0) 
				split[0] = records_per_device*ratio;
				else 
				split[i] = split[i-1] + records_per_device*ratio;
								
	}//for
	split[num_gpus + num_cpus_groups-1] = num_sorted_intermediate_record;

	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){
	
		int start_row_id = 0;
		if (dev_id>0) start_row_id = split[dev_id-1];
		int end_row_id = split[dev_id];
				
		if (thread_info[dev_id].device_type == GPU_ACC){
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);
				AddReduceInputRecordGPU(d_g_state,(panda->sorted_intermediate_keyvals_arr), start_row_id, end_row_id);
		}//if

		if (thread_info[dev_id].device_type == CPU_ACC){
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);
				AddReduceInputRecordCPU(d_g_state,(panda->sorted_intermediate_keyvals_arr), start_row_id, end_row_id);
		}//if
	}//for

	for (int dev_id = 0; dev_id < (num_gpus+num_cpus_groups); dev_id++){
		if (pthread_create(&(no_threads[dev_id]),NULL,Panda_Reduce,(char *)&(thread_info[dev_id]))!=0) 
			perror("Thread creation failed!\n");
	}//for
		
	for (int i=0; i < num_gpus + num_cpus_groups; i++){
		void *exitstat;
		if (pthread_join(no_threads[i],&exitstat)!=0) perror("joining failed");
	}//for

	//TODO Reduce Merge
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	//Panda_Reduce_Merge(&thread_info[num_gpus-1]);															 //
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	
	int total_output_records = 0;
	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){
	
		if (thread_info[dev_id].device_type == GPU_ACC){
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);
				total_output_records += d_g_state->d_reduced_keyval_arr_len;
		}//if

		if (thread_info[dev_id].device_type == CPU_ACC){
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);
				total_output_records += d_g_state->sorted_keyvals_arr_len;
		}//if
		
	}//for
	DoLog("number of reduce output:%d\n",total_output_records);
	DoLog("=====panda mapreduce job finished=====");

}//PandaMetaScheduler


#endif // _PRESCHED_CU_
