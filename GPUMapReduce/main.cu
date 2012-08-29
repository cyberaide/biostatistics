/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: main.cu 
	First Version:	2012-07-01 V0.1
	Current Version: V0.3	
	Last Updates:   2012-8-29

	Developer: Hui Li (lihui@indiana.edu)
	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */


#include "Panda.h"
#include "Global.h"
#include <ctype.h>

//-----------------------------------------------------------------------
//	Panda Matrix Multiplication
//	Data: 8/29/2012
//-----------------------------------------------------------------------

static float *GenMatrix(int M_ROW_COUNT, int M_COL_COUNT, float init)
{
        float *matrix = (float*)malloc(sizeof(float)*M_ROW_COUNT*M_COL_COUNT);
        //srand(time(0));
		//memset(matrix,1.0,sizeof(float)*M_ROW_COUNT*M_COL_COUNT);
		for (int i = 0; i < M_ROW_COUNT; i++)
                for (int j = 0; j < M_COL_COUNT; j++)
                        //matrix[i*M_COL_COUNT+j] = (float)(rand() % 100);
						matrix[i*M_COL_COUNT+j] = init;
		return matrix;
}

static float *RotateMatrix(float *matrix, int rowCount, int colCount)
{
        float *m = (float*)malloc(sizeof(float)*rowCount*colCount);
        for (int i = 0; i < rowCount; i++)
                for (int j = 0; j < colCount; j++)
                                m[i * colCount + j] = matrix[i + colCount * j];
        return m;
}//static float
		
int main(int argc, char** argv)
{		
	if (argc != 8)
	{	
		printf("usage: %s [rowNum][colNum][num gpus][num cpu groups][num_mappers][cpu/gpu work ratio][sample rate]\n", argv[0]);
		exit(-1);	
	}//if
	
	DoLog("configure input data for Panda job");

	int ROW_NUM = atoi(argv[1]);
	int COL_NUM = atoi(argv[2]);
	int num_gpus = atoi(argv[3]);
	int num_cpus_groups = atoi(argv[4]);
	int num_mappers = atoi(argv[5]);
	float ratio = atof(argv[6]);
	int sample_rate = atoi(argv[7]);

	double t1 = PandaTimer();

	float *matrix1 = GenMatrix(ROW_NUM,COL_NUM, 1.0);
	float *tmpMatrix2 = GenMatrix(COL_NUM,ROW_NUM, 1.0);
    float *matrix2 = RotateMatrix(tmpMatrix2,COL_NUM,ROW_NUM);
	float *matrix3 = GenMatrix(ROW_NUM,COL_NUM, 0.0);

	double t2 = PandaTimer();
	DoLog("load matrices  num_gpus:%d  num_cpus_groups:%d", num_gpus, num_cpus_groups);
	double t3 = PandaTimer();

	MM_KEY_T key;
    MM_VAL_T val;
	val.row_dim = ROW_NUM;
    val.col_dim = COL_NUM;
	val.bz = BLOCK_SIZE;

	int start_row_id, end_row_id;
	start_row_id = end_row_id = 0;

	int cpu_task_num = (int)(ROW_NUM*ratio);
	int gpu_task_num = ROW_NUM - cpu_task_num;
	//panda worker
	thread_info_t *thread_info = (thread_info_t*)malloc(sizeof(thread_info_t)*(num_gpus + num_cpus_groups));

	for (int dev_id=0; dev_id<num_cpus_groups; dev_id++){
		if (cpu_task_num == 0)
			break;

		//panda job
		job_configuration *cpu_job_conf = GetJobConf();
		cpu_job_conf->num_cpus_groups = num_cpus_groups;
		cpu_job_conf->num_cpus_cores = getCPUCoresNum();

		int partitions = cpu_job_conf->num_cpus_cores*num_cpus_groups;
		int task_per_partition = ((cpu_task_num)/partitions);

		key.h_matrix1 = matrix1;
		key.h_matrix2 = matrix2;
		key.h_matrix3 = matrix3;

		key.matrix1 = NULL;
		key.matrix2 = NULL;
		key.matrix3 = NULL;

		for (int i=0;i<partitions;i++){
			start_row_id = task_per_partition*i;
			end_row_id = start_row_id+task_per_partition;
			val.row = start_row_id;
			val.col = end_row_id;
			if( i==(partitions-1) )
				val.col = cpu_task_num;

			if (val.col>val.row)
				AddPandaTask(cpu_job_conf, &key, &val, sizeof(MM_KEY_T), sizeof(MM_VAL_T));
		}//for

		thread_info[dev_id].job_conf = cpu_job_conf;
		thread_info[dev_id].device_type = CPU_ACC;

	}//for
		
	for (int dev_id=0; dev_id<num_gpus; dev_id++){

		job_configuration *gpu_job_conf = GetJobConf();
		gpu_job_conf->num_gpus = num_gpus;
		gpu_job_conf->num_mappers = num_mappers;
		gpu_job_conf->auto_tuning = false;
		gpu_job_conf->ratio = (double)ratio;
		gpu_job_conf->auto_tuning_sample_rate = sample_rate;

		if ( dev_id == 0 )
			start_row_id = cpu_task_num;	
		else 
			start_row_id = cpu_task_num+(dev_id)*(gpu_task_num/num_gpus);

		end_row_id = start_row_id + gpu_task_num/num_gpus;
		if ( dev_id == (num_gpus - 1) )
			end_row_id = cpu_task_num + gpu_task_num;

		//copy to data into different GPU device
		cudaSetDevice(dev_id);  

		int matrixSize = sizeof(float)*ROW_NUM*COL_NUM;
		float *d_matrix1 = NULL;
		cudaMalloc((void **)&d_matrix1,matrixSize);
		cudaMemcpy(d_matrix1, matrix1, matrixSize, cudaMemcpyHostToDevice);
		float *d_matrix2 = NULL;
		cudaMalloc((void**)&d_matrix2,matrixSize);
		cudaMemcpy(d_matrix2,matrix2,matrixSize,cudaMemcpyHostToDevice);
		float *d_matrix3 = NULL;
		cudaMalloc((void**)&d_matrix3,matrixSize);
		cudaMemcpy(d_matrix3,matrix3,matrixSize,cudaMemcpyHostToDevice);

		key.matrix1 = d_matrix1;
		key.matrix2 = d_matrix2;
		key.matrix3 = d_matrix3;

		key.h_matrix1 = NULL;
		key.h_matrix2 = NULL;
		key.h_matrix3 = NULL;

		if(end_row_id>start_row_id)
			for (int i = start_row_id/BLOCK_SIZE; i < (end_row_id + BLOCK_SIZE-1)/BLOCK_SIZE; i++)
			{
				val.row = i;
				for (int j = 0; j < (COL_NUM+BLOCK_SIZE-1)/BLOCK_SIZE; j++)
				{
					val.col = j;
					AddPandaTask(gpu_job_conf, &key, &val, sizeof(MM_KEY_T), sizeof(MM_VAL_T));
				}//for
			}//for
			thread_info[num_cpus_groups + dev_id].job_conf = gpu_job_conf;
			thread_info[num_cpus_groups + dev_id].device_type = GPU_ACC;

	}

	double t4 = PandaTimer();
	PandaMetaScheduler(thread_info, num_gpus, num_cpus_groups);
	cudaThreadSynchronize();
	double t5 = PandaTimer();

	DoLog("GenMatrix:%f",t2-t1);
	DoLog("Copy to GPU:%f",t4-t3);
	DoLog("Compute:%f",t5-t4);
	char str[128];
	sprintf(str,"matrix size:%d copy to GPU:%f  compute:%f cpu/gpu ratio:%f sample rate:%d", ROW_NUM, t3-t2, t5-t4, (double)ratio,sample_rate);
	DoDiskLog(str);

	return 0;
}//		