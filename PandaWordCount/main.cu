/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: main.cu 
	Time: 2012-07-01 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */


#include "Panda.h"
#include "Global.h"
#include <ctype.h>
//#include <sys/time.h>

//-----------------------------------------------------------------------
//usage: WordCount datafile
//param: datafile 
//-----------------------------------------------------------------------



void word_count_config_input(thread_info_t *thread_info){
/*
		char str[256];
		char strInput[10100];
		FILE *myfp;
		char *fn = thread_info->file_name;
		gpu_context *d_g_state = thread_info->d_g_state;

		myfp = fopen(fn, "r");
		int iKey = 0;
		int totalLen = 0;
		
		while(fgets(str,sizeof(str),myfp) != NULL)
		{
			for (int i = 0; i < strlen(str); i++)
			str[i] = toupper(str[i]);
			//printf("%s\t len:%d\n", str,strlen(str));
			strcpy((strInput + totalLen),str);
			totalLen += (int)strlen(str);
			if(totalLen>1000){
			//printf("strLen:%s\n",strInput);
			totalLen = 100;
			//TODO
			AddMapInputRecord2(d_g_state, &iKey, strInput, sizeof(int), totalLen);
			totalLen=0;
			iKey++;
			if (iKey%100==0)
				printf("iKey:%d\n",iKey);
			}//if
		}//while
		fclose(myfp);
		printf("%s input:%s  \ttotal lines:%d\n",thread_info->device_name, fn,iKey);
*/
}
		
int main(int argc, char** argv)
{		
	
	if (argc != 6)
	{	
		printf("usage: %s [text file][num gpu][num cpu groups][num_mappers][cpu/gpu work ratio]\n", argv[0]);
		exit(-1);	
	}//if
	DoLog("run %s  %s ...\n",argv[0],argv[1]);
	char *fn = argv[1];
	int num_gpus = atoi(argv[2]);
	int num_cpus_groups = atoi(argv[3]);
	int num_mappers = atoi(argv[4]);
	float ratio = atof(argv[5]);
	
	DoLog("configure input data for Panda job");
	char str[256];
	char strInput[10100];
	FILE *myfp;
	
	myfp = fopen(fn, "r");
	int iKey = 0;
	int totalLen = 0;

	thread_info_t *thread_info = (thread_info_t*)malloc(sizeof(thread_info_t)*(num_gpus + num_cpus_groups));

	for (int dev_id = 0; dev_id < num_gpus; dev_id++){
		//configure gpu job
		job_configuration *gpu_job_conf = (job_configuration*)malloc(sizeof(job_configuration));
		gpu_job_conf->num_gpus = num_gpus;
		gpu_job_conf->num_mappers = num_mappers;
		gpu_job_conf->auto_tuning = false;
		gpu_job_conf->ratio = (double)ratio;

		//construct input data
		while(fgets(str,sizeof(str),myfp) != NULL)
		{
			for (int i = 0; i < strlen(str); i++)
			str[i] = toupper(str[i]);
			
			strcpy((strInput + totalLen),str);
			totalLen += (int)strlen(str);
			if(totalLen>1000){
			
			totalLen = 100;
			AddPandaTask(gpu_job_conf, &iKey, strInput, sizeof(int), totalLen);
			totalLen=0;
			iKey++;
			if (iKey%100==0)
				printf("iKey:%d\n",iKey);
			}//if
		}//while

		fclose(myfp);
		DoLog("input:%s  \ttotal lines:%d\n",fn,iKey);
		DoLog("configure resources for Panda job");

		//configure panda worker
		thread_info[dev_id].job_conf = gpu_job_conf;
		thread_info[dev_id].device_type = GPU_ACC;
		
	}//for


	for (int dev_id=0; dev_id < num_cpus_groups; dev_id++){

		job_configuration *cpu_job_conf = GetJobConf();
		cpu_job_conf->num_cpus_groups = num_cpus_groups;
		cpu_job_conf->num_cpus_cores = getCPUCoresNum();

		while(fgets(str,sizeof(str),myfp) != NULL)
		{
			for (int i = 0; i < strlen(str); i++)
			str[i] = toupper(str[i]);
			
			strcpy((strInput + totalLen),str);
			totalLen += (int)strlen(str);
			if(totalLen>1000){
			
			totalLen = 100;
			AddPandaTask(cpu_job_conf, &iKey, strInput, sizeof(int), totalLen);
			totalLen=0;
			iKey++;
			if (iKey%100==0)
				printf("iKey:%d\n",iKey);
			}//if
		}//while

		fclose(myfp);
		DoLog("input:%s  \ttotal lines:%d\n",fn,iKey);
		DoLog("configure resources for Panda job");
		
		thread_info[dev_id].job_conf = cpu_job_conf;
		thread_info[dev_id].device_type = CPU_ACC;
		
	}//for

	PandaMetaScheduler(thread_info, num_gpus, num_cpus_groups);
	
	return 0;
		
}//		