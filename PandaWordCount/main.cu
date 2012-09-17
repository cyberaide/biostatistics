/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: main.cu 
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2012-09-02

	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 */


#include "Panda.h"
#include "UserAPI.h"
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>

//#include <sys/time.h>

//-----------------------------------------------------------------------
//usage: WordCount datafile
//param: datafile 
//-----------------------------------------------------------------------
		
int main(int argc, char** argv)
{		
	
	if (argc != 6)
	{	
		printf("usage: %s [text file][num gpu][num cpu groups][num_mappers][cpu/gpu work ratio]\n", argv[0]);
		exit(-1);	
	}//if
	ShowLog("%s %s",argv[0],argv[1]);
	char *fn = argv[1];
	int num_gpus = atoi(argv[2]);
	int num_cpus_groups = atoi(argv[3]);
	int num_mappers = atoi(argv[4]);
	float ratio = atof(argv[5]);
	
	char str[256];
	char strInput[10100];
	FILE *myfp;
	
	myfp = fopen(fn, "r");

	struct stat filestatus;
	stat( fn, &filestatus );

	int iKey = 0;
	int totalLen = 0;

	thread_info_t *thread_info = (thread_info_t*)malloc(sizeof(thread_info_t)*(num_gpus + num_cpus_groups));

	int cpuWorkLoad = filestatus.st_size * ratio;
	int gpuWorkLoad = filestatus.st_size * (1 - ratio);
	ShowLog("input file size:%d cpu workload:%d gpu workload:%d", filestatus.st_size, cpuWorkLoad, gpuWorkLoad );
	double t3 = PandaTimer();

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
		
			if(totalLen>2000){
			
			//totalLen = 100;
			AddPandaTask(gpu_job_conf, &iKey, strInput, sizeof(int), totalLen);
			gpuWorkLoad -= totalLen;
			
			totalLen=0;
			iKey++;
			/*if (iKey%100==0)
				printf("iKey:%d\n",iKey);*/
			}//if
			if (gpuWorkLoad <=100)
				break;
		}//while

		//fclose(myfp);

		ShowLog("configure resources for Panda gpu jobs. input:%s  total map tasks:%d",fn,iKey);

		//configure panda worker
		thread_info[dev_id].job_conf = gpu_job_conf;
		thread_info[dev_id].device_type = GPU_ACC;
		
	}//for


	for (int dev_id=num_gpus; dev_id < num_gpus+num_cpus_groups; dev_id++){

		job_configuration *cpu_job_conf = CreateJobConf();
		cpu_job_conf->num_cpus_groups = num_cpus_groups;
		cpu_job_conf->num_cpus_cores = getCPUCoresNum();

		while(fgets(str,sizeof(str),myfp) != NULL)
		{
			for (int i = 0; i < strlen(str); i++)
			str[i] = toupper(str[i]);
			
			strcpy((strInput + totalLen),str);
			totalLen += (int)strlen(str);
			if(totalLen>500){
			
			//totalLen = 100;
			AddPandaTask(cpu_job_conf, &iKey, strInput, sizeof(int), totalLen);
			totalLen=0;
			iKey++;
			/*if (iKey%100==0)
				printf("iKey:%d\n",iKey);*/
			}//if
		}//while
		
		ShowLog("configure resources for Panda cpu jobs. input:%s  total map tasks:%d",fn,iKey);
		
		thread_info[dev_id].job_conf = cpu_job_conf;
		thread_info[dev_id].device_type = CPU_ACC;
		
	}//for
	fclose(myfp);

	double t4 = PandaTimer();
	panda_context *panda = CreatePandaContext();
	panda->num_gpus = num_gpus;
	panda->num_cpus_groups = num_cpus_groups;
	panda->ratio = ratio;
	PandaMetaScheduler(thread_info, panda);
	double t5 = PandaTimer();

	ShowLog("Copy Input Data:%f",t4-t3);
	ShowLog("Compute:%f",t5-t4);
	char strLog[128];
	sprintf(strLog,"data size:%d KB copy data:%f compute:%f cpu/gpu ratio:%f", filestatus.st_size/1024, t4-t3, t5-t4, (double)ratio);
	DoDiskLog(strLog);

	return 0;
		
}//		