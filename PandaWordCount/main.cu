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
	if (argc != 2)
	{	
		printf("usage: %s [data file1]\n", argv[0]);
		exit(-1);	
	}//if
	printf("run %s  %s ...\n",argv[0],argv[1]);
	
	job_configuration *job_conf = (job_configuration*)malloc(sizeof(job_configuration));
	job_conf->num_input_record = 0;
	job_conf->input_keyval_arr = NULL;
	
	DoLog("configure input data for Panda job");
	char str[256];
	char strInput[10100];
	FILE *myfp;
	char *fn = argv[1];
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
			AddPandaTask(job_conf, &iKey, strInput, sizeof(int), totalLen);
			totalLen=0;
			iKey++;
			if (iKey%100==0)
				printf("iKey:%d\n",iKey);
			}//if
		}//while
	fclose(myfp);
	printf("input:%s  \ttotal lines:%d\n",fn,iKey);

	DoLog("configure resources for Panda job");
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	
	/*
	struct timeval tim;
	gettimeofday(&tim,NULL);
	double t1 = tim.tv_sec+tim.tv_usec/1000000.0;
	*/

	job_conf->num_cpus = 12;
	job_conf->num_gpus = num_gpus;
	job_conf->num_cpus_groups = 1;

	Start_Panda_Job(job_conf);

	
	/*gettimeofday(&tim,NULL);
	double t2 = tim.tv_sec+tim.tv_usec/1000000.0;;
	//printf ("\telapsed wall clock time: %ld\n", (long) (t1 - t0));
	DoLog("\t job time:%f", t2-t1);*/

	return 0;
		
}//		