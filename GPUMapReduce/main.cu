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


//-----------------------------------------------------------------------
//usage: WordCount datafile
//param: datafile 
//-----------------------------------------------------------------------

void * GPU_MapReduce(void *ptr){

	thread_info_t *thread_info = (thread_info_t *)ptr;
	int tid = thread_info->tid;
	char *fn = thread_info->file_name;
	int num_gpus = thread_info->num_gpus;
	cudaSetDevice(tid % num_gpus);        // "% num_gpus" allows more CPU threads than GPU devices
	int gpu_id;
    cudaGetDevice(&gpu_id);
	printf("tid:%d num_gpus:%d gpu_id:%d fn:%s\n",tid,num_gpus,gpu_id,fn);

	FILE *myfp;
	myfp = fopen(fn, "r");

	//configuration for Panda MapRedduce
	d_global_state *d_g_state = GetDGlobalState();
	//spec->workflow = MAP_GROUP;
	
	char str[256];
	char strInput[10100];
	//FILE *myfp;
	//myfp = fopen(argv[1], "r");
	

	int iKey = 0;
	int totalLen = 0;

	while(fgets(str,sizeof(str),myfp) != NULL)
    {
		for (int i = 0; i < strlen(str); i++)
		str[i] = toupper(str[i]);
		//printf("%s\t len:%d\n", str,strlen(str));
		strcpy((strInput + totalLen),str);
		totalLen += (int)strlen(str);
		if(totalLen>10000){
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
	printf("input:%s  \ttotal lines:%d\n",fn,iKey);
	
	//----------------------------------------------
	//map/reduce
	//----------------------------------------------
	MapReduce2(d_g_state);
	
	//endTimer("all", &allTimer);

	//----------------------------------------------
	//finish
	//----------------------------------------------

	FinishMapReduce2(d_g_state);
	//cudaFree(d_filebuf);
	
	//handle the buffer different
	//free(h_filebuf);
	//handle the buffer different
	return NULL;
}

int main( int argc, char** argv) 
{
	
	if (argc != 3)
	{
		printf("usage: %s [data file1][data file2]\n", argv[0]);
		exit(-1);	
	}//if
	
	printf("start %s  %s..%s.\n",argv[0],argv[1],argv[2]);

	/*
	TimeVal_t allTimer;
	startTimer(&allTimer);
	TimeVal_t preTimer;
	startTimer(&preTimer);
	*/

	pthread_t no_threads[2];
	thread_info_t thread_info[2];
	
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	

	for (int i=0; i<num_gpus; i++){
		thread_info[i].file_name = argv[i+1];
		thread_info[i].num_gpus = num_gpus;
		thread_info[i].tid = i;
		cudaDeviceProp gpu_dev;
		cudaGetDeviceProperties(&gpu_dev, i);
        printf("   %d: %s\n", i, gpu_dev.name);
		if (pthread_create(&(no_threads[i]),NULL,GPU_MapReduce,(char *)&(thread_info[i]))!=0) 
			perror("Thread creation failed!\n");
	}//for

	for (int i=0; i<num_gpus;i++){
	void *exitstat;
	if (pthread_join(no_threads[i],&exitstat)!=0)
	perror("joining failed");
	}//for
	
	return 0;
}
