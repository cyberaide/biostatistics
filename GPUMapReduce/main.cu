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



void word_count_config_input(thread_info_t *thread_info){

		char str[256];
		char strInput[10100];
		FILE *myfp;
		char *fn = thread_info->file_name;
		d_global_state *d_g_state = thread_info->d_g_state;

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
}
		
int main(int argc, char** argv) 
{		
	if (argc != 3)
	{	
		printf("usage: %s [data file1][data file2]\n", argv[0]);
		exit(-1);	
	}//if
	printf("start %s  %s  %s\n",argv[0],argv[1],argv[2]);
		
	/*	
	TimeVal_t allTimer;
	startTimer(&allTimer);
	TimeVal_t preTimer;
	startTimer(&preTimer);
	*/	
		
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
		
	pthread_t *no_threads = (pthread_t*)malloc(sizeof(pthread_t)*num_gpus);
	thread_info_t *thread_info = (thread_info_t*)malloc(sizeof(thread_info_t)*num_gpus);
		
	for (int i=0; i<num_gpus; i++){
		thread_info[i].tid = i;
		thread_info[i].file_name = argv[i+1];
		thread_info[i].num_gpus = num_gpus;
		thread_info[i].device_type = GPU_ACC;
		
		cudaDeviceProp gpu_dev;
		cudaGetDeviceProperties(&gpu_dev, i);
		DoLog("Configure Device ID:%d: Device Name:%s\n", i, gpu_dev.name);
		thread_info[i].device_name = gpu_dev.name;
		d_global_state *d_g_state = GetDGlobalState();
		thread_info[i].d_g_state = d_g_state;
		
		word_count_config_input(&thread_info[i]);
		
		if (pthread_create(&(no_threads[i]),NULL,Panda_Map,(char *)&(thread_info[i]))!=0) 
			perror("Thread creation failed!\n");
		////////////////////////////////////////////////////
		//configuration for Panda MapRedduce
		
	}//for num_gpus
		
	for (int i=0; i<num_gpus; i++){
		void *exitstat;
		if (pthread_join(no_threads[i],&exitstat)!=0) perror("joining failed");
	}//for
		
	//TODO
		
	checkCudaErrors(cudaSetDevice(thread_info[1].tid % num_gpus));        // "% num_gpus" allows more CPU threads than GPU devices
	checkCudaErrors(cudaDeviceReset());

	Panda_Shuffle(thread_info[0].d_g_state, thread_info[1].d_g_state);
	//printf("printData3 len:%d\n",thread_info[1].d_g_state->d_sorted_keyvals_arr_len);
	//printData3<<<NUM_BLOCKS,NUM_THREADS>>>(*(thread_info[1].d_g_state));
	cudaThreadSynchronize(); 
	//printf("printData3 Done\n");
	Panda_Reduce(&thread_info[1]);

	return 0;

}//