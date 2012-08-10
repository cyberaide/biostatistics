/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: map.cu 
	Time: 2012-07-01 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */

#ifndef __MAP_CU__
#define __MAP_CU__

#include "Panda.h"
#include "Global.h"

 
/*
__device__ int hash_func(char* str, int len)
{
	int hash, i;
	for (i = 0, hash=len; i < len; i++)
		hash = (hash<<4)^(hash>>28)^str[i];
	return hash;
}
*/

void cpu_map(void *KEY, void*VAL, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx){
		
		int wsize = 0;
		char *start;
		char *p = (char *)VAL;
		while(1)
		{
			start = p;
			for(;*p>='A' && *p<='Z';p++);
			*p='\0';
			++p;
			wsize=(int)(p-start);
			if (wsize>6){
				char *wkey = (char *) malloc (wsize);
				memcpy(wkey,start,wsize);
				
				int *wc = (int *) malloc (sizeof(int));
				*wc=1;
				CPUEmitIntermediate(wkey, wc, wsize, sizeof(int), d_g_state, map_task_idx);
				//printf("\t\tcpu_map: map_id:%d, key:%s\n", map_task_idx, wkey);
			}//if
			valSize = valSize - wsize;
			if(valSize<=0)
				break;
		}//while
		
}//map2



__device__ void map2(void *KEY, void*VAL, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){
		//printf("map2 TID:%d, key:%d, val:%s \n",TID,*(int*)KEY,(char *)VAL);
		int wsize = 0;
		char *start;
		char *p = (char *)VAL;
		while(1)
		{
			start = p;
			for(;*p>='A' && *p<='Z';p++);
			*p='\0';
			++p;
			wsize=(int)(p-start);
			if (wsize>6){
				char *wkey = (char *) malloc (wsize);
				memcpy(wkey,start,wsize);

				int *wc = (int *) malloc (sizeof(int));
				*wc=1;
				EmitIntermediate2(wkey, wc, wsize, sizeof(int), d_g_state, map_task_idx);
				//printf("\t\tmap2: TID:%d, index:%s\n",TID, wkey);
			}//if
			valSize = valSize - wsize;
			if(valSize<=0)
				break;
		}//while
		__syncthreads();
}//map2


#endif //__MAP_CU__