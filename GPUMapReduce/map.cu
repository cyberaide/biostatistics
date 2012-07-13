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


__device__ void map2(void *KEY, void*VAL, int keySize, int valSize, d_global_state *d_g_state, int map_task_idx){
		//printf("map2 TID:%d, key:%d, val:%s\n",TID,*(int*)KEY,(char *)VAL);

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
				//if(TID>1500)
				//printf("\t\tmap2: TID:%d, index:%s\n",TID, wkey);
			}//if
			valSize = valSize - wsize;
			if(valSize<=0)
				break;
		}//while

}//map2


//__device__ void MAP_FUNC//(void *key, void val, size_t keySize, size_t valSize)
__device__ void map(void *key, void *val, int keySize, int valSize, 
		 int*	psKeySizes, int*psValSizes, int*psCounts, int2*	keyValOffsets, 
		 char*	interKeys, char*interVals, int4*interOffsetSizes, int*curIndex,
		 d_global_state d_g_state,int map_task_id)
{
		
	//__syncthreads();
}
#endif //__MAP_CU__