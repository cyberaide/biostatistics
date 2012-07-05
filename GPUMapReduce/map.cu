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

__device__ void MAP_COUNT_FUNC//(void *key, void *val, size_t keySize, size_t valSize)
{
	WC_KEY_T* pKey = (WC_KEY_T*)key;
	WC_VAL_T* pVal = (WC_VAL_T*)val;

	char* ptrBuf = pKey->file + pVal->line_offset;
	int line_size = pVal->line_size;

	char* p = ptrBuf;
	int lsize = 0;
	int wsize = 0;
	char* start = ptrBuf;

	while(1)
	{
		for (; *p >= 'A' && *p <= 'Z'; p++, lsize++);
		*p = '\0';
		++p;
		++lsize;
		wsize = (int)(p - start);
		if (wsize > 6)
		{
			//printf("%s, wsize:%d\n", start, wsize);	
			//EMIT_INTER_COUNT_FUNC(wsize, sizeof(int));
			EMIT_INTER_COUNT_FUNC(sizeof(char), sizeof(int));
		}//if
		for (; (lsize < line_size) && (*p < 'A' || *p > 'Z'); p++, lsize++);
		if (lsize >= line_size) break;
		start = p;
	}
}

__device__ void map2(d_global_state *d_g_state,int map_task_id){
		
		if (map_task_id>=d_g_state->h_num_input_record)	return;
		char *p = (char *)(d_g_state->d_input_keyval_arr[map_task_id].val);
		int len = d_g_state->d_input_keyval_arr[map_task_id].valSize;
		int wsize = 0;
		char *start;
		//printf("map2 TID:%d, index:%d val:%s\n",TID, index,p);
		while(1)
		{
			start = p;
			for(;*p>='A' && *p<='Z';p++);
			*p='\0';
			++p;
			wsize=(int)(p-start);
			if (wsize>6){
				char *key = (char *)malloc(wsize);
				memcpy(key,start,wsize);
				int wc = 1;
				EmitIntermediate2(key, &wc, wsize, sizeof(int), d_g_state, map_task_id);
				//printf("\t\tTID:%d, index:%d: %s\n",TID, index, start);
			}//if
			len = len-wsize;
			if(len<=0)
				break;
		}//while
}//map2


//__device__ void MAP_FUNC//(void *key, void val, size_t keySize, size_t valSize)
__device__ void map(void *key, void *val, int keySize, int valSize, 
		 int*	psKeySizes, int*psValSizes, int*psCounts, int2*	keyValOffsets, 
		 char*	interKeys, char*interVals, int4*interOffsetSizes, int*curIndex,
		 d_global_state d_g_state,int map_task_id)
{
	
	WC_KEY_T* pKey = (WC_KEY_T*)key;
	WC_VAL_T* pVal = (WC_VAL_T*)val;

	char* filebuf = pKey->file;
	char* ptrBuf = filebuf + pVal->line_offset;
	int line_size = pVal->line_size;

	char *pWord = ptrBuf;
	*(pWord+line_size)='\0';

	printf("Map->EmitIntermediate: map_task_id:%d Key:%s\n",map_task_id,pWord);

	char* p = ptrBuf;
	char* start = ptrBuf;
	int lsize = 0;
	int wsize = 0;
	
	while(1)
	{
		for (; *p >= 'A' && *p <= 'Z'; p++, lsize++);
		*p = '\0';
		++p;
		++lsize;
		wsize = (int)(p - start);
		int* o_val = (int*)GET_OUTPUT_BUF(0);
		*o_val = wsize;
		if (wsize > 6) 
		{
		//printf("%s, %d\n", start, wsize);
		int count = 1;
		//printf("Map->EmitIntermediate -> while: map_task_id:%d key:%s\n",map_task_id,start);
		EmitIntermediate((char*)start,(char*)&count,wsize,sizeof(int),psKeySizes,psValSizes,
			                         psCounts,keyValOffsets,interKeys,interVals,interOffsetSizes,curIndex, (d_g_state), map_task_id);
		//EMIT_INTERMEDIATE_FUNC(start, o_val, wsize, sizeof(int));
		}//if
		for (; (lsize < line_size) && (*p < 'A' || *p > 'Z'); p++, lsize++);
		if (lsize >= line_size) break;
		start = p;	
	}//while
	//__syncthreads();
}
#endif //__MAP_CU__