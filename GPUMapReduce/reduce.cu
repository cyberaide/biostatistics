/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: reduce.cu 
	Time: 2012-07-01 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */

#ifndef __REDUCE_CU__
#define __REDUCE_CU__

#include "Panda.h"

//-------------------------------------------------------------------------
//No Reduce in this application
//-------------------------------------------------------------------------
__device__ void REDUCE_COUNT_FUNC//(void* key, void* vals, size_t keySize, size_t valCount)
{
}

__device__ void REDUCE_FUNC//(void* key, void* vals, size_t keySize, size_t valCount)
{
}

__device__ void reduce2(d_global_state d_g_state,int map_task_id){
		
		printf("reduce2   map_task_id:%d  arr_len:%d\n",map_task_id,d_g_state.d_sorted_keyvals_arr_len);
		if (map_task_id>=d_g_state.d_sorted_keyvals_arr_len)	return;

		keyvals_t *p = &(d_g_state.d_sorted_keyvals_arr[map_task_id]);
		int len = p->val_arr_len;
		int count = 0;
		for (int i=0;i<len;i++){
			count += *(int *)(p->vals[i].val);
			printf("word: %s val:%d  val_arr_len:%d\n",p->key, *(int *)(p->vals[i].val),len);					
		}	

}//map2

#endif //__REDUCE_CU__
