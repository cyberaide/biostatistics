/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: reduce.cu
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2012-09-02

	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */

#ifndef __REDUCE_CU__
#define __REDUCE_CU__

#include "Panda.h"

//-------------------------------------------------------------------------
//Reduce Function in this application
//-------------------------------------------------------------------------


__device__ void gpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, gpu_context d_g_state){

		int count = 0;
		for (int i=0;i<valCount;i++){
			count += *(int *)(VAL[i].val);
		}//
		
		GPUEmitReduceOuput(KEY,&count,keySize,sizeof(int),&d_g_state);
		
}//reduce2

void cpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, cpu_context* d_g_state){

		int count = 0;
		for (int i=0;i<valCount;i++){
			count += *(int *)(VAL[i].val);
		}//
		
		CPUEmitReduceOuput(KEY,&count,keySize,sizeof(int),d_g_state);
		//printf("reduce:%s :%d  \n",(char *)KEY, count);
		
}//reduce2


#endif //__REDUCE_CU__
