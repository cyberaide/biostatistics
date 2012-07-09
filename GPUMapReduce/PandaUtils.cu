/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: PandaUtils.cu 
	Time: 2012-07-01 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */

#include "Panda.h"



#ifndef __PANDAUTILS_CU__
#define __PANDAUTILS_CU__

void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);        
    }
}

__global__ void printData(d_global_state d_g_state ){
	//printf("-----------printData TID:%d\n",TID);
	
	if(TID>d_g_state.h_num_input_record)return;
	
	int begin=0;
	int end=0;
	for (int i=0;i<TID;i++){
		begin += (d_g_state.d_intermediate_keyval_arr_arr[i].arr_len);
	}//for
	end = begin + (d_g_state.d_intermediate_keyval_arr_arr[TID].arr_len);
	//printf("copyData:%d begin:%d, end:%d\n",TID,begin,end);
	
	for(int i=begin; i<end; i++){
		keyval_t * p1 = &(d_g_state.d_intermediate_keyval_arr[i]);
		printf("printData TID:%d keySize:%d key %s val:%d\n",TID,p1->keySize, p1->key, *(int*)p1->val);
	}//for
}//printData


__global__ void printData2(d_global_state d_g_state ){
	//printf("-----------printData TID:%d\n",TID);
	if(TID>d_g_state.h_num_input_record)return;
	keyval_t * p1 = &(d_g_state.d_input_keyval_arr[TID]);
	int len = p1->valSize -1;
	((char *)(p1->val))[len] = '\0';
	printf("printData TID:%d keySize:%d key %d val:%s\n",TID,p1->keySize, *(int*)(p1->key), p1->val);
}//printData

__global__ void printData3(d_global_state d_g_state ){
	//printf("-----------printData TID:%d\n",TID);
	if(TID>d_g_state.h_num_input_record)return;
	keyvals_t * p1 = &(d_g_state.d_sorted_keyvals_arr[TID]);
	//printf("printData3 TID:%d key:%s",TID, p1->key);
	for (int i=0;i<p1->val_arr_len;i++)
		printf("printData3 :TID:%d, i:%d  key:%s, val:%d\n",TID, i,p1->key, *(int*)p1->vals[i].val);
	//printf("\n");
	//printf("printData 3 TID:%d i:[%d] keySize:%d key %s val:%d\n",TID,i, p1->keySize, p1->key, *(int*)(p1->vals[i].val));
}//printData

__global__ void printData4(int index, int j, val_t *p){
	for (int i=0;i<j;i++){
		//printf("print4: i:%d, j:%d valSize:%d val:%d \n",index, j, (p[i]->valSize),*(int*)p[i]->val);
		printf("print4: index:%d, i:%d valSize:%d val:%d \n",index, i, p[i].valSize,*(int*)p[i].val);
	}//for
}


//--------------------------------------------------------
//start a timer
//
//param	: start_tv
//--------------------------------------------------------



void startTimer(TimeVal_t *start_tv)
{
   //gettimeofday((struct timeval*)start_tv, NULL);
}

//--------------------------------------------------------
//end a timer, and print out a message
//
//param	: msg message to print out
//param	: start_tv
//--------------------------------------------------------
void endTimer(char *msg, TimeVal_t *start_tv)
{
   /*cudaThreadSynchronize();
   struct timeval end_tv;

   gettimeofday(&end_tv, NULL);

   time_t sec = end_tv.tv_sec - start_tv->tv_sec;
   time_t ms = end_tv.tv_usec - start_tv->tv_usec;

   time_t diff = sec * 1000000 + ms;
	*/
   //printf("%10s:\t\t%fms\n", msg, (double)((double)diff/1000.0));
}//void


//----------------------------------------------------------
//print output records
//
//param: spec
//param: num -- maximum number of output records to print
//param: printFunc -- a function pointer
//	void printFunc(void* key, void* val, int keySize, int valSize)
//----------------------------------------------------------
void PrintOutputRecords(Spec_t* spec, int num, PrintFunc_t printFunc)
{
	/*
	int maxNum = num;
	if (maxNum > spec->outputRecordCount || maxNum < 0) maxNum = spec->outputRecordCount;
	for (int i = 0; i < maxNum; ++i)
	{
		int4 index = spec->outputOffsetSizes[i];
		printFunc((char*)spec->outputKeys + index.x, (char*)spec->outputVals + index.z, index.y, index.w);
	}
	*/
}//void

#endif //__PANDAUTILS_CU__
