
/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: PandaUtils.cu 
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2012-09-02

	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */


#include "Panda.h"
#include "Global.h"


#ifdef _WIN32 
#include <windows.h> 
#include <time.h>
#elif MACOS 
#include <sys/param.h> 
#include <sys/sysctl.h> 
#elif __linux
#include <unistd.h> 
#include <sys/time.h>
#endif 


int getGPUCoresNum() { 
	//assert(tid<total);
	int arch_cores_sm[3] = {1, 8, 32 };
	cudaDeviceProp gpu_dev;
	int tid = 0;
	cudaGetDeviceProperties(&gpu_dev, tid);

	int sm_per_multiproc = 1;
	if (gpu_dev.major == 9999 && gpu_dev.minor == 9999)
			sm_per_multiproc = 1;
	else if (gpu_dev.major <=2)
			sm_per_multiproc = arch_cores_sm[gpu_dev.major];
	else
			sm_per_multiproc = arch_cores_sm[2];

	return ((gpu_dev.multiProcessorCount)*(sm_per_multiproc));
	//DoLog("Configure Device ID:%d: Device Name:%s MultProcessorCount:%d sm_per_multiproc:%d", i, gpu_dev.name,gpu_dev.multiProcessorCount,sm_per_multiproc);

}

int getCPUCoresNum() { 

#ifdef WIN32 
    SYSTEM_INFO sysinfo; 
    GetSystemInfo(&sysinfo); 
    return sysinfo.dwNumberOfProcessors; 
#elif MACOS 
    int nm[2]; 
    size_t len = 4; 
    uint32_t count; 
 
    nm[0] = CTL_HW; nm[1] = HW_AVAILCPU; 
    sysctl(nm, 2, &count, &len, NULL, 0); 
 
    if(count < 1) { 
        nm[1] = HW_NCPU; 
        sysctl(nm, 2, &count, &len, NULL, 0); 
        if(count < 1) { count = 1; } 
    } 
    return count; 
#elif __linux
    return sysconf(_SC_NPROCESSORS_ONLN); 
#endif 

}




#ifndef __PANDAUTILS_CU__
#define __PANDAUTILS_CU__

void DoDiskLog(const char *str){
	FILE *fptr;
	char file_name[128];
	sprintf(file_name,"%s","panda.log");
	fptr = fopen(file_name,"a");
	fprintf(fptr,"[PandaDiskLog]\t\t:%s\n",str);
	//fprintf(fptr,"%s",__VA_ARGS__);
	fclose(fptr);
	//printf("\n");
}//void

double PandaTimer(){

	#ifndef _WIN32
	static struct timeval tv;
	gettimeofday(&tv,NULL);
	double curTime = tv.tv_sec + tv.tv_usec/1000000.0;

	//DoLog("\t Panda CurTime:%f", curTime);
	return curTime;
	#else
	//newtime = localtime( &long_time2 ); 
	double curTime = GetTickCount(); 
	//DoLog("\t Panda CurTime:%f", curTime);
	curTime /=1000.0;
	return curTime;
	#endif

}

void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "[PandaError][%s][%i]: CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit((int)err);        
    }
}

__global__ void printData(gpu_context d_g_state ){
	//printf("-----------printData TID:%d\n",TID);
	int num_records_per_thread = (d_g_state.num_input_record+(gridDim.x*blockDim.x)-1)/(gridDim.x*blockDim.x);
	int block_start_row_idx = num_records_per_thread*blockIdx.x*blockDim.x;
	int thread_start_row_idx = block_start_row_idx 
		+ (threadIdx.x/STRIDE)*num_records_per_thread*STRIDE
		+ (threadIdx.x%STRIDE);
	int thread_end_idx = thread_start_row_idx+num_records_per_thread*STRIDE;

	if(thread_end_idx>d_g_state.num_input_record)
		thread_end_idx = d_g_state.num_input_record;

	int begin, end, val_pos, key_pos;
	char *val_p,*key_p;

	for(int map_task_idx=thread_start_row_idx; map_task_idx < thread_end_idx; map_task_idx+=STRIDE){
	
		begin=0;
		end=0;
		for (int i=0;i<map_task_idx;i++){
		//	begin += (d_g_state.d_intermediate_keyval_arr_arr[i].arr_len);
		}//for
		//end = begin + (d_g_state.d_intermediate_keyval_arr_arr[map_task_idx].arr_len);
		//printf("copyData:%d begin:%d, end:%d\n",TID,begin,end);
	
		for(int i=begin;i<end;i++){
			//keyval_t * p1 = &(d_g_state.d_intermediate_keyval_arr[i]);
			val_pos = d_g_state.d_intermediate_keyval_pos_arr[i].valPos;
			key_pos = d_g_state.d_intermediate_keyval_pos_arr[i].keyPos;
			val_p = (char*)(d_g_state.d_intermediate_vals_shared_buff)+val_pos;
			key_p = (char*)(d_g_state.d_intermediate_keys_shared_buff)+key_pos;
			
			//keyval_t * p2 = &(d_g_state.d_intermediate_keyval_arr_arr[map_task_idx].arr[i-begin]);
			//memcpy(key_p,p2->key,p2->keySize);
			//memcpy(val_p,p2->val,p2->valSize);
			printf("printData: TID:%d key: %s  val:%d\n",TID,key_p,*(int *)val_p);
		}//for
		//if (index*recordsPerTask >= recordNum) return;
	}//for

}//printData

#ifdef DEV_MODE
__global__ void printData2(gpu_context d_g_state ){
	//printf("-----------printData TID:%d\n",TID);
	//if(TID>=d_g_state.num_input_record)return;
	//printf("printData2------------------------------%d\n",d_g_state.d_intermediate_keyval_arr_arr[TID].arr_len);

	if (TID==0){
	int keyPos = (d_g_state.d_input_keyval_pos_arr[0]).keyPos;
	int valPos = (d_g_state.d_input_keyval_pos_arr[0]).valPos;
	char *keyBuf = (char *)(d_g_state.d_input_keys_shared_buff)+keyPos;
	MM_KEY_T *key = (MM_KEY_T*)keyBuf;

	printf("Key2 =====================:%d\n",key->test);
	for (int i=0;i<10;i++)
		printf("%f ",key->matrix1[i]);
	printf("\n");

	for (int i=0;i<10;i++)
		printf("%f ",key->matrix2[i]);
	printf("\n");

	for (int i=0;i<10;i++)
		printf("%f ",key->matrix3[i]);
	printf("\n");

	}
	
	//keyval_t * p1 = &(d_g_state.d_input_keyval_arr[TID]);
	//int len = p1->valSize -1;
	//((char *)(p1->val))[len] = '\0';
	//printf("printData TID:%d keySize:%d key %d val:%s\n",TID,p1->keySize, *(int*)(p1->key), p1->val);
}//printData
#endif


__global__ void printData3(float *C ){

	//if(TID==1){
	printf("TID ==1  printC \n");
	
	
	for (int i=0;i<10;i++){
		printf("%f ",C[i]);
	}
	printf("\n");

	//}
	//printf("printData3 TID:%d key:%s",TID, p1->key);
	//for (int i=0;i<p1->val_arr_len;i++)
	//	printf("printData3 :TID:%d, i:%d  key:%s, val:%d\n",TID, i,p1->key, *(int*)p1->vals[i].val);
	//printf("\n");
	//printf("printData 3 TID:%d i:[%d] keySize:%d key %s val:%d\n",TID,i, p1->keySize, p1->key, *(int*)(p1->vals[i].val));
	
}//printData



//--------------------------------------------------------
//start_row_id a timer
//
//param	: start_row_id_tv
//--------------------------------------------------------

/*
void start_row_idTimer(TimeVal_t *start_row_id_tv)
{
   //gettimeofday((struct timeval*)start_row_id_tv, NULL);
}
*/

//--------------------------------------------------------
//end a timer, and print out a message
//
//param	: msg message to print out
//param	: start_row_id_tv
//--------------------------------------------------------
/*
void endTimer(char *msg, TimeVal_t *start_row_id_tv)
{
   cudaThreadSynchronize();
   struct timeval end_tv;

   gettimeofday(&end_tv, NULL);

   time_t sec = end_tv.tv_sec - start_row_id_tv->tv_sec;
   time_t ms = end_tv.tv_usec - start_row_id_tv->tv_usec;

   time_t diff = sec * 1000000 + ms;

   //printf("%10s:\t\t%fms\n", msg, (double)((double)diff/1000.0));
}//void
*/

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