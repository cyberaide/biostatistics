/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: Global.h 
	Time: 2012-07-01 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 */	

#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#define BLOCK_SIZE 50
#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif



extern "C"
void cpu_map(void *KEY, void*VAL, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx);
		

extern "C"
__device__ int compare(const void *d_a, int len_a, const void *d_b, int len_b);

extern "C"
__device__ void map2(void *KEY, void*VAL, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx);


//extern "C" void cpu_matrix(float *A, float *B, float *C, int wide, int start_row_id, int end);

typedef struct
{

        float* matrix1;
        float* matrix2;
		float* matrix3;

		float* h_matrix1;
		float* h_matrix2;
		float* h_matrix3;

		int test;

} MM_KEY_T;

typedef struct
{		
        int row;
        int col;
		
		int bz;
		int ty;
		
        int row_dim;
        int col_dim;
} MM_VAL_T;

#endif
