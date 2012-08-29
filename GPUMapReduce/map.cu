/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: map.cu 
	First Version:	2012-07-01 V0.1
	Current Version: V0.3	
	Last Updates:   2012-8-29

	Developer: Hui Li (lihui@indiana.edu)
	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */

#ifndef __MAP_CU__
#define __MAP_CU__

#include "Panda.h"
#include "Global.h"

__device__ float operator*(float4 a, float4 b)
{
	return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w);
}//__device__


void cpu_1d_blocked_matrix(float *A, float *B, float *C, int wA,int start_row_id_id,int end_id, int bz);
void cpu_2d_blocked_matrix(float *A, float *B, float *C, int wA,int row_id,int col_id, int bz);

void cpu_map(void *KEY, void*VAL, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx){

	MM_KEY_T* pKey = ((MM_KEY_T*)KEY);
	MM_VAL_T* pVal = ((MM_VAL_T*)VAL);

	int rowId = pVal->row;
	int colId = pVal->col;
	int bz = pVal->bz;

	int wA = pVal->col_dim;
	int wB = pVal->col_dim;

	float *A = pKey->h_matrix1;
	float *B = pKey->h_matrix2;
	float *C = pKey->h_matrix3;
	cpu_1d_blocked_matrix(A, B, C, wA,rowId,colId,bz);
	
}//map2

__device__ void gpu_map(void *KEY, void*VAL, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){

	MM_KEY_T* pKey = ((MM_KEY_T*)KEY);
	MM_VAL_T* pVal = ((MM_VAL_T*)VAL);

	int start_row_id_id = pVal->row;
	int end_id = pVal->col;

	int wA = pVal->col_dim;
	int wB = pVal->col_dim;
	int bz = pVal->bz; //size of each tile
	int m = wA;
	bz = BLOCK_SIZE;

	float Csub = 0.0;
	float *A = pKey->matrix1;
	float *B = pKey->matrix2;
	float *C = pKey->matrix3;

	float4*As = (float4*)A;
	float4*Bs = (float4*)B;

	int i,j,k;
	int start_row_idPointA = pVal->row*bz;
	int start_row_idPointB = pVal->col*bz;

	int aHeight = bz;
	int aHeightBlocks = aHeight/bz;
	int aLastBlockHeight = aHeight - (aHeightBlocks*bz);

	if (aLastBlockHeight>0){
		aHeightBlocks++;
	}//if

	int bWidth = bz;
	int bWidthBlocks = bWidth/bz;
	int bLastBlockWidth = bWidth - (bWidthBlocks*bz);
	if (bLastBlockWidth>0){
		bWidthBlocks++;
	}//if

	int commBlocks = m/bz;
	int commLastBlockWidth = m - (commBlocks*bz);
	if (commLastBlockWidth >0){
		commBlocks++;
	}//fi

	int aBlockHeight = bz;
	int bBlockWidth = bz;
	int commBlockWidth = bz;
	int ib,jb,kb;
	float4 b4,c4;
	float aik;

	for (ib=0; ib<aHeightBlocks; ib++){
		if (aLastBlockHeight>0 && ib==(aHeightBlocks-1)){
			aBlockHeight = aLastBlockHeight;
		}//if

		bBlockWidth = bz;
		for (jb=0; jb<bWidthBlocks;jb++){
			if (bLastBlockWidth>0&&jb==(bWidthBlocks-1))
				bBlockWidth = bLastBlockWidth;

			commBlockWidth = bz;
			for (kb =0;kb<commBlocks;kb++){
				if (commLastBlockWidth>0 && kb==(commBlocks-1))
					commBlockWidth = commLastBlockWidth;
				for (i = start_row_idPointA + ib*bz;i<start_row_idPointA+(ib*bz)+aBlockHeight;i++){
					for (k = kb*bz;k<(kb*bz)+(commBlockWidth);k++){
						aik = A[i*m+k];
						float4 *Bsub = (float4*)(B+k*m+jb*bz);
						float4 *Csub = (float4*)(C+i*m+jb*bz);
						//for (j= jb*bz;j<(jb*bz)+(bBlockWidth)/4;j++){
						for (j=0; j<(bBlockWidth/4); j++){
							b4 = *((Bsub)+j);
							c4 = *((Csub)+j);
							c4.x += aik*b4.x;
							c4.y += aik*b4.y;
							c4.z += aik*b4.z;
							c4.w += aik*b4.w;
							*((Csub)+j) = c4;
							//(C[i*m+j]+=A[i*m+k]*B[k*m+j];
						}//for
						for (int rj=0; rj<(bBlockWidth%4); rj++){
							int index = jb*bz+4*(bBlockWidth/4)+rj;
							C[i*m+index] += aik*(*(B+k*m+index));
						}
					}
				}//for
			}//for
		}//for
	}//for
	//check results	
	/*if (map_task_idx == 1){
		for (int j=10;j<20;j++)
		for (int i=0;i<5;i++){
			printf("%f ",C[j*wA+i]);
		}
		printf("\n");
	}*/
}


__device__ void gpu_map2(void *KEY, void*VAL, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){

	MM_KEY_T* pKey = ((MM_KEY_T*)KEY);
	MM_VAL_T* pVal = ((MM_VAL_T*)VAL);

	int start_row_id_id = pVal->row;
	int end_id = pVal->col;

	int wA = pVal->col_dim;
	int wB = pVal->col_dim;
	int bz = pVal->bz; //size of each tile
	int m = wA;

	float Csub = 0.0;
	//printf("map2 TID:%d, key:%d, val:%s \n",TID,*(int*)KEY,(char *)VAL);
	float *A = pKey->matrix1;
	float *B = pKey->matrix2;
	float *C = pKey->matrix3;

	int i,j,k;

	int start_row_idpoint = start_row_id_id;
	int endpoint = end_id;
	//DoLog("start_row_id_id:%d end_id:%d",start_row_id_id,end_id);

	int aHeight = endpoint - start_row_idpoint;
	int aHeightBlocks = aHeight/bz;
	int aLastBlockHeight = aHeight - (aHeightBlocks*bz);

	if (aLastBlockHeight>0){
		aHeightBlocks++;
	}//if
	int bWidthBlocks = m/bz;
	int bLastBlockWidth = m - (bWidthBlocks*bz);
	if (bLastBlockWidth>0){
		bWidthBlocks++;
	}//if

	int commBlocks = m/bz;
	int commLastBlockWidth = m - (commBlocks*bz);
	if (commLastBlockWidth >0){
		commBlocks++;
	}//fi

	int aBlockHeight = bz;
	int bBlockWidth = bz;
	int commBlockWidth = bz;
	int ib,jb,kb;

	for (ib=0;ib<aHeightBlocks;ib++){
		if (aLastBlockHeight>0 && ib==(aHeightBlocks-1)){
			aBlockHeight = aLastBlockHeight;
		}//if

		bBlockWidth = bz;
		for (jb=0; jb<bWidthBlocks;jb++){
			if (bLastBlockWidth>0&&jb==(bWidthBlocks-1))
				bBlockWidth = bLastBlockWidth;

			commBlockWidth = bz;
			for (kb =0;kb<commBlocks;kb++){
				if (commLastBlockWidth>0 && kb==(commBlocks-1))
					commBlockWidth = commLastBlockWidth;
				for (i = start_row_idpoint+ib*bz;i<start_row_idpoint+(ib*bz)+aBlockHeight;i++){
					for (k = kb*bz;k<(kb*bz)+commBlockWidth;k++){
						for (j= jb*bz;j<(jb*bz)+bBlockWidth;j++){
							C[i*m+j]+=A[i*m+k]*B[k*m+j];
						}//for
					}
				}//for
			}//for
		}//for
	}//for

}


__device__ void gpu_map1(void *KEY, void*VAL, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){

	MM_KEY_T* pKey = ((MM_KEY_T*)KEY);
	MM_VAL_T* pVal = ((MM_VAL_T*)VAL);

	int rowId = pVal->row;
	int colId = pVal->col;

	int wA = pVal->col_dim;
	int wB = pVal->col_dim;
	//int BLOCK_SIZE = pVal->bz;

	// Index of the first sub-matrix of A processed by the block
	float Csub = 0.0;
	//float4 *A = (float4*)(pKey->matrix1);
	//float4 *B = (float4*)(pKey->matrix2);
	float *C = pKey->matrix3;
	float *A = (float*)(pKey->matrix1);
	float *B = (float*)(pKey->matrix2);
	
	//float4* As;
	//float4* Bs;

	/*if (map_task_idx == 1){
		printf("in gpu_map\n");
		for (int i=0;i<10;i++){
			printf("%f :%f",pKey->matrix2[i], pKey->matrix3[i]);
		}
		printf("\n");
	}*/

	int aBase = (rowId)*wA*BLOCK_SIZE;
	int bBase = (colId)*wB*BLOCK_SIZE;
	// Index of the last sub-matrix of A processed by the block
	int aBegin = aBase;
	int bBegin = bBase;
	int x, y;
	for (int step = 0; step < wA/BLOCK_SIZE; step++){
		for (int n=0;n<BLOCK_SIZE;n++){
			int aEnd = aBegin + BLOCK_SIZE;

			for (int step2 = 0; step2< wB/BLOCK_SIZE; step2++){
				for (int n2=0;n2<BLOCK_SIZE;n2++){

					//int aBegin = aBase + n*wA + step*BLOCK_SIZE;	
					int bEnd = bBegin + BLOCK_SIZE;
					//int x = (rowId-1)*BLOCK_SIZE + n;
					//int y = (colId-1)*BLOCK_SIZE + n;

					//As = (float4*)(A+aBegin);
					//Bs = (float4*)(B+bBegin);

					for (int i=0;i<(aEnd-aBegin);i++)
						for (int j=0;j<(bEnd-bBegin);j++){
							
							/*
							Csub += As[i].x*Bs[j].x;		
							Csub += As[i].y*Bs[j].y;		
							Csub += As[i].z*Bs[j].z;		
							Csub += As[i].w*Bs[j].w;		
							*/
							Csub += A[i]*B[j];		
							//Csub += 4;

						}//for

						/*
					int aRe = (aEnd-aBegin) & 0x00000003;
					int bRe = (bEnd-bBegin) & 0x00000003;

					float *rMatrix1 = (float*)(As + (aEnd-aBegin)/4);
					float *rMatrix2 = (float*)(Bs + (bEnd-bBegin)/4);
					//float f1,f2;
					for (int i = 0; i < aRe; i++)
						for (int j = 0; j < bRe; j++)
						{
							//f1 = rMatrix1[i];
							//f2 = rMatrix2[j];
							//Csub += (rMatrix1[i] * rMatrix2[j]);
							Csub += 1.0;
						}//for
						*/
							/*
							for (int i=aBegin;i<aEnd;i++)
							for (int j=bBegin;j<bEnd;j++){
							Csub += As[i]*Bs[j];		
							}//for
							*/
					 x = (rowId)*BLOCK_SIZE + n;
					 y = (colId)*BLOCK_SIZE + n2;
					if (x>=wA) x = wA -1;
					if (y>=wA) y = wA -1;
					//C[x+y] = Csub;
					bBegin = bBase + wB;
				}
				bBegin = bBase + BLOCK_SIZE;
			}
			aBegin = aBase + wA;
		}//int
		aBegin = aBase + BLOCK_SIZE;
	}//for
	C[x+y] = Csub;
	printf("Csub:%f\n",Csub);
	/*
	if (map_task_idx == 1){
		for (int i=0;i<3;i++){
			printf("%f ",C[i]);
		}
		printf("\n");
	}*/
}


__device__ void gpu_map3(void *KEY, void*VAL, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){

	MM_KEY_T* pKey = ((MM_KEY_T*)KEY);
	MM_VAL_T* pVal = ((MM_VAL_T*)VAL);

	int rowId = pVal->row;
	int colId = pVal->col;
	//int BLOCK_SIZE = pVal->bz;
	int M_COL_COUNT = pVal->col_dim;

	/*
	float4 *matrix1 = (float4*)(pKey->matrix1+rowId*M_COL_COUNT);
	float4 *matrix2 = (float4*)(pKey->matrix2+colId*M_COL_COUNT);

	float newVal = 0.0f;
	int col4 = M_COL_COUNT >> 2;
	int remainder = M_COL_COUNT & 0x00000003;
	*/

	int bx = blockIdx.x;
	int by = blockIdx.y;
	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int wA = pVal->col_dim;
	int wB = pVal->col_dim;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;
	// Index of the last sub-matrix of A processed by the block
	int aEnd   = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep  = BLOCK_SIZE;
	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;
	// Step size used to iterate through the sub-matrices of B
	int bStep  = BLOCK_SIZE * wB;
	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;
	//printf("map2 TID:%d, key:%d, val:%s \n",TID,*(int*)KEY,(char *)VAL);
	float *A = pKey->matrix1;
	float *B = pKey->matrix2;

	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep) {

			//printf("a:%d, b:%d ty:%d tx:%d\n",a,b,ty,tx);

			// Declaration of the shared memory array As used to
			// store the sub-matrix of A
			/*
			__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

			// Declaration of the shared memory array Bs used to
			// store the sub-matrix of B
			__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

			// Load the matrices from device memory
			// to shared memory; each thread loads
			// one element of each matrix
			int index = a + wA * ty + tx;
			//int bound = wA*wA;
			//if (index >= bound) index = bound -1;
			//index = 0;

			for (int i=0;i<BLOCK_SIZE;i++){
			for (int j=0;j<BLOCK_SIZE;j++){
			//AS(ty, tx) = A[a + wA * ty + tx];
			AS(i, j) = A[a + wA * i + j];
			BS(i, j) = B[b + wB * i + j];
			}//for
			}//for
			*/

			//AS(ty, tx) = A[index];
			//index = b + wB * ty + tx;
			//if (index >= bound) index = bound -1;
			//BS(ty, tx) = B[b + wB * ty + tx];
			//index = 0;
			//BS(ty, tx) = B[index];
			// Synchronize to make sure the matrices are loaded
			__syncthreads();

			// Multiply the two matrices together;
			// each thread computes one element
			// of the block sub-matrix
			for (int i=0;i<BLOCK_SIZE;i++)
				for(int j=0;j<BLOCK_SIZE;j++){
#pragma unroll
					for (int k = 0; k < BLOCK_SIZE; ++k)
						//Csub += AS(ty, k) * AS(k, tx);
						//			Csub += AS(i, k) * AS(k, j);
						int kk;
				}
				//Csub += AS(ty, k) * BS(k, tx);
				// Synchronize to make sure that the preceding
				// computation is done before loading two new
				// sub-matrices of A and B in the next iteration
				__syncthreads();

	}


	// Write the block sub-matrix to device memory;
	// each thread writes one element

	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	//C[c + wB * ty + tx] = Csub;

	/*for (int i = 0; i < col4; i++)
	{
	float4 v1 = matrix1[i];
	float4 v2 = matrix2[i];

	newVal += v1.x * v2.x;
	newVal += v1.y * v2.y;
	newVal += v1.z * v2.z;
	newVal += v1.w * v2.w;
	}
	float *rMatrix1 = (float*)(matrix1+col4);
	float *rMatrix2 = (float*)(matrix2+col4);
	for (int i = 0; i < remainder; i++)
	{
	float f1 = rMatrix1[i];
	float f2 = rMatrix2[i];
	newVal += (f1 * f2);
	}
	__syncthreads(); */

}//map2


#endif //__MAP_CU__
