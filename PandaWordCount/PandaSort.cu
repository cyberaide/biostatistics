
/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: PandaSort.cu 
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2012-09-02

	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//includes CUDA
#include <cuda_runtime.h>

#ifndef _PANDASORT_CU_
#define _PANDASORT_CU_

#include "Panda.h"
#include "compare.cu"

#define NUM_BLOCK_PER_CHUNK_BITONIC_SORT 512//b256
#define SHARED_MEM_INT2 256					
#define NUM_BLOCKS_CHUNK 256				//(512)
#define	NUM_THREADS_CHUNK 256				//(256)
#define CHUNK_SIZE (NUM_BLOCKS_CHUNK*NUM_THREADS_CHUNK)
#define NUM_CHUNKS_R (NUM_RECORDS_R/CHUNK_SIZE)


__device__ int getCompareValue(void *d_rawData, cmp_type_t value1, cmp_type_t value2)
{
	int compareValue=0;
	int v1=value1.x;
	int v2=value2.x;
	if((v1==-1) || (v2==-1))
	{
		if(v1==v2)
			compareValue=0;
					else
			if(v1==-1)
				compareValue=-1;
			else
				compareValue=1;
	}//if
	else
		compareValue=compare((void*)(((char*)d_rawData)+v1), value1.y, (void*)(((char*)d_rawData)+v2), value2.y); 
	return compareValue;
}//__device__

void * s_qsRawData=NULL;

__global__ void
partBitonicSortKernel( void* d_rawData, int totalLenInBytes,cmp_type_t* d_R, unsigned int numRecords, int chunkIdx, int unitSize)
{
	__shared__ cmp_type_t shared[NUM_THREADS_CHUNK];

	int tx = threadIdx.x;
	int bx = blockIdx.x;

	//load the data
	int dataIdx = chunkIdx*CHUNK_SIZE+bx*blockDim.x+tx;
	int unitIdx = ((NUM_BLOCKS_CHUNK*chunkIdx + bx)/unitSize)&1;
	shared[tx] = d_R[dataIdx];
	__syncthreads();
	int ixj=0;
	int a=0;
	cmp_type_t temp1;
	cmp_type_t temp2;
	int k = NUM_THREADS_CHUNK;

	if(unitIdx == 0)
	{
		for (int j = (k>>1); j>0; j =(j>>1))
		{
			ixj = tx ^ j;
			//a = (shared[tx].y - shared[ixj].y);				
			temp1=shared[tx];
			temp2= shared[ixj];
			if (ixj > tx) {
				//a=temp1.y-temp2.y;
				//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x)); 
				a=getCompareValue(d_rawData, temp1, temp2);
				if ((tx & k) == 0) {
					if ( (a>0)) {
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
				else {
					if ( (a<0)) {
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
			}
				
			__syncthreads();
		}
	}
	else
	{
		for (int j = (k>>1); j>0; j =(j>>1))
		{
			ixj = tx ^ j;
			temp1=shared[tx];
			temp2= shared[ixj];
			
			if (ixj > tx) {					
				//a=temp1.y-temp2.y;					
				//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x));
				a=getCompareValue(d_rawData, temp1, temp2);
				if ((tx & k) == 0) {
					if( (a<0))
					{
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
				else {
					if( (a>0))
					{
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
			}
			__syncthreads();
		}
	}

	d_R[dataIdx] = shared[tx];
}

__global__ void
unitBitonicSortKernel(void* d_rawData, int totalLenInBytes, cmp_type_t* d_R, unsigned int numRecords, int chunkIdx )
{
	__shared__ cmp_type_t shared[NUM_THREADS_CHUNK];

	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int unitIdx = (NUM_BLOCKS_CHUNK*chunkIdx + bx)&1;

	//load the data
	int dataIdx = chunkIdx*CHUNK_SIZE+bx*blockDim.x+tx;
	shared[tx] = d_R[dataIdx];
	__syncthreads();

	cmp_type_t temp1;
	cmp_type_t temp2;
	int ixj=0;
	int a=0;
	if(unitIdx == 0)
	{
		for (int k = 2; k <= NUM_THREADS_CHUNK; (k =k<<1))
		{
			// bitonic merge:
			for (int j = (k>>1); j>0; (j=j>>1))
			{
				ixj = tx ^ j;	
				temp1=shared[tx];
				temp2= shared[ixj];
				if (ixj > tx) {					
					//a=temp1.y-temp2.y;
					//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x));
					a=getCompareValue(d_rawData, temp1, temp2);
					if ((tx & k) == 0) {
						if ( (a>0)) {
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
					else {
						if ( (a<0)) {
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
				}
				
				__syncthreads();
			}
		}
	}
	else
	{
		for (int k = 2; k <= NUM_THREADS_CHUNK; (k =k<<1))
		{
			// bitonic merge:
			for (int j = (k>>1); j>0; (j=j>>1))
			{
				ixj = tx ^ j;
				temp1=shared[tx];
				temp2= shared[ixj];
				if (ixj > tx) {					
					//a=temp1.y-temp2.y;
					//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x));
					a=getCompareValue(d_rawData, temp1, temp2);
					if ((tx & k) == 0) {
						if( (a<0))
						{
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
					else {
						if( (a>0))
						{
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
				}
				
				__syncthreads();
			}
		}

	}

	d_R[dataIdx] = shared[tx];
}

__global__ void
bitonicKernel( void* d_rawData, int totalLenInBytes, cmp_type_t* d_R, unsigned int numRecords, int k, int j)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tid = threadIdx.x;
	int dataIdx = by*gridDim.x*blockDim.x + bx*blockDim.x + tid;

	int ixj = dataIdx^j;

	if( ixj > dataIdx )
	{
		cmp_type_t tmpR = d_R[dataIdx];
		cmp_type_t tmpIxj = d_R[ixj];
		if( (dataIdx&k) == 0 )
		{
			//if( tmpR.y > tmpIxj.y )
			//if(compareString((void*)(((char4*)d_rawData)+tmpR.x),(void*)(((char4*)d_rawData)+tmpIxj.x))==1) 
			if(getCompareValue(d_rawData, tmpR, tmpIxj)==1)
			{
				d_R[dataIdx] = tmpIxj;
				d_R[ixj] = tmpR;
			}
		}
		else
		{
			//if( tmpR.y < tmpIxj.y )
			//if(compareString((void*)(((char4*)d_rawData)+tmpR.x),(void*)(((char4*)d_rawData)+tmpIxj.x))==-1) 
			if(getCompareValue(d_rawData, tmpR, tmpIxj)==-1)
			{
				d_R[dataIdx] = tmpIxj;
				d_R[ixj] = tmpR;
			}
		}
	}
}

__device__ inline void swap(cmp_type_t & a, cmp_type_t & b)
{
	// Alternative swap doesn't use a temporary register:
	// a ^= b;
	// b ^= a;
	// a ^= b;
	
    cmp_type_t tmp = a;
    a = b;
    b = tmp;
}

__global__ void bitonicSortSingleBlock_kernel(void* d_rawData, int totalLenInBytes, cmp_type_t * d_values, int rLen, cmp_type_t* d_output)
{
	__shared__ cmp_type_t bs_cmpbuf[SHARED_MEM_INT2];
	

    //const int by = blockIdx.y;
	//const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	//const int bid=bx+by*gridDim.x;
	//const int numThread=blockDim.x;
	//const int resultID=(bx)*numThread+tid;
	
	if(tid<rLen)
	{
		bs_cmpbuf[tid] = d_values[tid];
	}
	else
	{
		bs_cmpbuf[tid].x =-1;
	}

    __syncthreads();

    // Parallel bitonic sort.
	int compareValue=0;
    for (int k = 2; k <= SHARED_MEM_INT2; k *= 2)
    {
        // Bitonic merge:
        for (int j = k / 2; j>0; j /= 2)
        {
            int ixj = tid ^ j;
            
            if (ixj > tid)
            {
                if ((tid & k) == 0)
                {
					compareValue=getCompareValue(d_rawData, bs_cmpbuf[tid], bs_cmpbuf[ixj]);
					//if (shared[tid] > shared[ixj])
					if(compareValue>0)
                    {
                        swap(bs_cmpbuf[tid], bs_cmpbuf[ixj]);
                    }
                }
                else
                {
					compareValue=getCompareValue(d_rawData, bs_cmpbuf[tid], bs_cmpbuf[ixj]);
                    //if (shared[tid] < shared[ixj])
					if(compareValue<0)
                    {
                        swap(bs_cmpbuf[tid], bs_cmpbuf[ixj]);
                    }
                }
            }
            
            __syncthreads();
        }
    }

    // Write result.
	/*if(tid<rLen)
	{
		d_output[tid] = bs_cmpbuf[tid+SHARED_MEM_INT2-rLen];
	}*/
	int start_row_idCopy=SHARED_MEM_INT2-rLen;
	if(tid>=start_row_idCopy)
	{
		d_output[tid-start_row_idCopy]=bs_cmpbuf[tid];
	}
}

__global__ void bitonicSortMultipleBlocks_kernel(void* d_rawData, int totalLenInBytes, cmp_type_t * d_values, int* d_bound, int start_row_idBlock, int numBlock, cmp_type_t *d_output)
{
	__shared__ int bs_pStart;
	__shared__ int bs_pEnd;
	__shared__ int bs_numElement;
    __shared__ cmp_type_t bs_shared[SHARED_MEM_INT2];
	

    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	//const int numThread=blockDim.x;
	//const int resultID=(bx)*numThread+tid;
	if(bid>=numBlock) return;

	if(tid==0)
	{
		bs_pStart=d_bound[(bid+start_row_idBlock)<<1];
		bs_pEnd=d_bound[((bid+start_row_idBlock)<<1)+1];
		bs_numElement=bs_pEnd-bs_pStart;
			
	}
	__syncthreads();
    // Copy input to shared mem.
	if(tid<bs_numElement)
	{
		bs_shared[tid] = d_values[tid+bs_pStart];
		
	}
	else
	{
		bs_shared[tid].x =-1;
	}

    __syncthreads();

    // Parallel bitonic sort.
	int compareValue=0;
    for (int k = 2; k <= SHARED_MEM_INT2; k *= 2)
    {
        // Bitonic merge:
        for (int j = k / 2; j>0; j /= 2)
        {
            int ixj = tid ^ j;
            
            if (ixj > tid)
            {
                if ((tid & k) == 0)
                {
					compareValue=getCompareValue(d_rawData, bs_shared[tid], bs_shared[ixj]);
					//if (shared[tid] > shared[ixj])
					if(compareValue>0)
                    {
                        swap(bs_shared[tid], bs_shared[ixj]);
                    }
                }
                else
                {
					compareValue=getCompareValue(d_rawData, bs_shared[tid], bs_shared[ixj]);
                    //if (shared[tid] < shared[ixj])
					if(compareValue<0)
                    {
                        swap(bs_shared[tid], bs_shared[ixj]);
                    }
                }
            }
            
            __syncthreads();
        }
    }

    // Write result.
	//if(tid<bs_numElement)
	//{
	//	d_output[tid+bs_pStart] = bs_shared[tid+SHARED_MEM_INT2-bs_numElement];
	//}
	//int start_row_idCopy=SHARED_MEM_INT2-bs_numElement;
	if(tid>=bs_numElement)
	{
		d_output[tid-bs_numElement]=bs_shared[tid];
	}
}


__global__ void initialize_kernel(cmp_type_t* d_data, int start_row_idPos, int rLen, cmp_type_t value)
{

}

void bitonicSortMultipleBlocks(void* d_rawData, int totalLenInBytes, cmp_type_t * d_values, int* d_bound, int numBlock, cmp_type_t * d_output)
{
	
}


void bitonicSortSingleBlock(void* d_rawData, int totalLenInBytes, cmp_type_t * d_values, int rLen, cmp_type_t * d_output)
{
	int numThreadsPerBlock_x=SHARED_MEM_INT2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=1;
	int numBlock_y=1;
	

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	bitonicSortSingleBlock_kernel<<<grid,thread>>>(d_rawData, totalLenInBytes, d_values, rLen, d_output);
	cudaThreadSynchronize();
}


void initialize(cmp_type_t *d_data, int rLen, cmp_type_t value)
{
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start_row_id=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start_row_id=i*chunkSize;
		end=start_row_id+chunkSize;
		if(end>rLen)
			end=rLen;
		initialize_kernel<<<grid,thread>>>(d_data, start_row_id, rLen, value);
	} 
	cudaThreadSynchronize();
}
void bitonicSortGPU(void* d_rawData, int totalLenInBytes, cmp_type_t* d_Rin, int rLen, void *d_Rout)
{
	unsigned int numRecordsR;

	unsigned int size = rLen;
	unsigned int level = 0;
	while( size != 1 )
	{
		size = size/2;
		level++;
	}
	
	if( (1<<level) < rLen )
	{
		level++;
	}

	numRecordsR = (1<<level);
	if(rLen<=NUM_THREADS_CHUNK)
	{
		bitonicSortSingleBlock((void*)d_rawData, totalLenInBytes, d_Rin, rLen, (cmp_type_t*)d_Rout);
	}
	else
	if( rLen <= 256*1024 )
	{
		//unsigned int numRecordsR = rLen;
		
		unsigned int numThreadsSort = NUM_THREADS_CHUNK;
		if(numRecordsR<NUM_THREADS_CHUNK)
			numRecordsR=NUM_THREADS_CHUNK;
		unsigned int numBlocksXSort = numRecordsR/numThreadsSort;
		unsigned int numBlocksYSort = 1;
		dim3 gridSort( numBlocksXSort, numBlocksYSort );		
		unsigned int memSizeRecordsR = sizeof( cmp_type_t ) * numRecordsR;
		//copy the <offset, length> pairs.
		cmp_type_t* d_R;
		//checkCudaErrors
		( cudaMalloc( (void**) &d_R, memSizeRecordsR) );
		cmp_type_t tempValue;
		tempValue.x=tempValue.y=-1;
		initialize(d_R, numRecordsR, tempValue);
		( cudaMemcpy( d_R, d_Rin, rLen*sizeof(cmp_type_t), cudaMemcpyDeviceToDevice) );
		

		for( int k = 2; k <= numRecordsR; k *= 2 )
		{
			for( int j = k/2; j > 0; j /= 2 )
			{
				bitonicKernel<<<gridSort, numThreadsSort>>>((void*)d_rawData, totalLenInBytes, d_R, numRecordsR, k, j);
			}
		}
		( cudaMemcpy( d_Rout, d_R+(numRecordsR-rLen), sizeof(cmp_type_t)*rLen, cudaMemcpyDeviceToDevice) );
		cudaFree( d_R );
		cudaThreadSynchronize();
	}
	else
	{
		unsigned int numThreadsSort = NUM_THREADS_CHUNK;
		unsigned int numBlocksYSort = 1;
		unsigned int numBlocksXSort = (numRecordsR/numThreadsSort)/numBlocksYSort;
		if(numBlocksXSort>=(1<<16))
		{
			numBlocksXSort=(1<<15);
			numBlocksYSort=(numRecordsR/numThreadsSort)/numBlocksXSort;			
		}
		unsigned int numBlocksChunk = NUM_BLOCKS_CHUNK;
		unsigned int numThreadsChunk = NUM_THREADS_CHUNK;
		
		unsigned int chunkSize = numBlocksChunk*numThreadsChunk;
		unsigned int numChunksR = numRecordsR/chunkSize;

		dim3 gridSort( numBlocksXSort, numBlocksYSort );
		unsigned int memSizeRecordsR = sizeof( cmp_type_t ) * numRecordsR;

		cmp_type_t* d_R;
		( cudaMalloc( (void**) &d_R, memSizeRecordsR) );
		cmp_type_t tempValue;
		tempValue.x=tempValue.y=-1;
		initialize(d_R, numRecordsR, tempValue);
		( cudaMemcpy( d_R, d_Rin, rLen*sizeof(cmp_type_t), cudaMemcpyDeviceToDevice) );

		for( int chunkIdx = 0; chunkIdx < numChunksR; chunkIdx++ )
		{
			unitBitonicSortKernel<<< numBlocksChunk, numThreadsChunk>>>( (void*)d_rawData, totalLenInBytes, d_R, numRecordsR, chunkIdx );
		}

		int j;
		for( int k = numThreadsChunk*2; k <= numRecordsR; k *= 2 )
		{
			for( j = k/2; j > numThreadsChunk/2; j /= 2 )
			{
				bitonicKernel<<<gridSort, numThreadsSort>>>( (void*)d_rawData, totalLenInBytes, d_R, numRecordsR, k, j);
			}

			for( int chunkIdx = 0; chunkIdx < numChunksR; chunkIdx++ )
			{
				partBitonicSortKernel<<< numBlocksChunk, numThreadsChunk>>>((void*)d_rawData, totalLenInBytes, d_R, numRecordsR, chunkIdx, k/numThreadsSort );
			}
		}
		( cudaMemcpy( d_Rout, d_R+(numRecordsR-rLen), sizeof(cmp_type_t)*rLen, cudaMemcpyDeviceToDevice) );
		cudaFree( d_R );
		cudaThreadSynchronize();
	}
}

__global__ void getIntYArray_kernel(int2* d_input, int start_row_idPos, int rLen, int* d_output)
{

}


__global__ void getXYArray_kernel(cmp_type_t* d_input, int start_row_idPos, int rLen, int2* d_output)
{

}

__global__ void getZWArray_kernel(cmp_type_t* d_input, int start_row_idPos, int rLen, int2* d_output)
{

}


__global__ void setXYArray_kernel(cmp_type_t* d_input, int start_row_idPos, int rLen, int2* d_value)
{

}

__global__ void setZWArray_kernel(cmp_type_t* d_input, int start_row_idPos, int rLen, int2* d_value)
{

}

void getIntYArray(int2 *d_data, int rLen, int* d_output)
{

}

void getXYArray(cmp_type_t *d_data, int rLen, int2* d_output)
{

}

void getZWArray(cmp_type_t *d_data, int rLen, int2* d_output)
{
	cudaThreadSynchronize();
}



void setXYArray(cmp_type_t *d_data, int rLen, int2* d_value)
{
	cudaThreadSynchronize();
}

void setZWArray(cmp_type_t *d_data, int rLen, int2* d_value)
{
	cudaThreadSynchronize();
}
__global__ void copyChunks_kernel(void *d_source, int start_row_idPos, int2* d_Rin, int rLen, int *d_sum, void *d_dest)
{

}

__global__ void getChunkBoundary_kernel(void* d_rawData, int start_row_idPos, cmp_type_t *d_Rin, 
										int rLen, int* d_start_row_idArray)
{

}

__global__ void setBoundaryInt2_kernel(int* d_boundary, int start_row_idPos, int numKey, int rLen,
										  int2* d_boundaryRange)
{

}

__global__ void writeBoundary_kernel(int start_row_idPos, int rLen, int* d_start_row_idArray,
									int* d_start_row_idSumArray, int* d_bounary)
{

}

void copyChunks(void *d_source, int2* d_Rin, int rLen, void *d_dest)
{
}

//return the number of chunks.
int getChunkBoundary(void *d_source, cmp_type_t* d_Rin, int rLen, int2 ** h_outputKeyListRange)
{
	return 0;
}	
	
__global__ void copyDataFromDevice2Host1(gpu_context d_g_state)
{	
	
	int num_records_per_thread = (d_g_state.num_input_record+(gridDim.x*blockDim.x)-1)/(gridDim.x*blockDim.x);
	int block_start_row_idx = num_records_per_thread*blockIdx.x*blockDim.x;
	int thread_start_row_idx = block_start_row_idx 
		+ (threadIdx.x/STRIDE)*num_records_per_thread*STRIDE
		+ (threadIdx.x%STRIDE);
	int thread_end_idx = thread_start_row_idx+num_records_per_thread*STRIDE;
	
	//if (TID>=d_g_state.num_input_record)return;
	if(thread_end_idx>d_g_state.num_input_record)
		thread_end_idx = d_g_state.num_input_record;
	
	for(int map_task_idx=thread_start_row_idx; map_task_idx < thread_end_idx; map_task_idx+=STRIDE){
	
		int begin=0;
		int end=0;
		for (int i=0;i<map_task_idx;i++){
			begin += d_g_state.d_intermediate_keyval_arr_arr_p[i]->arr_len;
		}//for

	end = begin + (d_g_state.d_intermediate_keyval_arr_arr_p[map_task_idx]->arr_len);
	//printf("map_task_idx:%d begin:%d	end:%d\n",map_task_idx, begin,end);
	
	for(int i=begin;i<end;i++){
		keyval_t * p1 = &(d_g_state.d_intermediate_keyval_arr[i]);
		keyval_t * p2 = &(d_g_state.d_intermediate_keyval_arr_arr_p[map_task_idx]->arr[i-begin]);
		memcpy(p1,p2,sizeof(keyval_t));
		//printf("copyData1: TID:%d keySize %d valSize:%d p2->key:%s  p1->key:%s\n",map_task_idx,p1->keySize,p1->valSize,p2->key,p1->key);
	}//for
	

	}//for int map_task_idx;
}	
	
__global__ void copyDataFromDevice2Host3(gpu_context d_g_state)
{	
	
	int num_records_per_thread = (d_g_state.num_input_record+(gridDim.x*blockDim.x)-1)/(gridDim.x*blockDim.x);
	int block_start_row_idx = num_records_per_thread*blockIdx.x*blockDim.x;
	int thread_start_row_idx = block_start_row_idx 
		+ (threadIdx.x/STRIDE)*num_records_per_thread*STRIDE
		+ (threadIdx.x%STRIDE);
	int thread_end_idx = thread_start_row_idx+num_records_per_thread*STRIDE;
	
	//if (TID>=d_g_state.num_input_record)return;
	if(thread_end_idx>d_g_state.num_input_record)
		thread_end_idx = d_g_state.num_input_record;
	
	int begin, end, val_pos, key_pos;
	char *val_p,*key_p;
	
	for(int map_task_idx=thread_start_row_idx; map_task_idx < thread_end_idx; map_task_idx+=STRIDE){
		
		begin=0;
		end=0;
		for (int i=0;i<map_task_idx;i++){
			begin += (d_g_state.d_intermediate_keyval_arr_arr_p[i]->arr_len);
		}//for
		end = begin + (d_g_state.d_intermediate_keyval_arr_arr_p[map_task_idx]->arr_len);
		//printf("copyData:%d begin:%d, end:%d\n",TID,begin,end);
		
		for(int i=begin;i<end;i++){
			
			//keyval_t * p1 = &(d_g_state.d_intermediate_keyval_arr[i]);
			val_pos = d_g_state.d_intermediate_keyval_pos_arr[i].valPos;
			key_pos = d_g_state.d_intermediate_keyval_pos_arr[i].keyPos;
			
			/*if (key_pos>=d_g_state.totalKeySize){
				printf("keyPos2:%d   totalKeySize:%d begin:%d end:%d  i:%d map_task_idx:%d\n",key_pos,d_g_state.totalKeySize, begin, end, i, map_task_idx);
				key_pos = 0;
			}
			if (val_pos>=d_g_state.totalValSize){
				//printf("keyPos:%d   totalKeySize:%d begin:%d end:%d  i:%d map_task_idx:%d\n",key_pos,d_g_state.totalKeySize, begin, end, i, map_task_idx);
				val_pos = 0;
			}*/

			val_p = (char*)(d_g_state.d_intermediate_vals_shared_buff)+val_pos;
			key_p = (char*)(d_g_state.d_intermediate_keys_shared_buff)+key_pos;
			keyval_t * p2 = &(d_g_state.d_intermediate_keyval_arr_arr_p[map_task_idx]->arr[i-begin]);
			
			memcpy(key_p,p2->key,p2->keySize);
			//int totalKeySize;
			//int totalValSize;
			memcpy(val_p,p2->val,p2->valSize);
			//added by Hui
			//free(p2->key);
			//free(p2->val);
			//free(p2);

			//printf("copyDataFromDevice2Host2: TID:%d key: %s  val:%d\n",TID,p2->key,*(int *)p2->val);
		}//for
		//free(d_g_state.d_intermediate_keyval_arr_arr[map_task_idx].arr);
		//if (index*recordsPerTask >= recordNum) return;
		//free(&d_g_state.d_intermediate_keyval_arr_arr[map_task_idx])
	}//for
	//free(d_g_state.d_intermediate_keyval_pos_arr);
	//
}//__global__	
		
#ifdef REMOVE
__global__ void copyDataFromDevice2Host2(gpu_context d_g_state)
{	
	
	int num_records_per_thread = (d_g_state.num_input_record+(gridDim.x*blockDim.x)-1)/(gridDim.x*blockDim.x);
	int block_start_row_idx = num_records_per_thread*blockIdx.x*blockDim.x;
	int thread_start_row_idx = block_start_row_idx 
		+ (threadIdx.x/STRIDE)*num_records_per_thread*STRIDE
		+ (threadIdx.x%STRIDE);
	int thread_end_idx = thread_start_row_idx+num_records_per_thread*STRIDE;

	//if (TID>=d_g_state.num_input_record)return;
	if(thread_end_idx>d_g_state.num_input_record)
		thread_end_idx = d_g_state.num_input_record;

	for(int map_task_idx=thread_start_row_idx; map_task_idx < thread_end_idx; map_task_idx+=STRIDE){
	
		int begin=0;
		int end=0;
		for (int i=0;i<map_task_idx;i++){
			begin += (d_g_state.d_intermediate_keyval_arr_arr[i].arr_len);
		}//for
		end = begin + (d_g_state.d_intermediate_keyval_arr_arr[map_task_idx].arr_len);
		//printf("copyData:%d begin:%d, end:%d\n",TID,begin,end);
	
		for(int i=begin;i<end;i++){
			keyval_t * p1 = &(d_g_state.d_intermediate_keyval_arr[i]);
			keyval_t * p2 = &(d_g_state.d_intermediate_keyval_arr_arr[map_task_idx].arr[i-begin]);
			memcpy(p1->key,p2->key,p2->keySize);
			memcpy(p1->val,p2->val,p2->valSize);
			//printf("copyDataFromDevice2Host2: TID:%d key: %s  val:%d\n",TID,p2->key,*(int *)p2->val);
		}//for
		//if (index*recordsPerTask >= recordNum) return;
	}//for
}//__global__	
#endif

void StartCPUShuffle2(thread_info_t *thread_info){
	
	cpu_context *d_g_state = (cpu_context*)(thread_info->d_g_state);
	job_configuration *cpu_job_conf = (job_configuration*)(thread_info->job_conf);


	//TODO put all jobs related object to job_conf
	bool configured;	
	int cpu_group_id;	
	int num_input_record;
	int num_cpus;	
								
	keyval_t * input_keyval_arr;
	keyval_arr_t *intermediate_keyval_arr_arr_p = d_g_state->intermediate_keyval_arr_arr_p;
					
	long total_count = 0;
	int index = 0;
	for(int i=0;i<d_g_state->num_input_record;i++){
		total_count += intermediate_keyval_arr_arr_p[i].arr_len;
	}//for

	
	
	d_g_state->sorted_intermediate_keyvals_arr = NULL;
	keyvals_t * sorted_intermediate_keyvals_arr = d_g_state->sorted_intermediate_keyvals_arr;

	int sorted_key_arr_len = 0;
	for(int i=0;i<d_g_state->num_input_record;i++){
		int len = intermediate_keyval_arr_arr_p[i].arr_len;
		for (int j=0;j<len;j++){
					
			char *key_i = (char *)(intermediate_keyval_arr_arr_p[i].arr[j].key);
			int keySize_i = (intermediate_keyval_arr_arr_p[i].arr[j].keySize);
								
			char *val_i = (char *)(intermediate_keyval_arr_arr_p[i].arr[j].val);
			int valSize_i = (intermediate_keyval_arr_arr_p[i].arr[j].valSize);
			
			int k = 0;
			for (; k<sorted_key_arr_len; k++){
				char *key_k = (char *)(sorted_intermediate_keyvals_arr[k].key);
				int keySize_k = sorted_intermediate_keyvals_arr[k].keySize;
					 
				if ( cpu_compare(key_i, keySize_i, key_k, keySize_k) != 0 )
				continue;
				
				//found the match
				val_t *vals = sorted_intermediate_keyvals_arr[k].vals;
				sorted_intermediate_keyvals_arr[k].val_arr_len++;
				sorted_intermediate_keyvals_arr[k].vals = (val_t*)realloc(vals, sizeof(val_t)*(sorted_intermediate_keyvals_arr[k].val_arr_len));
				
				int index = sorted_intermediate_keyvals_arr[k].val_arr_len - 1;
				sorted_intermediate_keyvals_arr[k].vals[index].valSize = valSize_i;
				sorted_intermediate_keyvals_arr[k].vals[index].val = (char *)malloc(sizeof(char)*valSize_i);
				memcpy(sorted_intermediate_keyvals_arr[k].vals[index].val,val_i,valSize_i);
				break;

			}//for

			if (k == sorted_key_arr_len){
				
				if (sorted_key_arr_len == 0)
					sorted_intermediate_keyvals_arr = NULL;

				sorted_key_arr_len++;
				sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(sorted_intermediate_keyvals_arr, sizeof(keyvals_t)*sorted_key_arr_len);
				int index = sorted_key_arr_len-1;
				keyvals_t* kvals_p = (keyvals_t *)&(sorted_intermediate_keyvals_arr[index]);
				
				kvals_p->keySize = keySize_i;
				kvals_p->key = malloc(sizeof(char)*keySize_i);
				memcpy(kvals_p->key, key_i, keySize_i);
				
				kvals_p->vals = (val_t *)malloc(sizeof(val_t));
				kvals_p->val_arr_len = 1;

				kvals_p->vals[0].valSize = valSize_i;
				kvals_p->vals[0].val = (char *)malloc(sizeof(char)*valSize_i);
				memcpy(kvals_p->vals[0].val,val_i, valSize_i);

			}//if
		}//for j;
	}//for i;
	d_g_state->sorted_intermediate_keyvals_arr = sorted_intermediate_keyvals_arr;
	d_g_state->sorted_keyvals_arr_len = sorted_key_arr_len;

	DoLog("CPU_GROUP_ID:[%d] #Intermediate Records:%d; #Intermediate Records:%d After Shuffle",d_g_state->cpu_group_id, total_count,sorted_key_arr_len);

}


void StartCPUShuffle(cpu_context *d_g_state){
#ifdef DEV_MODE
	bool configured;	
	int cpu_group_id;	
	int num_input_record;
	int num_cpus;	
							
	keyval_t * input_keyval_arr;
	keyval_arr_t *intermediate_keyval_arr_arr_p = d_g_state->intermediate_keyval_arr_arr_p;
					
	long total_count = 0;
	int index = 0;
	for(int i=0;i<d_g_state->num_input_record;i++){
		total_count += intermediate_keyval_arr_arr_p[i].arr_len;
	}//for

	DoLog("total intermediate record count:%d\n",total_count);
	
	d_g_state->sorted_intermediate_keyvals_arr = NULL;
	keyvals_t * sorted_intermediate_keyvals_arr = d_g_state->sorted_intermediate_keyvals_arr;

	int sorted_key_arr_len = 0;
	for(int i=0;i<d_g_state->num_input_record;i++){
		int len = intermediate_keyval_arr_arr_p[i].arr_len;
		for (int j=0;j<len;j++){
					
			char *key_i = (char *)(intermediate_keyval_arr_arr_p[i].arr[j].key);
			int keySize_i = (intermediate_keyval_arr_arr_p[i].arr[j].keySize);
			
					
			char *val_i = (char *)(intermediate_keyval_arr_arr_p[i].arr[j].val);
			int valSize_i = (intermediate_keyval_arr_arr_p[i].arr[j].valSize);
			
			int k = 0;
			for (; k<sorted_key_arr_len; k++){
				char *key_k = (char *)(sorted_intermediate_keyvals_arr[k].key);
				int keySize_k = sorted_intermediate_keyvals_arr[k].keySize;
					 
				if ( cpu_compare(key_i, keySize_i, key_k, keySize_k) != 0 )
				continue;
				
				//found the match
				val_t *vals = sorted_intermediate_keyvals_arr[k].vals;
				sorted_intermediate_keyvals_arr[k].val_arr_len++;
				sorted_intermediate_keyvals_arr[k].vals = (val_t*)realloc(vals, sizeof(val_t)*(sorted_intermediate_keyvals_arr[k].val_arr_len));
				
				int index = sorted_intermediate_keyvals_arr[k].val_arr_len - 1;
				sorted_intermediate_keyvals_arr[k].vals[index].valSize = valSize_i;
				sorted_intermediate_keyvals_arr[k].vals[index].val = (char *)malloc(sizeof(char)*valSize_i);
				memcpy(sorted_intermediate_keyvals_arr[k].vals[index].val,val_i,valSize_i);
				break;

			}//for

			if (k == sorted_key_arr_len){
				
				if (sorted_key_arr_len == 0)
					sorted_intermediate_keyvals_arr = NULL;

				sorted_key_arr_len++;
				sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(sorted_intermediate_keyvals_arr, sizeof(keyvals_t)*sorted_key_arr_len);
				int index = sorted_key_arr_len-1;
				keyvals_t* kvals_p = (keyvals_t *)&(sorted_intermediate_keyvals_arr[index]);
				
				kvals_p->keySize = keySize_i;
				kvals_p->key = malloc(sizeof(char)*keySize_i);
				memcpy(kvals_p->key, key_i, keySize_i);
				
				kvals_p->vals = (val_t *)malloc(sizeof(val_t));
				kvals_p->val_arr_len = 1;

				kvals_p->vals[0].valSize = valSize_i;
				kvals_p->vals[0].val = (char *)malloc(sizeof(char)*valSize_i);
				memcpy(kvals_p->vals[0].val,val_i, valSize_i);

			}//if
		}//for j;
	}//for i;
	d_g_state->sorted_intermediate_keyvals_arr = sorted_intermediate_keyvals_arr;
	d_g_state->sorted_keyvals_arr_len = sorted_key_arr_len;
	DoLog("total number of different intermediate records:%d",sorted_key_arr_len);
	
#endif
}



void Shuffle4GPUOutput(gpu_context* d_g_state){
	
	cudaThreadSynchronize();
	int *count_arr = (int *)malloc(sizeof(int) * d_g_state->num_input_record);
	//DoLog("begin to copy data from device to host memory  num_input_record:%d",d_g_state->num_input_record);
	//DoLog("allocate memory for d_intermediate_keyval_total_count size:%d\n",sizeof(int)*d_g_state->num_input_record);
	checkCudaErrors(cudaMemcpy(count_arr, d_g_state->d_intermediate_keyval_total_count, sizeof(int)*d_g_state->num_input_record, cudaMemcpyDeviceToHost));

	long total_count = 0;
	int index = 0;
	for(int i=0;i<d_g_state->num_input_record;i++){
		//printf("arr_len[%d]=:%d\n",i,count_arr[i]);
		total_count += count_arr[i];
		index++;
	}//for
	free(count_arr);

	checkCudaErrors(cudaMalloc((void **)&(d_g_state->d_intermediate_keyval_arr),sizeof(keyval_t)*total_count));
	
	int num_blocks = (d_g_state->num_mappers + (NUM_THREADS)-1)/(NUM_THREADS);
	copyDataFromDevice2Host1<<<num_blocks,NUM_THREADS>>>(*d_g_state);

	//copyDataFromDevice2Host1<<<NUM_BLOCKS,NUM_THREADS>>>(*d_g_state);
	cudaThreadSynchronize();

	keyval_t * h_keyval_buff = (keyval_t *)malloc(sizeof(keyval_t)*total_count);
	checkCudaErrors(cudaMemcpy(h_keyval_buff, d_g_state->d_intermediate_keyval_arr, sizeof(keyval_t)*total_count, cudaMemcpyDeviceToHost));
	d_g_state->h_intermediate_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*total_count);
	keyval_pos_t *h_intermediate_keyvals_pos_arr = d_g_state->h_intermediate_keyval_pos_arr;

	int totalKeySize = 0;
	int totalValSize = 0;

	for (int i=0;i<total_count;i++){
		h_intermediate_keyvals_pos_arr[i].valPos= totalValSize;
		h_intermediate_keyvals_pos_arr[i].keyPos = totalKeySize;
		
		h_intermediate_keyvals_pos_arr[i].keySize = h_keyval_buff[i].keySize;
		h_intermediate_keyvals_pos_arr[i].valSize = h_keyval_buff[i].valSize;
		totalKeySize += h_keyval_buff[i].keySize;
		totalValSize += h_keyval_buff[i].valSize;
	}//for
	d_g_state->totalValSize = totalValSize;
	d_g_state->totalKeySize = totalKeySize;
		
	//DoLog("totalKeySize:%d totalValSize:%d ",totalKeySize,totalValSize);
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_intermediate_keys_shared_buff,totalKeySize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_intermediate_vals_shared_buff,totalValSize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_intermediate_keyval_pos_arr,sizeof(keyval_pos_t)*total_count));
														
	checkCudaErrors(cudaMemcpy(d_g_state->d_intermediate_keyval_pos_arr, h_intermediate_keyvals_pos_arr, sizeof(keyval_pos_t)*total_count, cudaMemcpyHostToDevice));
																								
	//DoLog("copyDataFromDevice2Host3");
	cudaThreadSynchronize();
	copyDataFromDevice2Host3<<<num_blocks,NUM_THREADS>>>(*d_g_state);
	//printData<<<NUM_BLOCKS,NUM_THREADS>>>(*d_g_state);

	cudaThreadSynchronize();

	d_g_state->h_intermediate_keys_shared_buff = malloc(sizeof(char)*totalKeySize);
	d_g_state->h_intermediate_vals_shared_buff = malloc(sizeof(char)*totalValSize);
	
	checkCudaErrors(cudaMemcpy(d_g_state->h_intermediate_keys_shared_buff,d_g_state->d_intermediate_keys_shared_buff,sizeof(char)*totalKeySize,cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(d_g_state->h_intermediate_vals_shared_buff,d_g_state->d_intermediate_vals_shared_buff,sizeof(char)*totalValSize,cudaMemcpyDeviceToHost));
	
	/*	for(int i=0;i<total_count;i++){
		printf("keySize:%d, valSize:%d  key:%s val:%d\n",h_buff[i].keySize,h_buff[i].valSize,(char *)h_buff[i].key,*(int *)h_buff[i].val);
	}//for	*/
	
	//////////////////////////////////////////////
		
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_sorted_keys_shared_buff,totalKeySize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_sorted_vals_shared_buff,totalValSize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_keyval_pos_arr,sizeof(keyval_pos_t)*total_count));
	
	d_g_state->h_sorted_keys_shared_buff = malloc(sizeof(char)*totalKeySize);
	d_g_state->h_sorted_vals_shared_buff = malloc(sizeof(char)*totalValSize);
	//d_g_state->h_sorted_keyval_pos_arr = (sorted_keyval_pos_t *)malloc(sizeof(sorted_keyval_pos_t)*total_count);
	
	char *sorted_keys_shared_buff = (char *)d_g_state->h_sorted_keys_shared_buff;
	char *sorted_vals_shared_buff = (char *)d_g_state->h_sorted_vals_shared_buff;
	//sorted_keyval_pos_t * h_sorted_keyval_pos_arr = d_g_state->h_sorted_keyval_pos_arr;

	char *intermediate_key_shared_buff = (char *)d_g_state->h_intermediate_keys_shared_buff;
	char *intermediate_val_shared_buff = (char *)d_g_state->h_intermediate_vals_shared_buff;

	memcpy(sorted_keys_shared_buff, intermediate_key_shared_buff, totalKeySize);
	memcpy(sorted_vals_shared_buff, intermediate_val_shared_buff, totalValSize);

	int sorted_key_arr_len = 0;
	
	///////////////////////////////////////////////////////////////////////////////////////////////////
	//transfer the d_sorted_keyval_pos_arr to h_sorted_keyval_pos_arr
	//DoLog("transfer the d_sorted_keyval_pos_arr to h_sorted_keyval_pos_arr");

	sorted_keyval_pos_t * h_sorted_keyval_pos_arr = NULL;

	for (int i=0; i<total_count; i++){
		int iKeySize = h_intermediate_keyvals_pos_arr[i].keySize;
		
		int j = 0;

		for (; j<sorted_key_arr_len; j++){
			int jKeySize = h_sorted_keyval_pos_arr[j].keySize;

			char *key_i = (char *)(intermediate_key_shared_buff + h_intermediate_keyvals_pos_arr[i].keyPos);
			char *key_j = (char *)(sorted_keys_shared_buff + h_sorted_keyval_pos_arr[j].keyPos);

			if (cpu_compare(key_i,iKeySize,key_j,jKeySize)!=0)
				continue;
			

			//found the match
			int arr_len = h_sorted_keyval_pos_arr[j].val_arr_len;
			h_sorted_keyval_pos_arr[j].val_pos_arr = (val_pos_t *)realloc(h_sorted_keyval_pos_arr[j].val_pos_arr, sizeof(val_pos_t)*(arr_len+1));
			h_sorted_keyval_pos_arr[j].val_pos_arr[arr_len].valSize = h_intermediate_keyvals_pos_arr[i].valSize;
			h_sorted_keyval_pos_arr[j].val_pos_arr[arr_len].valPos = h_intermediate_keyvals_pos_arr[i].valPos;
			h_sorted_keyval_pos_arr[j].val_arr_len += 1;
			break;
		}//for
		
		if(j==sorted_key_arr_len){
			sorted_key_arr_len++;
			//printf("d_g_state->d_sorted_keyvals_arr_len:%d\n",d_g_state->d_sorted_keyvals_arr_len);
			h_sorted_keyval_pos_arr = (sorted_keyval_pos_t *)realloc(h_sorted_keyval_pos_arr,sorted_key_arr_len*sizeof(sorted_keyval_pos_t));
			
			sorted_keyval_pos_t *p = &(h_sorted_keyval_pos_arr[sorted_key_arr_len - 1]);
			p->keySize = iKeySize;
			p->keyPos = h_intermediate_keyvals_pos_arr[i].keyPos;

			p->val_arr_len = 1;
			p->val_pos_arr = (val_pos_t*)malloc(sizeof(val_pos_t));
			p->val_pos_arr[0].valSize = h_intermediate_keyvals_pos_arr[i].valSize;
			p->val_pos_arr[0].valPos = h_intermediate_keyvals_pos_arr[i].valPos;
		}//if
	}
	
	d_g_state->h_sorted_keyval_pos_arr = h_sorted_keyval_pos_arr;
	d_g_state->d_sorted_keyvals_arr_len = sorted_key_arr_len;
	
	keyval_pos_t *tmp_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*total_count);
	
	DoLog("GPU_ID:[%d] #input_records:%d #intermediate_records:%lu #different_intermediate_records:%d totalKeySize:%d totalValSize:%d", 
		d_g_state->gpu_id, d_g_state->num_input_record, total_count, sorted_key_arr_len,totalKeySize,totalValSize);
	
	int *pos_arr_4_pos_arr = (int*)malloc(sizeof(int)*sorted_key_arr_len);
	memset(pos_arr_4_pos_arr,0,sizeof(int)*sorted_key_arr_len);

	index = 0;
	for (int i=0;i<sorted_key_arr_len;i++){
		sorted_keyval_pos_t *p = (sorted_keyval_pos_t *)&(h_sorted_keyval_pos_arr[i]);
		
		for (int j=0;j<p->val_arr_len;j++){
			tmp_keyval_pos_arr[index].keyPos = p->keyPos;
			tmp_keyval_pos_arr[index].keySize = p->keySize;
			tmp_keyval_pos_arr[index].valPos = p->val_pos_arr[j].valPos;
			tmp_keyval_pos_arr[index].valSize = p->val_pos_arr[j].valSize;
			//printf("tmp_keyval_pos_arr[%d].keyPos:%d\n",index,p->keyPos);
			index++;
		}//for
		pos_arr_4_pos_arr[i] = index;
	}
	
	checkCudaErrors(cudaMemcpy(d_g_state->d_keyval_pos_arr,tmp_keyval_pos_arr,sizeof(keyval_pos_t)*total_count,cudaMemcpyHostToDevice));
	d_g_state->d_sorted_keyvals_arr_len = sorted_key_arr_len;

	checkCudaErrors(cudaMalloc((void**)&d_g_state->d_pos_arr_4_sorted_keyval_pos_arr,sizeof(int)*sorted_key_arr_len));
	checkCudaErrors(cudaMemcpy(d_g_state->d_pos_arr_4_sorted_keyval_pos_arr,pos_arr_4_pos_arr,sizeof(int)*sorted_key_arr_len,cudaMemcpyHostToDevice));

	/*verify the d_sorted_keyval_arr_len results
	for (int i=0;i<d_g_state->d_sorted_keyvals_arr_len;i++){
		keyvals_t *p = &(d_g_state->h_sorted_keyvals_arr[i]);
		printf("sort CPU 3 key:%s len:%d",p->key,p->val_arr_len);
		for (int j=0;j<p->val_arr_len;j++)
			printf("\t%d",*(int*)p->vals[j].val);
		printf("\n");
	}//for */
		
	//start_row_id sorting
	//partition
}


//host function sort_CPU
//copy intermediate records from device memory to host memory and sort the intermediate records there. 
//The host API cannot copy from dynamically allocated addresses on device runtime heap, only device code can access them

void sort_CPU(gpu_context* d_g_state){

#ifdef REMOVE
	
	//start_row_id sorting
	//partition

#endif

}



void PandaShuffleMergeCPU(panda_context *d_g_state_0, cpu_context *d_g_state_1){
	DoLog("PandaShuffleMergeCPU CPU_GROUP_ID:[%d]", d_g_state_1->cpu_group_id);
	
	keyvals_t * panda_sorted_intermediate_keyvals_arr = d_g_state_0->sorted_intermediate_keyvals_arr;

	keyvals_t * cpu_sorted_intermediate_keyvals_arr = d_g_state_1->sorted_intermediate_keyvals_arr;

	void *key_0, *key_1;
	int keySize_0, keySize_1;
	bool equal;	

	for (int i=0; i<d_g_state_1->sorted_keyvals_arr_len; i++){
		key_1 = cpu_sorted_intermediate_keyvals_arr[i].key;
		keySize_1 = cpu_sorted_intermediate_keyvals_arr[i].keySize;
			
		int j;
		for (j=0; j<d_g_state_0->sorted_keyvals_arr_len; j++){
			key_0 = panda_sorted_intermediate_keyvals_arr[j].key;
			keySize_0 = panda_sorted_intermediate_keyvals_arr[j].keySize;
			
			if(cpu_compare(key_0,keySize_0,key_1,keySize_1)!=0)
				continue;
			
		
			//copy values from cpu_contex to panda context
			int val_arr_len_1 = cpu_sorted_intermediate_keyvals_arr[i].val_arr_len;
			int index = panda_sorted_intermediate_keyvals_arr[j].val_arr_len;
			if (panda_sorted_intermediate_keyvals_arr[j].val_arr_len ==0)
				panda_sorted_intermediate_keyvals_arr[j].vals = NULL;
			panda_sorted_intermediate_keyvals_arr[j].val_arr_len += val_arr_len_1;
			
			val_t *vals = panda_sorted_intermediate_keyvals_arr[j].vals;
			panda_sorted_intermediate_keyvals_arr[j].vals = (val_t*)realloc(vals, sizeof(val_t)*(panda_sorted_intermediate_keyvals_arr[j].val_arr_len));

			for (int k=0;k<val_arr_len_1;k++){
				char *val_0 = (char *)(cpu_sorted_intermediate_keyvals_arr[i].vals[k].val);
				int valSize_0 = cpu_sorted_intermediate_keyvals_arr[i].vals[k].valSize;

				panda_sorted_intermediate_keyvals_arr[j].vals[index+k].val = malloc(sizeof(char)*valSize_0);
				panda_sorted_intermediate_keyvals_arr[j].vals[index+k].valSize = valSize_0;
				memcpy(panda_sorted_intermediate_keyvals_arr[j].vals[index+k].val, val_0, valSize_0);

			}//for
			break;
		}//for
		
		if (j == d_g_state_0->sorted_keyvals_arr_len){

			if (d_g_state_0->sorted_keyvals_arr_len == 0) panda_sorted_intermediate_keyvals_arr = NULL;

			val_t *vals = cpu_sorted_intermediate_keyvals_arr[i].vals;
			int val_arr_len = cpu_sorted_intermediate_keyvals_arr[i].val_arr_len;

			d_g_state_0->sorted_keyvals_arr_len++;
			panda_sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(panda_sorted_intermediate_keyvals_arr, 
				sizeof(keyvals_t)*(d_g_state_0->sorted_keyvals_arr_len));

			int index = d_g_state_0->sorted_keyvals_arr_len-1;
			keyvals_t* kvals_p = (keyvals_t *)&(panda_sorted_intermediate_keyvals_arr[index]);
				
			kvals_p->keySize = keySize_1;
			kvals_p->key = malloc(sizeof(char)*keySize_1);
			memcpy(kvals_p->key, key_1, keySize_1);
				
			kvals_p->vals = (val_t *)malloc(sizeof(val_t)*val_arr_len);
			kvals_p->val_arr_len = val_arr_len;

			for (int k=0; k < val_arr_len; k++){
				char *val_0 = (char *)(cpu_sorted_intermediate_keyvals_arr[i].vals[k].val);
				int valSize_0 = cpu_sorted_intermediate_keyvals_arr[i].vals[k].valSize;

				kvals_p->vals[k].valSize = valSize_0;
				kvals_p->vals[k].val = (char *)malloc(sizeof(char)*valSize_0);

				memcpy(kvals_p->vals[k].val,val_0, valSize_0);

			}//for
		}//if (j == sorted_key_arr_len){
	}//if
	d_g_state_0->sorted_intermediate_keyvals_arr = cpu_sorted_intermediate_keyvals_arr;
	DoLog("CPU_GROUP_ID:[%d] DONE.",d_g_state_1->cpu_group_id);

}


void PandaShuffleMergeGPU(panda_context *d_g_state_1, gpu_context *d_g_state_0){
	
	DoLog("PandaShuffleMergeGPU GPU_ID:[%d]",d_g_state_0->gpu_id);
	
	char *sorted_keys_shared_buff_0 = (char *)d_g_state_0->h_sorted_keys_shared_buff;
	char *sorted_vals_shared_buff_0 = (char *)d_g_state_0->h_sorted_vals_shared_buff;

	sorted_keyval_pos_t *keyval_pos_arr_0 = d_g_state_0->h_sorted_keyval_pos_arr;
	keyvals_t * sorted_intermediate_keyvals_arr = d_g_state_1->sorted_intermediate_keyvals_arr;

	void *key_0, *key_1;
	int keySize_0, keySize_1;
	bool equal;	
	

	for (int i=0;i<d_g_state_0->d_sorted_keyvals_arr_len;i++){
		//DoLog("keyPos:%d",keyval_pos_arr_0[i].keyPos);
		key_0 = sorted_keys_shared_buff_0 + keyval_pos_arr_0[i].keyPos;
		keySize_0 = keyval_pos_arr_0[i].keySize;
			
		int j = 0;
		for (; j<d_g_state_1->sorted_keyvals_arr_len; j++){
			
			key_1 = sorted_intermediate_keyvals_arr[j].key;
			
			keySize_1 = sorted_intermediate_keyvals_arr[j].keySize;
			
			if(cpu_compare(key_0,keySize_0,key_1,keySize_1)!=0)
				continue;
			
			val_t *vals = sorted_intermediate_keyvals_arr[j].vals;
			//copy values from gpu to cpu context
			int val_arr_len_0 =keyval_pos_arr_0[i].val_arr_len;
			val_pos_t * val_pos_arr =keyval_pos_arr_0[i].val_pos_arr;

			int index = sorted_intermediate_keyvals_arr[j].val_arr_len;
			sorted_intermediate_keyvals_arr[j].val_arr_len += val_arr_len_0;
			sorted_intermediate_keyvals_arr[j].vals = (val_t*)realloc(vals, sizeof(val_t)*(sorted_intermediate_keyvals_arr[j].val_arr_len));
			
			for (int k=0;k<val_arr_len_0;k++){
				
				char *val_0 = sorted_vals_shared_buff_0 + val_pos_arr[k].valPos;
				int valSize_0 = val_pos_arr[k].valSize;
			
				sorted_intermediate_keyvals_arr[j].vals[index+k].val = malloc(sizeof(char)*valSize_0);
				sorted_intermediate_keyvals_arr[j].vals[index+k].valSize = valSize_0;
				memcpy(sorted_intermediate_keyvals_arr[j].vals[index+k].val, val_0, valSize_0);
			}//for
			break;
		}//for
	
		if (j == d_g_state_1->sorted_keyvals_arr_len){
				
			if (d_g_state_1->sorted_keyvals_arr_len == 0) sorted_intermediate_keyvals_arr = NULL;

			//val_t *vals = sorted_intermediate_keyvals_arr[j].vals;
			int val_arr_len =keyval_pos_arr_0[i].val_arr_len;
			
			val_pos_t * val_pos_arr =keyval_pos_arr_0[i].val_pos_arr;
			
			d_g_state_1->sorted_keyvals_arr_len++;
			sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(sorted_intermediate_keyvals_arr, sizeof(keyvals_t)*(d_g_state_1->sorted_keyvals_arr_len));
			
			int index = d_g_state_1->sorted_keyvals_arr_len-1;
			keyvals_t* kvals_p = (keyvals_t *)&(sorted_intermediate_keyvals_arr[index]);
				
			kvals_p->keySize = keySize_0;
			kvals_p->key = malloc(sizeof(char)*keySize_0);
			memcpy(kvals_p->key, key_0, keySize_0);
				
			kvals_p->vals = (val_t *)malloc(sizeof(val_t)*val_arr_len);
			kvals_p->val_arr_len = val_arr_len;

			for (int k=0; k < val_arr_len; k++){
				char *val_0 = sorted_vals_shared_buff_0 + val_pos_arr[k].valPos;
				int valSize_0 = val_pos_arr[k].valSize;

				kvals_p->vals[k].valSize = valSize_0;
				kvals_p->vals[k].val = (char *)malloc(sizeof(char)*valSize_0);
				memcpy(kvals_p->vals[k].val,val_0, valSize_0);
			}//for
		}//if (j == sorted_key_arr_len){
	}//if

	d_g_state_1->sorted_intermediate_keyvals_arr = sorted_intermediate_keyvals_arr;
	DoLog("GPU_ID:[%d] DONE",d_g_state_0->gpu_id);
}
			
void Panda_Shuffle_Merge(gpu_context *d_g_state_0, gpu_context *d_g_state_1){
			
	char *sorted_keys_shared_buff_0 = (char *)d_g_state_0->h_sorted_keys_shared_buff;
	char *sorted_vals_shared_buff_0 = (char *)d_g_state_0->h_sorted_vals_shared_buff;
			
	char *sorted_keys_shared_buff_1 = (char *)d_g_state_1->h_sorted_keys_shared_buff;
	char *sorted_vals_shared_buff_1 = (char *)d_g_state_1->h_sorted_vals_shared_buff;
			
	sorted_keyval_pos_t *keyval_pos_arr_0 = d_g_state_0->h_sorted_keyval_pos_arr;
	sorted_keyval_pos_t *keyval_pos_arr_1 = d_g_state_1->h_sorted_keyval_pos_arr;
			
	int totalValSize_1 = d_g_state_1->totalValSize;
	int totalKeySize_1 = d_g_state_1->totalKeySize;
			
	void *key_0,*key_1;
	int keySize_0,keySize_1;
	bool equal;	
	//DoLog("len1:%d  len2:%d\n",d_g_state_0->d_sorted_keyvals_arr_len, d_g_state_1->d_sorted_keyvals_arr_len);
	for (int i=0;i<d_g_state_0->d_sorted_keyvals_arr_len;i++){
		key_0 = sorted_keys_shared_buff_0 + keyval_pos_arr_0[i].keyPos;
		keySize_0 = keyval_pos_arr_0[i].keySize;
			
		int j;
		for (j=0;j<d_g_state_1->d_sorted_keyvals_arr_len;j++){
			key_1 = sorted_keys_shared_buff_1 + keyval_pos_arr_1[j].keyPos;
			keySize_1 = keyval_pos_arr_1[j].keySize;
			
			if(cpu_compare(key_0,keySize_0,key_1,keySize_1)!=0)
				continue;
			
			//copy all vals in d_g_state_0->h_sorted_keyval_pos_arr[i] to d_g_state_1->h_sorted_keyval_pos_arr[j];
			int incValSize = 0;
			int len0 = keyval_pos_arr_0[i].val_arr_len;
			int len1 = keyval_pos_arr_1[j].val_arr_len;
			//DoLog("i:%d j:%d compare: key_0:%s key_1:%s  true:%s len0:%d len1:%d\n", i, j, key_0,key_1,(equal ? "true":"false"),len0,len1);
			keyval_pos_arr_1[j].val_pos_arr = (val_pos_t*)realloc(keyval_pos_arr_1[j].val_pos_arr,sizeof(val_pos_t)*(len0+len1));
			keyval_pos_arr_1[j].val_arr_len = len0+len1;

			for (int k = len1; k < len1 + len0; k++){
				keyval_pos_arr_1[j].val_pos_arr[k].valSize = keyval_pos_arr_0[i].val_pos_arr[k-len1].valSize;
				keyval_pos_arr_1[j].val_pos_arr[k].valPos = keyval_pos_arr_0[i].val_pos_arr[k-len1].valPos;
				incValSize += keyval_pos_arr_0[i].val_pos_arr[k-len1].valSize;
			}//for
			
			sorted_vals_shared_buff_1 = (char*)realloc(sorted_vals_shared_buff_1, totalValSize_1 + incValSize);
			for (int k = len1; k < len1 + len0; k++){
				void *val_1 = sorted_vals_shared_buff_1 + totalValSize_1;
				void *val_0 = sorted_vals_shared_buff_0+keyval_pos_arr_0[i].val_pos_arr[k-len1].valPos;
				memcpy(val_1, val_0, keyval_pos_arr_0[i].val_pos_arr[k-len1].valSize);
				totalValSize_1 += keyval_pos_arr_0[i].val_pos_arr[k-len1].valSize;
			}//for
			break;
		}//for (int j = 0;
	
		//key_0 is not exist in d_g_state_1->h_sorted_keyval_pos_arr, create new keyval pair position there
		if(j==d_g_state_1->d_sorted_keyvals_arr_len){
					
			sorted_keys_shared_buff_1 = (char*)realloc(sorted_keys_shared_buff_1, (totalKeySize_1 + keySize_0));
			//assert(keySize_0 == keyval_pos_arr_0[i].keySize);
			
			void *key_0 = sorted_keys_shared_buff_0 + keyval_pos_arr_0[i].keyPos;
			void *key_1 = sorted_keys_shared_buff_1 + totalKeySize_1;
			
			memcpy(key_1, key_0, keySize_0);
			totalKeySize_1 += keySize_0;
			
			keyval_pos_arr_1 = (sorted_keyval_pos_t *)realloc(keyval_pos_arr_1, sizeof(sorted_keyval_pos_t)*(d_g_state_1->d_sorted_keyvals_arr_len+1));
			sorted_keyval_pos_t *new_p = &(keyval_pos_arr_1[d_g_state_1->d_sorted_keyvals_arr_len]);
			d_g_state_1->d_sorted_keyvals_arr_len += 1;
			
			new_p->keySize = keySize_0;
			new_p->keyPos = totalKeySize_1 - keySize_0;
			
			int len0 = keyval_pos_arr_0[i].val_arr_len;
			new_p->val_arr_len = len0;
			new_p->val_pos_arr = (val_pos_t *)malloc(sizeof(val_pos_t)*len0);
			
			int incValSize = 0;
			for (int k = 0; k < len0; k++){
				new_p->val_pos_arr[k].valSize = keyval_pos_arr_0[i].val_pos_arr[k].valSize;
				new_p->val_pos_arr[k].valPos = keyval_pos_arr_0[i].val_pos_arr[k].valPos;
				incValSize += keyval_pos_arr_0[i].val_pos_arr[k].valSize;
			}//for
 			sorted_vals_shared_buff_1 = (char*)realloc(sorted_vals_shared_buff_1,(totalValSize_1 + incValSize));
			
			for (int k = 0; k < len0; k++){
				void *val_1 = sorted_vals_shared_buff_1 + totalValSize_1;
				void *val_0 = sorted_vals_shared_buff_0 + keyval_pos_arr_0[i].val_pos_arr[k].valPos;
				memcpy(val_1,val_0,keyval_pos_arr_0[i].val_pos_arr[k].valSize);
				totalValSize_1 += keyval_pos_arr_0[i].val_pos_arr[k].valSize;
			}//for
		}//if(j==arr_len)
	}//for (int i = 0;
	
	d_g_state_1->h_sorted_keyval_pos_arr = keyval_pos_arr_1;
	
	int total_count = 0;
	for (int i=0; i<d_g_state_1->d_sorted_keyvals_arr_len; i++){
		total_count += d_g_state_1->h_sorted_keyval_pos_arr[i].val_arr_len;
	}//for
	DoLog("total number of intermeidate records on two GPU's:%d",total_count);
	keyval_pos_t *tmp_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*total_count);
	DoLog("total number of different intermediate records on two GPU's:%d",d_g_state_1->d_sorted_keyvals_arr_len);
	
	int *pos_arr_4_pos_arr = (int*)malloc(sizeof(int)*d_g_state_1->d_sorted_keyvals_arr_len);
	memset(pos_arr_4_pos_arr,0,sizeof(int)*d_g_state_1->d_sorted_keyvals_arr_len);
	
	int	index = 0;
	for (int i=0; i<d_g_state_1->d_sorted_keyvals_arr_len; i++){
		sorted_keyval_pos_t *p = (sorted_keyval_pos_t *)&(d_g_state_1->h_sorted_keyval_pos_arr[i]);
			
		for (int j=0;j<p->val_arr_len;j++){
			tmp_keyval_pos_arr[index].keyPos = p->keyPos;
			tmp_keyval_pos_arr[index].keySize = p->keySize;
			tmp_keyval_pos_arr[index].valPos = p->val_pos_arr[j].valPos;
			tmp_keyval_pos_arr[index].valSize = p->val_pos_arr[j].valSize;
			//printf("tmp_keyval_pos_arr[%d].keyPos:%d  keySize:%d valPos:%d valSize:%d\n",
			//index,p->keyPos,p->keySize,p->val_pos_arr[j].valPos,p->val_pos_arr[j].valSize);
			//printf("key:%s val:%d\n",(char*)(sorted_keys_shared_buff_1+p->keyPos), *(int*)(sorted_vals_shared_buff_1+p->val_pos_arr[j].valPos));
			index++;
		}//for
		pos_arr_4_pos_arr[i] = index;
	}
	
	//printf("totalKeySize_1:%d  totalValSize_1:%d\n",totalKeySize_1,totalValSize_1);
	//printf("%s\n",sorted_keys_shared_buff_1);
	
	checkCudaErrors(cudaMalloc((void**)&d_g_state_1->d_keyval_pos_arr,sizeof(keyval_pos_t)*total_count));
	checkCudaErrors(cudaMemcpy(d_g_state_1->d_keyval_pos_arr,tmp_keyval_pos_arr,sizeof(keyval_pos_t)*total_count,cudaMemcpyHostToDevice));
	//d_g_state_1->d_sorted_keyvals_arr_len = d_g_state_1->d_sorted_keyvals_arr_len;
	checkCudaErrors(cudaMalloc((void**)&d_g_state_1->d_pos_arr_4_sorted_keyval_pos_arr,sizeof(int)*d_g_state_1->d_sorted_keyvals_arr_len));
	checkCudaErrors(cudaMemcpy(d_g_state_1->d_pos_arr_4_sorted_keyval_pos_arr,pos_arr_4_pos_arr,sizeof(int)*d_g_state_1->d_sorted_keyvals_arr_len,cudaMemcpyHostToDevice));
		
	//TODO release these buffer bebore allocate
	checkCudaErrors(cudaMalloc((void **)&d_g_state_1->d_sorted_keys_shared_buff,totalKeySize_1));
	checkCudaErrors(cudaMalloc((void **)&d_g_state_1->d_sorted_vals_shared_buff,totalValSize_1));
	
	checkCudaErrors(cudaMemcpy(d_g_state_1->d_sorted_keys_shared_buff,sorted_keys_shared_buff_1,totalKeySize_1,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_g_state_1->d_sorted_vals_shared_buff,sorted_vals_shared_buff_1,totalValSize_1,cudaMemcpyHostToDevice));

	//d_g_state_1->d_sorted_keys_shared_buff = sorted_keys_shared_buff_1; 
	//d_g_state_1->d_sorted_vals_shared_buff = sorted_vals_shared_buff_1;
	d_g_state_1->totalKeySize = totalKeySize_1;
	d_g_state_1->totalValSize = totalValSize_1;
	
}

#endif 
