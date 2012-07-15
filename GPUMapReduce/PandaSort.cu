/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: PandaSort.cu 
	Time: 2012-07-01 
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
#define NUM_BLOCKS_CHUNK 256//(512)
#define	NUM_THREADS_CHUNK 256//(256)
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
	int startCopy=SHARED_MEM_INT2-rLen;
	if(tid>=startCopy)
	{
		d_output[tid-startCopy]=bs_cmpbuf[tid];
	}
}

__global__ void bitonicSortMultipleBlocks_kernel(void* d_rawData, int totalLenInBytes, cmp_type_t * d_values, int* d_bound, int startBlock, int numBlock, cmp_type_t *d_output)
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
		bs_pStart=d_bound[(bid+startBlock)<<1];
		bs_pEnd=d_bound[((bid+startBlock)<<1)+1];
		bs_numElement=bs_pEnd-bs_pStart;
		//if(bid==82&& bs_pStart==6339)
		//	printf("%d, %d, %d\n", bs_pStart, bs_pEnd, bs_numElement);
		
	}
	__syncthreads();
    // Copy input to shared mem.
	if(tid<bs_numElement)
	{
		bs_shared[tid] = d_values[tid+bs_pStart];
		//if(bid==82 && bs_pStart==6339)
		//	printf("tid %d, pos, %d, %d, %d, %d\n", tid,tid+bs_pStart, bs_pStart,bs_pEnd, d_values[tid+bs_pStart].x);
		//if(6342==tid+bs_pStart)
		//	printf(")))tid %d, pos, %d, %d, %d, %d\n", tid,tid+bs_pStart, bs_pStart,bs_pEnd, d_values[tid+bs_pStart].x);
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
	//int startCopy=SHARED_MEM_INT2-bs_numElement;
	if(tid>=bs_numElement)
	{
		d_output[tid-bs_numElement]=bs_shared[tid];
	}
}


__global__ void initialize_kernel(cmp_type_t* d_data, int startPos, int rLen, cmp_type_t value)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	d_data[pos]=value;
}
void bitonicSortMultipleBlocks(void* d_rawData, int totalLenInBytes, cmp_type_t * d_values, int* d_bound, int numBlock, cmp_type_t * d_output)
{
	int numThreadsPerBlock_x=SHARED_MEM_INT2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=NUM_BLOCK_PER_CHUNK_BITONIC_SORT;
	int numBlock_y=1;
	int numChunk=numBlock/numBlock_x;
	if(numBlock%numBlock_x!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*numBlock_x;
		end=start+numBlock_x;
		if(end>numBlock)
			end=numBlock;
		//printf("bitonicSortMultipleBlocks_kernel: %d, range, %d, %d\n", i, start, end);
		bitonicSortMultipleBlocks_kernel<<<grid,thread>>>(d_rawData, totalLenInBytes, d_values, d_bound, start, end-start, d_output);
		cudaThreadSynchronize();
	}
//	cudaThreadSynchronize();
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
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		initialize_kernel<<<grid,thread>>>(d_data, start, rLen, value);
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

__global__ void getIntYArray_kernel(int2* d_input, int startPos, int rLen, int* d_output)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		int2 value=d_input[pos];
		d_output[pos]=value.y;
	}
}


__global__ void getXYArray_kernel(cmp_type_t* d_input, int startPos, int rLen, int2* d_output)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		cmp_type_t value=d_input[pos];
		d_output[pos].x=value.x;
		d_output[pos].y=value.y;
	}
}

__global__ void getZWArray_kernel(cmp_type_t* d_input, int startPos, int rLen, int2* d_output)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		cmp_type_t value=d_input[pos];
		d_output[pos].x=value.z;
		d_output[pos].y=value.w;
	}
}


__global__ void setXYArray_kernel(cmp_type_t* d_input, int startPos, int rLen, int2* d_value)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		cmp_type_t value=d_input[pos];
		value.x=d_value[pos].x;
		value.y=d_value[pos].y;
		d_input[pos]=value;
	}
}

__global__ void setZWArray_kernel(cmp_type_t* d_input, int startPos, int rLen, int2* d_value)
{
    const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		cmp_type_t value=d_input[pos];
		value.z=d_value[pos].x;
		value.w=d_value[pos].y;
		d_input[pos]=value;
	}
}

void getIntYArray(int2 *d_data, int rLen, int* d_output)
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
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		getIntYArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_output);
	} 
	cudaThreadSynchronize();
}

void getXYArray(cmp_type_t *d_data, int rLen, int2* d_output)
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
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		getXYArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_output);
	} 
	cudaThreadSynchronize();
}

void getZWArray(cmp_type_t *d_data, int rLen, int2* d_output)
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
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		getZWArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_output);
	} 
	cudaThreadSynchronize();
}



void setXYArray(cmp_type_t *d_data, int rLen, int2* d_value)
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
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		setXYArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_value);
	} 
	cudaThreadSynchronize();
}

void setZWArray(cmp_type_t *d_data, int rLen, int2* d_value)
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
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		setZWArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_value);
	} 
	cudaThreadSynchronize();
}
__global__ void copyChunks_kernel(void *d_source, int startPos, int2* d_Rin, int rLen, int *d_sum, void *d_dest)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	
	if(pos<rLen)
	{
		int2 value=d_Rin[pos];
		int offset=value.x;
		int size=value.y;
		int startWritePos=d_sum[pos];
		int i=0;
		char *source=(char*)d_source;
		char *dest=(char*)d_dest;
		for(i=0;i<size;i++)
		{
			dest[i+startWritePos]=source[i+offset];
		}
		value.x=startWritePos;
		d_Rin[pos]=value;
	}
}

__global__ void getChunkBoundary_kernel(void* d_rawData, int startPos, cmp_type_t *d_Rin, 
										int rLen, int* d_startArray)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	
	if(pos<rLen)
	{
		int result=0;
		if(pos==0)//the start position
		{
			result=1;
		}
		else
		{
			cmp_type_t cur=d_Rin[pos];
			cmp_type_t left=d_Rin[pos-1];
			if(getCompareValue(d_rawData, cur, left)!=0)
			{
				result=1;
			}
		}
		d_startArray[pos]=result;	
	}
}

__global__ void setBoundaryInt2_kernel(int* d_boundary, int startPos, int numKey, int rLen,
										  int2* d_boundaryRange)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	
	if(pos<numKey)
	{
		int2 flag;
		flag.x=d_boundary[pos];
		if((pos+1)!=numKey)
			flag.y=d_boundary[pos+1];
		else
			flag.y=rLen;
		d_boundaryRange[pos]=flag;
	}
}

__global__ void writeBoundary_kernel(int startPos, int rLen, int* d_startArray,
									int* d_startSumArray, int* d_bounary)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	
	if(pos<rLen)
	{
		int flag=d_startArray[pos];
		int writePos=d_startSumArray[pos];
		if(flag==1)
			d_bounary[writePos]=pos;
	}
}

void copyChunks(void *d_source, int2* d_Rin, int rLen, void *d_dest)
{
	//extract the size information for each chunk
	int* d_size;
	( cudaMalloc( (void**) (&d_size), sizeof(int)*rLen) );	
	getIntYArray(d_Rin, rLen, d_size);
	//compute the prefix sum for the output positions.
	int* d_sum;
	( cudaMalloc( (void**) (&d_sum), sizeof(int)*rLen) );
	saven_initialPrefixSum(rLen);
	prescanArray(d_sum,d_size,rLen);
	cudaFree(d_size);
	//output
	int numThreadsPerBlock_x=128;
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
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		copyChunks_kernel<<<grid,thread>>>(d_source, start, d_Rin, rLen, d_sum, d_dest);
	} 
	cudaThreadSynchronize();
	
	cudaFree(d_sum);
	
}
//return the number of chunks.
int getChunkBoundary(void *d_source, cmp_type_t* d_Rin, int rLen, int2 ** h_outputKeyListRange)
{
	int resultNumChunks=0;
	//get the chunk boundary[start of chunk0, start of chunk 1, ...]
	int* d_startArray;
	( cudaMalloc( (void**) (&d_startArray), sizeof(int)*rLen) );	
	
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
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		getChunkBoundary_kernel<<<grid,thread>>>(d_source, start, d_Rin, rLen, d_startArray);
	} 
	cudaThreadSynchronize();
	//prefix sum for write positions.
	int* d_startSumArray;
	( cudaMalloc( (void**) (&d_startSumArray), sizeof(int)*rLen) );
	saven_initialPrefixSum(rLen);
	prescanArray(d_startSumArray,d_startArray,rLen);

	//gpuPrint(d_startSumArray, rLen, "d_startSumArray");

	int lastValue=0;
	int partialSum=0;
	( cudaMemcpy( &lastValue, d_startArray+(rLen-1), sizeof(int), cudaMemcpyDeviceToHost) );
	//gpuPrint(d_startArray, rLen, "d_startArray");
	( cudaMemcpy( &partialSum, d_startSumArray+(rLen-1), sizeof(int), cudaMemcpyDeviceToHost) );
	//gpuPrint(d_startSumArray, rLen, "d_startSumArray");
	resultNumChunks=lastValue+partialSum;

	int* d_boundary;//[start of chunk0, start of chunk 1, ...]
	( cudaMalloc( (void**) (&d_boundary), sizeof(int)*resultNumChunks) );

	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		writeBoundary_kernel<<<grid,thread>>>(start, rLen, d_startArray,
									d_startSumArray, d_boundary);
	} 
	cudaFree(d_startArray);
	cudaFree(d_startSumArray);	

	//set the int2 boundary. 
	int2 *d_outputKeyListRange;
	( cudaMalloc( (void**) (&d_outputKeyListRange), sizeof(int2)*resultNumChunks) );
	numChunk=resultNumChunks/chunkSize;
	if(resultNumChunks%chunkSize!=0)
		numChunk++;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>resultNumChunks)
			end=resultNumChunks;
		setBoundaryInt2_kernel<<<grid,thread>>>(d_boundary, start, resultNumChunks, rLen, d_outputKeyListRange);
	} 
	cudaThreadSynchronize();
	
	*h_outputKeyListRange=(int2*)malloc(sizeof(int2)*resultNumChunks);
	( cudaMemcpy( *h_outputKeyListRange, d_outputKeyListRange, sizeof(int2)*resultNumChunks, cudaMemcpyDeviceToHost) );
	
	cudaFree(d_boundary);
	cudaFree(d_outputKeyListRange);
	return resultNumChunks;
}	
	
__global__ void copyDataFromDevice2Host1(d_global_state d_g_state)
{	
	
	int num_records_per_thread = (d_g_state.h_num_input_record+(gridDim.x*blockDim.x)-1)/(gridDim.x*blockDim.x);
	int block_start_idx = num_records_per_thread*blockIdx.x*blockDim.x;
	int thread_start_idx = block_start_idx 
		+ (threadIdx.x/STRIDE)*num_records_per_thread*STRIDE
		+ (threadIdx.x%STRIDE);
	int thread_end_idx = thread_start_idx+num_records_per_thread*STRIDE;
	
	//if (TID>=d_g_state.h_num_input_record)return;
	if(thread_end_idx>d_g_state.h_num_input_record)
		thread_end_idx = d_g_state.h_num_input_record;
	
	for(int map_task_idx=thread_start_idx; map_task_idx < thread_end_idx; map_task_idx+=STRIDE){
	
	int begin=0;
	int end=0;
	for (int i=0;i<map_task_idx;i++){
		begin += d_g_state.d_intermediate_keyval_arr_arr[i].arr_len;
	}//for
	end = begin + (d_g_state.d_intermediate_keyval_arr_arr[map_task_idx].arr_len);
	//printf("copyData:%d begin:%d, end:%d\n",TID,begin,end);
	for(int i=begin;i<end;i++){
		keyval_t * p1 = &(d_g_state.d_intermediate_keyval_arr[i]);
		keyval_t * p2 = &(d_g_state.d_intermediate_keyval_arr_arr[map_task_idx].arr[i-begin]);
		memcpy(p1,p2,sizeof(keyval_t));
		//printf("copyDataFromDevice2Host1 key:%s  val%d\n",p2->key,*(int *)p2->val);
		//printf("copyData1: TID:%d keySize %d valSize:%d p2->key:%s  p1->key:%s\n",map_task_idx,p1->keySize,p1->valSize,p2->key,p1->key);
	}//for
	//if (index*recordsPerTask >= recordNum) return;
	
	}
}	
	
__global__ void copyDataFromDevice2Host3(d_global_state d_g_state)
{	
	
	int num_records_per_thread = (d_g_state.h_num_input_record+(gridDim.x*blockDim.x)-1)/(gridDim.x*blockDim.x);
	int block_start_idx = num_records_per_thread*blockIdx.x*blockDim.x;
	int thread_start_idx = block_start_idx 
		+ (threadIdx.x/STRIDE)*num_records_per_thread*STRIDE
		+ (threadIdx.x%STRIDE);
	int thread_end_idx = thread_start_idx+num_records_per_thread*STRIDE;
	
	//if (TID>=d_g_state.h_num_input_record)return;
	if(thread_end_idx>d_g_state.h_num_input_record)
		thread_end_idx = d_g_state.h_num_input_record;
	
	int begin, end, val_pos, key_pos;
	char *val_p,*key_p;
	
	for(int map_task_idx=thread_start_idx; map_task_idx < thread_end_idx; map_task_idx+=STRIDE){
		
		begin=0;
		end=0;
		for (int i=0;i<map_task_idx;i++){
			begin += (d_g_state.d_intermediate_keyval_arr_arr[i].arr_len);
		}//for
		end = begin + (d_g_state.d_intermediate_keyval_arr_arr[map_task_idx].arr_len);
		//printf("copyData:%d begin:%d, end:%d\n",TID,begin,end);
		
		for(int i=begin;i<end;i++){
			//keyval_t * p1 = &(d_g_state.d_intermediate_keyval_arr[i]);
			val_pos = d_g_state.d_intermediate_keyval_pos_arr[i].valPos;
			key_pos = d_g_state.d_intermediate_keyval_pos_arr[i].keyPos;
			val_p = (char*)(d_g_state.d_intermediate_vals_shared_buff)+val_pos;
			key_p = (char*)(d_g_state.d_intermediate_keys_shared_buff)+key_pos;
			keyval_t * p2 = &(d_g_state.d_intermediate_keyval_arr_arr[map_task_idx].arr[i-begin]);
			memcpy(key_p,p2->key,p2->keySize);
			memcpy(val_p,p2->val,p2->valSize);
			//printf("copyDataFromDevice2Host2: TID:%d key: %s  val:%d\n",TID,p2->key,*(int *)p2->val);
		}//for
		//if (index*recordsPerTask >= recordNum) return;
	}//for
}//__global__	
		
		
__global__ void copyDataFromDevice2Host2(d_global_state d_g_state)
{	
	
	int num_records_per_thread = (d_g_state.h_num_input_record+(gridDim.x*blockDim.x)-1)/(gridDim.x*blockDim.x);
	int block_start_idx = num_records_per_thread*blockIdx.x*blockDim.x;
	int thread_start_idx = block_start_idx 
		+ (threadIdx.x/STRIDE)*num_records_per_thread*STRIDE
		+ (threadIdx.x%STRIDE);
	int thread_end_idx = thread_start_idx+num_records_per_thread*STRIDE;

	//if (TID>=d_g_state.h_num_input_record)return;
	if(thread_end_idx>d_g_state.h_num_input_record)
		thread_end_idx = d_g_state.h_num_input_record;

	for(int map_task_idx=thread_start_idx; map_task_idx < thread_end_idx; map_task_idx+=STRIDE){
	
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




void sort_CPU3(d_global_state* d_g_state){
	
	cudaThreadSynchronize();
	DoLog("sort CPU start begin to copy data from device to host memory len:%d",d_g_state->h_num_input_record);
	int *count_arr = (int *)malloc(sizeof(int)*d_g_state->h_num_input_record);
	DoLog("allocate memory for d_intermediate_keyval_total_count size:%d\n",sizeof(int)*d_g_state->h_num_input_record);
	checkCudaErrors(cudaMemcpy(count_arr, d_g_state->d_intermediate_keyval_total_count, sizeof(int)*d_g_state->h_num_input_record, cudaMemcpyDeviceToHost));
	long total_count = 0;
	
	int index = 0;
	for(int i=0;i<d_g_state->h_num_input_record;i++){
		//printf("arr_len[%d]=:%d\n",i,count_arr[i]);
		total_count += count_arr[i];
		index++;
	}//for
	
	DoLog("total_count:%lu  num_input_records:%d\n", total_count, d_g_state->h_num_input_record);
	checkCudaErrors(cudaMalloc((void **)&(d_g_state->d_intermediate_keyval_arr),sizeof(keyval_t)*total_count));
	copyDataFromDevice2Host1<<<NUM_BLOCKS,NUM_THREADS>>>(*d_g_state);
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
	DoLog("totalKeySize:%d totalValSize:%d\n",totalKeySize,totalValSize);
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_intermediate_keys_shared_buff,totalKeySize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_intermediate_vals_shared_buff,totalValSize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_intermediate_keyval_pos_arr,sizeof(keyval_pos_t)*total_count));
	cudaMemcpy(d_g_state->d_intermediate_keyval_pos_arr,h_intermediate_keyvals_pos_arr,sizeof(keyval_pos_t)*total_count,cudaMemcpyHostToDevice);
	
	DoLog("copyDataFromDevice2Host3");
	cudaThreadSynchronize();
	copyDataFromDevice2Host3<<<NUM_BLOCKS,NUM_THREADS>>>(*d_g_state);
	
	//printData<<<NUM_BLOCKS,NUM_THREADS>>>(*d_g_state);
	cudaThreadSynchronize();
	
	d_g_state->h_intermediate_keys_shared_buff = malloc(sizeof(char)*totalKeySize);
	d_g_state->h_intermediate_vals_shared_buff = malloc(sizeof(char)*totalValSize);
	cudaMemcpy(d_g_state->h_intermediate_keys_shared_buff,d_g_state->d_intermediate_keys_shared_buff,sizeof(char)*totalKeySize,cudaMemcpyDeviceToHost);
	cudaMemcpy(d_g_state->h_intermediate_vals_shared_buff,d_g_state->d_intermediate_vals_shared_buff,sizeof(char)*totalValSize,cudaMemcpyDeviceToHost);
	
	/*	for(int i=0;i<total_count;i++){
		printf("keySize:%d, valSize:%d  key:%s val:%d\n",h_buff[i].keySize,h_buff[i].valSize,(char *)h_buff[i].key,*(int *)h_buff[i].val);
	}//for	*/
	
	//////////////////////////////////////////////
		
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_sorted_keys_shared_buff,totalKeySize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_sorted_vals_shared_buff,totalValSize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_keyval_pos_arr,sizeof(keyval_pos_t)*total_count));
	
	d_g_state->h_sorted_keys_shared_buff = malloc(sizeof(char)*totalKeySize);
	d_g_state->h_sorted_vals_shared_buff = malloc(sizeof(char)*totalValSize);
	d_g_state->h_sorted_keyval_pos_arr = (sorted_keyval_pos_t *)malloc(sizeof(sorted_keyval_pos_t)*total_count);
	
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
	DoLog("transfer the d_sorted_keyval_pos_arr to h_sorted_keyval_pos_arr");

	sorted_keyval_pos_t * h_sorted_keyval_pos_arr = NULL;

	for (int i=0; i<total_count; i++){
		int iKeySize = h_intermediate_keyvals_pos_arr[i].keySize;
		
		int j = 0;

		for (; j<sorted_key_arr_len; j++){
			int jKeySize = h_sorted_keyval_pos_arr[j].keySize;
			if (iKeySize!=jKeySize)
				continue;

			bool equal = true;
			for (int k=0;k<iKeySize;k++){
				char *p1 = (char *)(intermediate_key_shared_buff + h_intermediate_keyvals_pos_arr[i].keyPos);
				char *p2 = (char *)(sorted_keys_shared_buff + h_sorted_keyval_pos_arr[j].keyPos);
				if (p1[k] != p2[k]){
					equal = false;
					break;
				}//if
			}//for
			if (!equal)
				continue;

			int arr_len = h_sorted_keyval_pos_arr[j].val_arr_len;
			h_sorted_keyval_pos_arr[j].val_pos_arr = (val_pos_t *)realloc(h_sorted_keyval_pos_arr[j].val_pos_arr, sizeof(val_pos_t)*(arr_len+1));
			h_sorted_keyval_pos_arr[j].val_pos_arr[arr_len].valSize = h_intermediate_keyvals_pos_arr[i].valSize;
			h_sorted_keyval_pos_arr[j].val_pos_arr[arr_len].valPos = h_intermediate_keyvals_pos_arr[i].valPos;
			h_sorted_keyval_pos_arr[j].val_arr_len +=1;
			break;//found the match
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

	keyval_pos_t *tmp_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*total_count);
	
	printf("sorted_keyval_pos_arr_len:%d\n",sorted_key_arr_len);

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
	
	cudaMemcpy(d_g_state->d_keyval_pos_arr,tmp_keyval_pos_arr,sizeof(keyval_pos_t)*total_count,cudaMemcpyHostToDevice);
	d_g_state->d_sorted_keyvals_arr_len = sorted_key_arr_len;

	cudaMalloc((void**)&d_g_state->d_pos_arr_4_sorted_keyval_pos_arr,sizeof(int)*sorted_key_arr_len);
	cudaMemcpy(d_g_state->d_pos_arr_4_sorted_keyval_pos_arr,pos_arr_4_pos_arr,sizeof(int)*sorted_key_arr_len,cudaMemcpyHostToDevice);

	/*verify the d_sorted_keyval_arr_len results
	for (int i=0;i<d_g_state->d_sorted_keyvals_arr_len;i++){
		keyvals_t *p = &(d_g_state->h_sorted_keyvals_arr[i]);
		printf("sort CPU 3 key:%s len:%d",p->key,p->val_arr_len);
		for (int j=0;j<p->val_arr_len;j++)
			printf("\t%d",*(int*)p->vals[j].val);
		printf("\n");
	}//for */
		
	//////////////////////////////////////////////////////////////////////////////
	//start sorting
	//partition
}






//host function sort_CPU
//copy intermediate records from device memory to host memory and sort the intermediate records there. 
//The host API cannot copy from dynamically allocated addresses on device runtime heap, only device code can access them

void sort_CPU(d_global_state* d_g_state){

#ifdef SORT_CPU

	cudaThreadSynchronize();
	DoLog("sort CPU start begin to copy data from device to host memory len:%d",d_g_state->h_num_input_record);

	int *count_arr = (int *)malloc(sizeof(int)*d_g_state->h_num_input_record);
	DoLog("allocate memory for d_intermediate_keyval_total_count size:%d\n",sizeof(int)*d_g_state->h_num_input_record);
	//checkCudaErrors(cudaMalloc((void**)&(d_g_state->d_intermediate_keyval_total_count),sizeof(int)*d_g_state->h_num_input_record));
	checkCudaErrors(cudaMemcpy(count_arr, d_g_state->d_intermediate_keyval_total_count, sizeof(int)*d_g_state->h_num_input_record, cudaMemcpyDeviceToHost));
	long total_count = 0;

	int index = 0;
	for(int i=0;i<d_g_state->h_num_input_record;i++){
		printf("arr_len[%d]=:%d\n",i,count_arr[i]);
		total_count += count_arr[i];
		index++;
		//if (index>1500&&index<1550)
		//	printf("index:%d  \ttotal_count1:%lu\n",index, total_count);
	}//for
	
	//int input_record_count = d_g_state->h_num_input_record;
	//keyval_arr_t *h_keyval_arr_arr = d_g_state->d_intermediate_keyval_arr_arr;
	//keyval_arr_t *d_keyval_arr_arr = d_g_state->d_intermediate_keyval_arr_arr;
	//DoLog("sort CPU 1 total_count:%d  num_input_record:%d",total_count,d_g_state->h_num_input_record);
	printf("total_count2:%lu  num_input_records:%d\n", total_count, d_g_state->h_num_input_record);

	

	
	checkCudaErrors(cudaMalloc((void **)&(d_g_state->d_intermediate_keyval_arr),sizeof(keyval_t)*total_count));
	//copyDataFromDevice2Host1<<<1,d_g_state->h_num_input_record>>>(*d_g_state);
	copyDataFromDevice2Host1<<<NUM_BLOCKS,NUM_THREADS>>>(*d_g_state);
	cudaThreadSynchronize();
	
	//printData<<<1,d_g_state->h_num_input_record>>>( *d_g_state);

	keyval_t * h_buff = (keyval_t *)malloc(sizeof(keyval_t)*total_count);
	checkCudaErrors(cudaMemcpy(h_buff, d_g_state->d_intermediate_keyval_arr, sizeof(keyval_t)*total_count, cudaMemcpyDeviceToHost));
	
	//void *pkey= (void*)malloc(sizeof(char));
	//void *pval= (void*)malloc(sizeof(char));
	
	
	int totalKeySize = 0;
	int totalValSize = 0;

	for (int i=0;i<total_count;i++){
		h_buff[i].valPos = totalValSize;
		h_buff[i].keyPos = totalKeySize;
		totalKeySize += h_buff[i].keySize;
		totalValSize += h_buff[i].valSize;
	}//for
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_intermediate_keys_shared_buff,totalKeySize));
	checkCudaErrors(cudaMalloc((void **)&d_g_state->d_intermediate_vals_shared_buff,totalValSize));
	
	
	for (int i=0;i<total_count;i++){
		//printf("sort_CPU cudaMalloc keySize:%d, valSize:%d		i:%d\n",h_buff[i].keySize,h_buff[i].valSize,i);
		checkCudaErrors(cudaMalloc((void **)&(h_buff[i].key),h_buff[i].keySize));
		checkCudaErrors(cudaMalloc((void **)&(h_buff[i].val),h_buff[i].valSize));
		//cudaMemset((h_buff[i].key), 0, h_buff[i].keySize);
		//cudaMemset((h_buff[i].val), 0, h_buff[i].valSize);
		//(d_g_state.d_intermediate_keyval_arr[i]);
	}//for

	
	//NOTE:copy h_buff[i].key and h_buff[i].key address to d_intermediate_keyval_arr
	cudaMemcpy(d_g_state->d_intermediate_keyval_arr,h_buff,sizeof(keyval_t)*total_count,cudaMemcpyHostToDevice);
	//cudaThreadSynchronize();

	DoLog("sort CPU 2 total_count");
	//printData<<<1,d_g_state->h_num_input_record>>>(*d_g_state);
	cudaThreadSynchronize();
	copyDataFromDevice2Host2<<<NUM_BLOCKS,NUM_THREADS>>>(*d_g_state);

	cudaThreadSynchronize();
	//printData<<<1,d_g_state->h_num_input_record>>>(*d_g_state);
	cudaThreadSynchronize();
	
	for(int i=0;i<total_count;i++){
		void *p1=h_buff[i].key;
		void *p2=h_buff[i].val;
		
		h_buff[i].key = (void *)malloc(h_buff[i].keySize);
		h_buff[i].val = (void *)malloc(h_buff[i].valSize);
		(cudaMemcpy(h_buff[i].key, p1, h_buff[i].keySize, cudaMemcpyDeviceToHost));
		(cudaMemcpy(h_buff[i].val, p2, h_buff[i].valSize, cudaMemcpyDeviceToHost));
		//printf("key:%s, val:%s\n",h_buff[i].key,h_buff[i].val);

	}//for

	/*
	for(int i=0;i<total_count;i++){
		printf("keySize:%d, valSize:%d  key:%s val:%d\n",h_buff[i].keySize,h_buff[i].valSize,(char *)h_buff[i].key,*(int *)h_buff[i].val);
	}//for
	*/

	//d_g_state->d_sorted_keyvals_arr_alloc_len = d_g_state->h_num_input_record;
	d_g_state->d_sorted_keyvals_arr_len = 0;
	//d_g_state->h_sorted_keyvals_arr = NULL;//(keyvals_t *)malloc(sizeof(keyvals_t)*d_g_state->d_sorted_keyvals_arr_alloc_len);

	for (int i=0;i<total_count;i++){
		int iKeySize = h_buff[i].keySize;
		int j = 0;
		for (;j<d_g_state->d_sorted_keyvals_arr_len;j++){
			int jKeySize = d_g_state->h_sorted_keyvals_arr[j].keySize;
			if (iKeySize!=jKeySize)
				continue;
			bool equal = true;
			for (int k=0;k<iKeySize;k++){
				char *p1 = (char *)h_buff[i].key;
				char *p2 = (char *)(d_g_state->h_sorted_keyvals_arr[j].key);
				if (p1[k]!=p2[k])
					equal = false;
			}//for
			if (!equal)
				continue;
			
			keyvals_t *p = &(d_g_state->h_sorted_keyvals_arr[j]);
			p->val_arr_len = p->val_arr_len+1;
			p->vals = (val_t*)realloc(p->vals,sizeof(val_t)*p->val_arr_len);
			val_t *p1 = &(p->vals[p->val_arr_len-1]);
			//val_t *p1 = (val_t*)malloc(sizeof(val_t));
			p1->valSize = h_buff[i].valSize;
			p1->val = (void *)malloc(sizeof(p1->valSize));
			memcpy(p1->val,h_buff[i].val,h_buff[i].valSize);
			//printf("find same key:%s val:%d\n",h_buff[i].key,*(int*)p1->val);
			break;//found the match
		}//for

		if(j==d_g_state->d_sorted_keyvals_arr_len){
			d_g_state->d_sorted_keyvals_arr_len++;
			d_g_state->h_sorted_keyvals_arr = (keyvals_t*)realloc(d_g_state->h_sorted_keyvals_arr,d_g_state->d_sorted_keyvals_arr_len*sizeof(keyvals_t));
			keyvals_t *p = &(d_g_state->h_sorted_keyvals_arr[d_g_state->d_sorted_keyvals_arr_len-1]);
			p->keySize = iKeySize;
			p->key = (void *)malloc(iKeySize);
			memcpy(p->key,h_buff[i].key,iKeySize);
			p->val_arr_len = 1;
			p->vals = (val_t*)malloc(sizeof(val_t));
			p->vals[0].valSize = h_buff[i].valSize;
			p->vals[0].val = (void *)malloc(p->vals[0].valSize);
			memcpy(p->vals[0].val,h_buff[i].val,h_buff[i].valSize);
		}//if
	}

	/*verify the d_sorted_keyval_arr_len results
	for (int i=0;i<d_g_state->d_sorted_keyvals_arr_len;i++){
		keyvals_t *p = &(d_g_state->h_sorted_keyvals_arr[i]);
		printf("sort CPU 3 key:%s len:%d",p->key,p->val_arr_len);
		for (int j=0;j<p->val_arr_len;j++)
			printf("\t%d",*(int*)p->vals[j].val);
		printf("\n");
	}//for */

	//Hui 7/7/2012
	DoLog("-------------------------sort CPU 3 copy sorted data from host to device");
	DoLog("			d_sorted_keyvals_arr_len:%d\n",d_g_state->d_sorted_keyvals_arr_len);
	//copy h_sorted_keyvals_arr to d_sorted_keyvals_arr for reduce computation
	cudaMalloc((void **)&d_g_state->d_sorted_keyvals_arr,sizeof(keyvals_t)*d_g_state->d_sorted_keyvals_arr_len);
	keyvals_t* keyvals_buff = (keyvals_t*)malloc(sizeof(keyvals_t)*d_g_state->d_sorted_keyvals_arr_len);

	for (int i=0;i<d_g_state->d_sorted_keyvals_arr_len;i++){
		keyvals_t *p = &(d_g_state->h_sorted_keyvals_arr[i]);
		keyvals_buff[i].keySize = p->keySize;
		keyvals_buff[i].val_arr_len = p->val_arr_len;
		checkCudaErrors(cudaMalloc((void**)&(keyvals_buff[i].key),p->keySize));
		checkCudaErrors(cudaMemcpy(keyvals_buff[i].key,p->key,p->keySize,cudaMemcpyHostToDevice));
		
		//TODO check the nested value;//
		//copy nested host data structure to device
		//val_t* val_buff = (val_t*)malloc(sizeof(val_t)*p->val_arr_len);
		val_t* d_val_buff_p = (val_t*)malloc(sizeof(val_t)*p->val_arr_len);
		for (int j=0;j<p->val_arr_len;j++){
			//val_t * val_t_p = (val_t*)malloc(sizeof(val_t));
			
			val_t *p2;
			checkCudaErrors(cudaMalloc((void**)&p2,p->vals[j].valSize));
			checkCudaErrors(cudaMemcpy(p2,p->vals[j].val,p->vals[j].valSize,cudaMemcpyHostToDevice));
			
			d_val_buff_p[j].val = p2;
			d_val_buff_p[j].valSize = p->vals[j].valSize;
			
			//printf("sort cpu check value: i:%d, j:%d, val:%d valSize:%d\n",i,j,*(int *)p->vals[j].val,p->vals[j].valSize);
			//printData4<<<1,1>>>(i,j,val_buff[i].val);
		}//for
		
		checkCudaErrors(cudaMalloc((void**)&(keyvals_buff[i].vals),sizeof(val_t)*p->val_arr_len));
		checkCudaErrors(cudaMemcpy(keyvals_buff[i].vals,d_val_buff_p,sizeof(val_t)*p->val_arr_len, cudaMemcpyHostToDevice));
		//printData4<<<1,1>>>(i,keyvals_buff[i].val_arr_len, keyvals_buff[i].vals);
		//printData4<<<1,1>>>(i,val_buff,p->val_arr_len);
		//cudaMemcpy(&(d_g_state->d_sorted_keyvals_arr[i]),p,sizeof(keyvals_t));
	}//for
	checkCudaErrors(cudaMemcpy(d_g_state->d_sorted_keyvals_arr,keyvals_buff,sizeof(keyvals_t)*d_g_state->d_sorted_keyvals_arr_len,cudaMemcpyHostToDevice));

	//printData3<<<1,d_g_state->d_sorted_keyvals_arr_len>>>(*d_g_state);
	
	//start sorting
	//partition
#endif

}



int sort_GPU (void * d_inputKeyArray, int totalKeySize, void * d_inputValArray, int totalValueSize, 
		  cmp_type_t * d_inputPointerArray, int rLen, 
		  void * d_outputKeyArray, void * d_outputValArray, 
		  cmp_type_t * d_outputPointerArray, int2 ** h_outputKeyListRange
		  )
{
	//array_startTime(1);
	int numDistinctKey=0;
	int totalLenInBytes=-1;
	bitonicSortGPU(d_inputKeyArray, totalLenInBytes, d_inputPointerArray, rLen, d_outputPointerArray);
	//array_endTime("sort", 1);
	//!we first scatter the values and then the keys. so that we can reuse d_PA. 
	int2 *d_PA;
	( cudaMalloc( (void**) (&d_PA), sizeof(int2)*rLen) );	
	//scatter the values.
	if(d_inputValArray!=NULL)
	{
		getZWArray(d_outputPointerArray, rLen, d_PA);
		copyChunks(d_inputValArray, d_PA, rLen, d_outputValArray);
		setZWArray(d_outputPointerArray, rLen, d_PA);
	}
	
	//scatter the keys.
	if(d_inputKeyArray!=NULL)
	{
		getXYArray(d_outputPointerArray, rLen, d_PA);
		copyChunks(d_inputKeyArray, d_PA, rLen, d_outputKeyArray);	
		setXYArray(d_outputPointerArray, rLen, d_PA);
	}
	//find the boudary for each key.

	numDistinctKey=getChunkBoundary(d_outputKeyArray, d_outputPointerArray, rLen, h_outputKeyListRange);

	return numDistinctKey;

}

#endif 
