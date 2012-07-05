/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: Panda.h 
	Time: 2012-07-01 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
*/


#ifndef __PANDA_H__
#define __PANDA_H__

//#include <unistd.h>
//#include <sys/mman.h>
//#include <pthread.h>
//#include <cutil.h>
//#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <stdarg.h>

//helper for shared that are common to CUDA SDK samples
#include <shrUtils.h>
#include <sdkHelper.h>  
#include <shrQATest.h>  


#define _DEBUG 0x01
#define CEIL(n,m) (n/m + (int)(n%m !=0))
#define THREAD_CONF(grid, block, gridBound, blockBound) do {\
	    block.x = blockBound;\
	    grid.x = gridBound; \
		if (grid.x > 65535) {\
		   grid.x = (int)sqrt((double)grid.x);\
		   grid.y = CEIL(gridBound, grid.x); \
		}\
	}while (0)

#define BLOCK_ID (gridDim.y * blockIdx.x + blockIdx.y)
#define THREAD_ID (threadIdx.x)
#define TID (BLOCK_ID * blockDim.x + THREAD_ID)

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError err, const char *file, const int line );



//used for unsorted values
typedef struct
{
   void *key;
   void *val;
   int keySize;
   int valSize;
} keyval_t;

typedef struct
{
   int *arr_len;
   int *arr_alloc_len;
   int **int_arr;

   keyval_t *arr;
} keyval_arr_t;

//used for sorted or partial sorted values
typedef struct
{
   void * val;
   int valSize;
} val_t;

typedef struct
{
   void * key;
   int keySize;
   int val_arr_len;
   val_t * vals;
} keyvals_t;
	
typedef struct 
{	
  int num_mapper;
  int num_reducers;
  
  //data for input results
  int h_num_input_record;
  keyval_t * h_input_keyval_arr;
  keyval_t * d_input_keyval_arr;

  //data for intermediate results
  int *d_intermediate_keyval_total_count;
  int *d_intermediate_keyval_arr_arr_len;
  keyval_arr_t *d_intermediate_keyval_arr_arr;
  keyval_arr_t *h_intermediate_keyval_arr_arr;
  keyval_t* d_intermediate_keyval_arr;

  //data for sorted intermediate results
  int d_sorted_keyvals_arr_len;
  int d_sorted_keyvals_arr_alloc_len;
  keyvals_t *d_sorted_keyvals_arr;
  keyvals_t *h_sorted_keyvals_arr;

  //data for reduce results
  keyval_t* d_reduced_keyval_arr;

} d_global_state;


//------------------------------------------------------
//PandaSort.cu
//------------------------------------------------------
typedef int4 cmp_type_t;


extern "C"
int sort_GPU (void * d_inputKeyArray, 
              int totalKeySize, 
              void * d_inputValArray, 
              int totalValueSize, 
              cmp_type_t * d_inputPointerArray, 
              int rLen, 
              void * d_outputKeyArray, 
              void * d_outputValArray, 
              cmp_type_t * d_outputPointerArray,
              int2 ** h_outputKeyListRange);

extern "C"
void sort_CPU(d_global_state *d_g_state);

extern "C"
void saven_initialPrefixSum(unsigned int maxNumElements);

//------------------------------------------------------
//PandaScan.cu
//------------------------------------------------------
extern "C"
void prescanArray(int *outArray, int *inArray, int numElements);

extern "C"
int prefexSum( int* d_inArr, int* d_outArr, int numRecords );


//-------------------------------------------------------
//PandaLib.cu
//-------------------------------------------------------

#define DEFAULT_DIMBLOCK	256
#define DEFAULT_NUMTASK		1

#define MAP_ONLY		0x01
#define MAP_GROUP		0x02
#define MAP_REDUCE		0x03




typedef struct
{
	//for input data on host
	char*		inputKeys;
	char*		inputVals;
	int4*		inputOffsetSizes;
	int		inputRecordCount;

	//for intermediate data on host
	char*		interKeys;
	char*		interVals;
	int4*		interOffsetSizes;
	int2*		interKeyListRange;
	int		interRecordCount;
	int		interDiffKeyCount;
	int		interAllKeySize;
	int		interAllValSize;

	//for output data on host
	char*		outputKeys;
	char*		outputVals;
	int4*		outputOffsetSizes;
	int2*		outputKeyListRange;
	int		outputRecordCount;
	int		outputAllKeySize;
	int		outputAllValSize;
	int		outputDiffKeyCount;

	//user specification
	char		workflow;
	char		outputToHost;

	int		numRecTaskMap;
	int		numRecTaskReduce;
	int		dimBlockMap;
	int		dimBlockReduce;
	//char* myKeys;

} Spec_t;



__device__ void EmitInterCount(int	keySize,
                               int	valSize,
                               int*	interKeysSizePerTask,
                               int*	interValsSizePerTask,
                               int*	interCountPerTask);

__device__ void EmitIntermediate2(void *key, 
								  void *val, 
								  int keySize, 
								  int valSize, 
								  d_global_state *d_g_state, 
								  int map_task_id);


__device__ void EmitIntermediate(void*		key, 
				 void*		val, 
				 int		keySize, 
				 int		valSize,
				 int*	psKeySizes,
				 int*	psValSizes,
				 int*	psCounts,
				 int2*		keyValOffsets,
				 char*		interKeys,
				 char*		interVals,
				 int4*		interOffsetSizes,
				 int*	curIndex, d_global_state d_g_state, int map_task_id);


__device__ void EmitCount(int		keySize,
			  int		valSize,
			  int*		outputKeysSizePerTask,
			  int*		outputValsSizePerTask,
			  int*		outputCountPerTask);

__device__ void Emit  (char*		key, 
                       char*		val, 
		       int		keySize, 
                       int		valSize,
		       int*		psKeySizes, 
		       int*		psValSizes, 
		       int*		psCounts, 
		       int2*		keyValOffsets, 
		       char*		outputKeys,
	               char*		outputVals,
	               int4*		outputOffsetSizes,
	               int*		curIndex);

__device__ void *GetVal(void *vals, int4* interOffsetSizes, int index, int valStartIndex);
__device__ void *GetKey(void *key, int4* interOffsetSizes, int index, int valStartIndex);

#define MAP_COUNT_FUNC \
	map_count(void*		key,\
		  void*		val,\
		  int	keySize,\
		  int	valSize,\
		  int*	interKeysSizePerTask,\
		  int*	interValsSizePerTask,\
		  int*	interCountPerTask)

#define EMIT_INTER_COUNT_FUNC(keySize, valSize)\
		EmitInterCount(keySize, valSize, \
		interKeysSizePerTask, interValsSizePerTask, interCountPerTask)



#define MAP_FUNC \
	 map	(void*		key, \
		 void*		val, \
		 int		keySize, \
		 int		valSize,\
		 int*	psKeySizes, \
		 int*	psValSizes, \
		 int*	psCounts, \
		 int2*		keyValOffsets, \
		 char*		interKeys,\
		 char*		interVals,\
		 int4*		interOffsetSizes,\
		 int*	curIndex,\
		 d_global_state d_g_state, int map_task_id)



#define REDUCE_COUNT_FUNC \
	reduce_count(void		*key,\
	         void		*vals,\
		 int		keySize,\
		 int		valCount,\
		 int4*		interOffsetSizes,\
		 int*	outputKeysSizePerTask,\
		 int*	outputValsSizePerTask,\
		 int*	outputCountPerTask)

#define EMIT_COUNT_FUNC(newKeySize, newValSize) \
	EmitCount(newKeySize,\
			  newValSize,\
			  outputKeysSizePerTask,\
			  outputValsSizePerTask,\
			  outputCountPerTask)

#define REDUCE_FUNC \
	reduce(void*	 key, \
		   void*	 vals, \
		   int	 keySize, \
		   int	 valCount, \
		   int*	 psKeySizes,\
		   int*	 psValSizes, \
		   int*	 psCounts, \
		   int2*	 keyValOffsets,\
		   int4*	 interOffsetSizes,\
		   char*	 outputKeys, \
		   char*	 outputVals,\
		   int4*	 outputOffsetSizes, \
		   int* curIndex,\
			int valStartIndex)

#define EMIT_FUNC(newKey, newVal, newKeySize, newValSize) \
	Emit((char*)newKey,\
	     (char*)newVal,\
		 newKeySize,\
		 newValSize,\
		 psKeySizes,\
		 psValSizes,\
		 psCounts, \
		 keyValOffsets, \
		 outputKeys,\
		 outputVals,\
		 outputOffsetSizes,\
		 curIndex)

extern __shared__ char sbuf[];
#define GET_OUTPUT_BUF(offset) (sbuf + threadIdx.x * 5 * sizeof(int) + offset)
#define GET_VAL_FUNC(vals, index) GetVal(vals, interOffsetSizes, index, valStartIndex)
#define GET_KEY_FUNC(key, index) GetKey(key, interOffsetSizes, index, valStartIndex)

extern "C"
Spec_t *GetDefaultSpec();

extern "C"
d_global_state *GetDGlobalState();

extern "C"
void AddMapInputRecord2(d_global_state*		spec, 
		   void*		key, 
		   void*		val, 
		   int		keySize, 
		   int		valSize);


extern "C"
void AddMapInputRecord(Spec_t*		spec, 
		   void*		key, 
		   void*		val, 
		   int		keySize, 
		   int		valSize);

extern "C"
void MapReduce(Spec_t *spec, d_global_state *d_g_state);

extern "C"
void FinishMapReduce(Spec_t* spec);

void MakeMapInput(Spec_t *spec,
		  char *fdata, 
		  int fsize,
		  void *(*make_routine)(void*), 
		  int threadNum,
		  void *other);

//-------------------------------------------------------
//MarsUtils.cu
//-------------------------------------------------------
typedef struct timeval TimeVal_t;

extern "C"
void startTimer(TimeVal_t *timer);

extern "C"
void endTimer(char *info, TimeVal_t *timer);

#ifdef _DEBUG
#define DoLog(...) do{printf(__VA_ARGS__);printf("\n");}while(0)
#else
#define DoLog(...) //do{printf(__VA_ARGS__);printf("\n");}while(0)
#endif

typedef void (*PrintFunc_t)(void* key, void* val, int keySize, int valSize);
void PrintOutputRecords(Spec_t* spec, int num, PrintFunc_t printFunc);

#endif //__PANDA_H__
