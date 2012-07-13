/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: PandaLib.cu 
	Versions:	2012-07-01 V0.1
				2012-07-09 V0.12

	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */

#ifndef __PANDALIB_CU__
#define __PANDALIB_CU__

#include "Panda.h"
#include "stdlib.h"
#include "map.cu"
#include "reduce.cu"


//----------------------------------------------
//Get default runtime configuration
//return: default spec
//----------------------------------------------

d_global_state *GetDGlobalState(){
	d_global_state *d_g_state = (d_global_state*)malloc(sizeof(d_global_state));
	if (d_g_state == NULL) exit(-1);
	memset(d_g_state, 0, sizeof(d_global_state));
	return d_g_state;
}//d_global_state


//--------------------------------------------------------
//Initiate map reduce spec
//--------------------------------------------------------

void InitMapReduce2(d_global_state* d_g_state)
{
	
	//init d_g_state
	//load input records from host memory to device memory. 
	cudaMalloc((void **)&d_g_state->d_input_keyval_arr,sizeof(keyval_t)*d_g_state->h_num_input_record);
	keyval_t* h_buff = (keyval_t*)malloc(sizeof(keyval_t)*(d_g_state->h_num_input_record));

	for(int i=0;i<d_g_state->h_num_input_record;i++){
		h_buff[i].keySize = d_g_state->h_input_keyval_arr[i].keySize;
		h_buff[i].valSize = d_g_state->h_input_keyval_arr[i].valSize;
		cudaMalloc((void **)&h_buff[i].key,h_buff[i].keySize);
		cudaMalloc((void **)&h_buff[i].val,h_buff[i].valSize);
		cudaMemcpy(h_buff[i].key,d_g_state->h_input_keyval_arr[i].key,h_buff[i].keySize,cudaMemcpyHostToDevice);
		cudaMemcpy(h_buff[i].val,d_g_state->h_input_keyval_arr[i].val,h_buff[i].valSize,cudaMemcpyHostToDevice);
	}//for
	cudaMemcpy(d_g_state->d_input_keyval_arr,h_buff,sizeof(keyval_t)*d_g_state->h_num_input_record,cudaMemcpyHostToDevice);
	//printData2<<<1,d_g_state->h_num_input_record>>>(*d_g_state);
	cudaThreadSynchronize(); 
}

/*
void InitMapReduce(Spec_t* spec)
{
	if (g_spec->dimBlockMap <= 0)
		g_spec->dimBlockMap = DEFAULT_DIMBLOCK;
	if (g_spec->dimBlockReduce <= 0)
		g_spec->dimBlockReduce = DEFAULT_DIMBLOCK;
	if (g_spec->numRecTaskReduce <= 0)
		g_spec->numRecTaskReduce = DEFAULT_NUMTASK;
	if (g_spec->numRecTaskMap <= 0)
		g_spec->numRecTaskMap = DEFAULT_NUMTASK;
	if (g_spec->workflow <= 0)
		g_spec->workflow = MAP_ONLY;
}*/

//--------------------------------------------------
//Add a map input record
//--------------------------------------------------

void AddMapInputRecord2(d_global_state* d_g_state,
						void*		key, 
						void*		val,
						int		keySize, 
						int		valSize){
	
	int len = d_g_state->h_num_input_record;
	if (len<0) return;
	d_g_state->h_input_keyval_arr = (keyval_t *)realloc(d_g_state->h_input_keyval_arr, sizeof(keyval_t)*(len+1));
	d_g_state->h_input_keyval_arr[len].keySize = keySize;
	d_g_state->h_input_keyval_arr[len].valSize = valSize;
	d_g_state->h_input_keyval_arr[len].key = malloc(keySize);
	d_g_state->h_input_keyval_arr[len].val = malloc(valSize);
	memcpy(d_g_state->h_input_keyval_arr[len].key,key,keySize);
	memcpy(d_g_state->h_input_keyval_arr[len].val,val,valSize);
	d_g_state->h_num_input_record++;
}

__device__ void Emit2  (void*		key, 
						void*		val, 
						int		keySize, 
						int		valSize,
						d_global_state *d_g_state){
						
			keyval_t *p = &(d_g_state->d_reduced_keyval_arr[TID]);
			p->keySize = keySize;
			p->key = malloc(keySize);
			memcpy(p->key,key,keySize);
			p->valSize = valSize;
			p->val = malloc(valSize);
			memcpy(p->val,val,valSize);
			printf("output: key:%s  val:%d\n",key,*(int *)val);

}//__device__ 


__device__ void EmitIntermediate2(void *key, void *val, int keySize, int valSize, d_global_state *d_g_state, int map_task_idx){
	
	
	
	keyval_arr_t *kv_arr_p = (keyval_arr_t *)&(d_g_state->d_intermediate_keyval_arr_arr[map_task_idx]);
	
	//printf("\tEmitInterMediate map task id[%d]  key:%s kv_arr_p->arr_len:%d\n",map_task_id,(char *)key, *(int *)(kv_arr_p->arr_len));
	//if there is not enough space to store intermediate key value pairs
	/*if ((kv_arr_p->arr_len)== *(kv_arr_p->arr_alloc_len)){
		*(kv_arr_p->arr_alloc_len) *= 2;
		//printf("\tincrease buffer for map task[%d] arr_len:%d\n", map_task_id, *(kv_arr_p->arr_alloc_len));
		char *p = (char *)kv_arr_p->arr;
		//kv_arr_p->arr = (keyval_t *)malloc(sizeof(keyval_t)*(*kv_arr_p->arr_alloc_len));
		for (int i=0;i<sizeof(keyval_t)*(*kv_arr_p->arr_alloc_len)/2;i++)
			//((char *)kv_arr_p->arr)[i] = p[i];
			memcpy(kv_arr_p->arr,p,(*kv_arr_p->arr_alloc_len)/2);
		//free(p);
	}*///if
	//__syncthreads();
	
	
	/*
	char *p = (char *)kv_arr_p->arr;
	kv_arr_p->arr = (keyval_t*)malloc(sizeof(keyval_t)*(kv_arr_p->arr_len+1));
	for (int i=0; i<sizeof(keyval_t)*(kv_arr_p->arr_len);i++)
			((char *)kv_arr_p->arr)[i] = p[i];
	*/

	
	keyval_t *p = (keyval_t *)kv_arr_p->arr;
	kv_arr_p->arr = (keyval_t*)malloc(sizeof(keyval_t)*(kv_arr_p->arr_len+1));
	for (int i=0; i<(kv_arr_p->arr_len);i++){
		kv_arr_p->arr[i].key = p[i].key;
		kv_arr_p->arr[i].keySize = p[i].keySize;
		kv_arr_p->arr[i].val = p[i].val;
		kv_arr_p->arr[i].valSize = p[i].valSize;
		//((char *)kv_arr_p->arr)[i] = p[i];
	}//for
	free(p);
	//TODO

	int current_map_output_index = (kv_arr_p->arr_len);
	keyval_t *kv_p = &(kv_arr_p->arr[current_map_output_index]);
	kv_p->key = (char *)malloc(sizeof(keySize));
	memcpy(kv_p->key,key,keySize);
	kv_p->keySize = keySize;
	
	kv_p->val = (char *)malloc(sizeof(valSize));
	memcpy(kv_p->val,val,valSize);
	kv_p->valSize = valSize;
	
	kv_arr_p->arr_len++;
	d_g_state->d_intermediate_keyval_total_count[map_task_idx] = kv_arr_p->arr_len;
	
	//int i=0;
	//if(TID>96&&TID<120)
	//for(int i=0;i<(kv_arr_p->arr_len);i++)
	
	//printf("EmitInterMediate2 TID[%d] map_task_id:%d, key:%s keyval_arr_len:%d\n",TID, map_task_idx, key, (kv_arr_p->arr_len));

	//if(TID>300&&TID<320)
	//for(int i=0;i<(kv_arr_p->arr_len);i++)
	//printf("EmitInterMediate2 [%d] map_task_id:%d,	key:%s  arr_len:%d\n",TID, map_task_id, kv_arr_p->arr[i].key,(kv_arr_p->arr_len));

	//printf("EmitIntermeidate2 kv_p->key:%s  kv_p->val:%d \n",kv_p->key,*((int *)kv_p->val));
	//__syncthreads();

}//__device__


//-------------------------------------------------
//called by user defined map function
//-------------------------------------------------



__global__ void Mapper2(
		   d_global_state d_g_state)
{	
	/*int index = TID;
	int bid = BLOCK_ID;
	int tid = THREAD_ID;*/

	
	int num_records_per_thread = (d_g_state.h_num_input_record+(gridDim.x*blockDim.x)-1)/(gridDim.x*blockDim.x);
	int block_start_idx = num_records_per_thread*blockIdx.x*blockDim.x;
	int thread_start_idx = block_start_idx 
		+ (threadIdx.x/STRIDE)*num_records_per_thread*STRIDE
		+ (threadIdx.x%STRIDE);
	int thread_end_idx = thread_start_idx+num_records_per_thread*STRIDE;

	//if (TID>=d_g_state.h_num_input_record)return;
	if(thread_end_idx>d_g_state.h_num_input_record)
		thread_end_idx = d_g_state.h_num_input_record;
	//printf("Mapper TID:%d, thread_start_idx:%d  thread_end_idx:%d totalThreads:%d\n",TID, thread_start_idx,thread_end_idx,gridDim.x*blockDim.x);
	

	for(int map_task_idx=thread_start_idx; map_task_idx < thread_end_idx; map_task_idx+=STRIDE){


		
		void *val = d_g_state.d_input_keyval_arr[map_task_idx].val;
		int valSize = d_g_state.d_input_keyval_arr[map_task_idx].valSize;

		void *key = d_g_state.d_input_keyval_arr[map_task_idx].key;
		int keySize = d_g_state.d_input_keyval_arr[map_task_idx].keySize;
	
		/////////////////////////////////////////////
		map2(key, val, keySize, valSize, &d_g_state, map_task_idx);
		/////////////////////////////////////////////


		//keyval_arr_t *kv_arr_p = (keyval_arr_t *)&(d_g_state.d_intermediate_keyval_arr_arr[map_task_idx]);
		
		//printf("\tmap_task_idx:%d  reduce_arrr_len:%d",map_task_idx,kv_arr_p->arr_len);
	}//for

	//int map_task_id = TID;
	//Note:
	//cindex is the map task id used in the intermediate record list;
	//coalecent input data access; i+=blockDim.x

}//__global__


//--------------------------------------------------
//mapper
//map tasks assignment happens here
//7/1/2012
//--------------------------------------------------
__global__ void Mapper(char*	inputKeys,
		   char*	inputVals,
		   int4*	inputOffsetSizes,
		   int*	psKeySizes,
		   int*	psValSizes,
		   int*	psCounts,
		   int2*	keyValOffsets,
		   char*	interKeys,
		   char*	interVals,
		   int4*	interOffsetSizes,
		   int*	curIndex,
		   int	recordNum, 
		   int	recordsPerTask,
		   int	taskNum,
		   d_global_state d_g_state)
{
/*	
	int index = TID;
	int bid = BLOCK_ID;
	int tid = THREAD_ID;
	
	if (index*recordsPerTask >= recordNum) return;
	int recordBase = bid * recordsPerTask * blockDim.x;
	int terminate = (bid + 1) * (recordsPerTask * blockDim.x);
	if (terminate > recordNum) terminate = recordNum;
	
	int l_psCounts = psCounts[index];
	int4 l_interOffsetSizes = interOffsetSizes[l_psCounts];
	l_interOffsetSizes.x = psKeySizes[index];
	l_interOffsetSizes.z = psValSizes[index];
	interOffsetSizes[l_psCounts] = l_interOffsetSizes;
	
	for (int i = recordBase + tid; i < terminate; i+=blockDim.x)
	{
		int cindex =  i;
		
		int4 offsetSize = inputOffsetSizes[cindex];
		char *key = inputKeys + offsetSize.x;
		char *val = inputVals + offsetSize.z;
		printf("recordBase:%d, tid:%d, terminate:%d, blockDim.x:%d  map_task_id:%d  i:%d\n",recordBase,tid,terminate,blockDim.x,cindex,i);
		
		//__syncthreads();

		map(key,
		val,
		offsetSize.y,
		offsetSize.w,
		psKeySizes,
		psValSizes,
		psCounts,
		keyValOffsets,
		interKeys,
		interVals,
		interOffsetSizes,
		curIndex,
		d_g_state,
		cindex);
	}	
*/	
}


int startMap2(d_global_state *d_g_state)
{
	//cudaHostAlloc((void**)&int_array_host, 10 * sizeof(int), cudaHostAllocMapped);
	//int_array_host = (int*)malloc(10*sizeof(int));
	//int_array_host = (int *)cudaHostAlloc(10*sizeof(int));
	//allocDeviceMemory<<<1, 1>>>();
    //cudaDeviceSynchronize();
    //printDeviceMemory<<<1,1>>>(int_array_host);
	
	//-------------------------------------------------------
	//0, Check status of d_g_state;
	//-------------------------------------------------------

	DoLog("startMap2		Check status of d_g_state");
	if (d_g_state->h_input_keyval_arr == NULL) { DoLog("Error: no any input keys"); exit(0);}
	
	//-------------------------------------------------------
	//1, upload map input data from host to device memory
	//-------------------------------------------------------
	DoLog("startMap2		upload map input data from host to device memory");
	keyval_arr_t *h_keyval_arr_arr = (keyval_arr_t *)malloc(sizeof(keyval_arr_t)*d_g_state->h_num_input_record);
	

	//d_g_state->d_intermediate_keyval_arr_arr = h_keyval_arr_arr;
	keyval_arr_t *d_keyval_arr_arr;
	(cudaMalloc((void**)&(d_keyval_arr_arr),d_g_state->h_num_input_record*sizeof(keyval_arr_t)));
	
	for (int i=0; i<d_g_state->h_num_input_record;i++){
		//keyval_t *d_keyval;
		//(cudaMalloc((void**)&(d_keyval),sizeof(keyval_t)));
		//h_keyval_arr_arr[i].arr = d_keyval;
		h_keyval_arr_arr[i].arr = NULL;
		//(cudaMalloc((void**)&(d_arr_len),sizeof(int)));
		//(cudaMemcpy(d_arr_len, &h_arr_len, sizeof(int), cudaMemcpyHostToDevice));
		h_keyval_arr_arr[i].arr_len = 0;
	}//for

	checkCudaErrors(cudaMemcpy(d_keyval_arr_arr, h_keyval_arr_arr, sizeof(keyval_arr_t)*d_g_state->h_num_input_record,cudaMemcpyHostToDevice));
	//(*d_g_state).intermediate_keyval_arr_arr_len = spec->inputRecordCount;
	d_g_state->d_intermediate_keyval_arr_arr = d_keyval_arr_arr;

	int *count = NULL;
	checkCudaErrors(cudaMalloc((void**)&(count),d_g_state->h_num_input_record*sizeof(int)));
	d_g_state->d_intermediate_keyval_total_count = count;
	cudaMemset(d_g_state->d_intermediate_keyval_total_count,0,d_g_state->h_num_input_record*sizeof(int));
	
	
	//----------------------------------------------
	//3, determine the number of threads to run
	//----------------------------------------------
	DoLog("startMap2		determine the number of threads to run");
	int num_threads = d_g_state->h_num_input_record;


	//--------------------------------------------------
	//4, start map
	//--------------------------------------------------
	
	/*dim3 h_dimBlock(512,1,1);
    dim3 h_dimGrid(4,1,1);
	dim3 h_dimThread(1,1,1);
	int sizeSmem = 128;*/
		
	

	Mapper2<<<NUM_BLOCKS,NUM_THREADS>>>(*d_g_state);
	cudaThreadSynchronize();
	return 0;
}//int 


//--------------------------------------------------
//start map
//
//1, get map input data on host
//2, upload map input data to device memory
//	 (keys, vals, keyOffsets, valOffsets, keySizes, valSizes)
//3, determine the number of threads to run 
//4, calculate intermediate data keys'buf size 
//	 and values' buf size
//5, do prefix sum on--
//	 i)		d_interKeysSizePerTask
//	 ii)	d_interValsSizePerTask
//	 iii)	d_interCountPerTask
//6, allocate intermediate memory on device memory
//7, start map
//8, free allocated memory
//--------------------------------------------------


int startMap(Spec_t* spec, d_global_state *d_g_state)
{
	    
	Spec_t* g_spec = spec;

	if (g_spec->inputKeys == NULL) { DoLog("Error: no any input keys"); exit(0);}
	if (g_spec->inputVals == NULL) { DoLog("Error: no any input values"); exit(0); }
	if (g_spec->inputOffsetSizes == NULL) { DoLog( "Error: no any input pointer info"); exit(0); }
	if (g_spec->inputRecordCount == 0) {DoLog( "Error: invalid input record count"); exit(0);}
	
	//-------------------------------------------------------
	//1, get map input data on host
	//-------------------------------------------------------
	
}//return 0;


void startGroup2(d_global_state*state){

	d_global_state* d_g_state = state;
	DoLog("===startGroup===");
	sort_CPU3(d_g_state);
	
}


void startGroup(Spec_t* spec, d_global_state *state)
{
	Spec_t* g_spec = spec;
}


//--------------------------------------------------------
//get a value from value list of the same key
//
//param	: vals
//param	: interOffsetSizes
//param	: index
//return: the wanted value
//--------------------------------------------------------

__device__ void *GetVal(void *vals, int4* interOffsetSizes, int keyIndex, int valStartIndex)
{
	int4 offset = interOffsetSizes[valStartIndex];
	return (void*)((char*)vals + keyIndex * offset.w);

}

__device__ void *GetKey(void *key, int4* interOffsetSizes, int keyIndex, int valStartIndex)
{
	int4 offset = interOffsetSizes[valStartIndex];
	return (void*)((char*)key + keyIndex * offset.y);
}//


//-------------------------------------------------------
//Reducer
//
//-------------------------------------------------------

__global__ void Reducer2(d_global_state d_g_state)
{

	int num_records_per_thread = (d_g_state.d_sorted_keyvals_arr_len+(gridDim.x*blockDim.x)-1)/(gridDim.x*blockDim.x);

	int block_start_idx = num_records_per_thread*blockIdx.x*blockDim.x;
	int thread_start_idx = block_start_idx 
		+ (threadIdx.x/STRIDE)*num_records_per_thread*STRIDE
		+ (threadIdx.x%STRIDE);

	int thread_end_idx = thread_start_idx+num_records_per_thread*STRIDE;
	
	//if (TID>=d_g_state.h_num_input_record)return;
	if(thread_end_idx>d_g_state.d_sorted_keyvals_arr_len)
		thread_end_idx = d_g_state.d_sorted_keyvals_arr_len;

	//printf("reducer2: TID:%d  start_idx:%d  end_idx:%d d_sorted_keyvals_arr_len:%d\n",TID,thread_start_idx,thread_end_idx,d_g_state.d_sorted_keyvals_arr_len);

	int start, end;
	for(int reduce_task_idx=thread_start_idx; reduce_task_idx < thread_end_idx; reduce_task_idx+=STRIDE){
		if (reduce_task_idx==0)
			start = 0;
		else
			start = d_g_state.d_pos_arr_4_sorted_keyval_pos_arr[reduce_task_idx-1];
		end = d_g_state.d_pos_arr_4_sorted_keyval_pos_arr[reduce_task_idx];

		val_t *val_t_arr = (val_t*)malloc(sizeof(val_t)*(end-start));

		int keySize = d_g_state.d_keyval_pos_arr[start].keySize;
		int keyPos = d_g_state.d_keyval_pos_arr[start].keyPos;
		void *key = (char*)d_g_state.d_intermediate_keys_shared_buff+keyPos;
		//printf("reduce_task_idx:%d		keyPos:%d,  keySize:%d, key:%s start:%d end:%d\n",reduce_task_idx,keyPos,keySize,key,start,end);

		
		for (int index = start;index<end;index++){
			int valSize = d_g_state.d_keyval_pos_arr[index].valSize;
			int valPos = d_g_state.d_keyval_pos_arr[index].valPos;
			//printf("reduce_task_idx:%d		valSize:%d  valPos:%d\n",reduce_task_idx,valSize,valPos);
			val_t_arr[index-start].valSize = valSize;
			val_t_arr[index-start].val = (char*)d_g_state.d_intermediate_vals_shared_buff + valPos;
			//printf("reduce_task_idx:%d		key:%s val:%d\n",reduce_task_idx,key, *(int*)val_t_arr[index-start].val);
		}
		reduce2(key, val_t_arr, keySize, end-start, d_g_state);
	}//for

	//int map_task_id = TID;
	//if (map_task_id>=d_g_state.d_sorted_keyvals_arr_len) return;
	//invoke user implemeted reduce function
	//run the assigned the reduce tasks. 

	
}




//----------------------------------------------
//start reduce
//
//1, if there is not a reduce phase, just return
//   then user uses spec->interKeys/spec->intervals 
//   for further processing
//2, get reduce input data on host
//3, upload reduce input data onto device memory
//4, determine the number of threads to run
//5, calculate output data keys'buf size 
//	 and values' buf size
//6, do prefix sum on--
//	 i)		d_outputKeysSizePerTask
//	 ii)	d_outputValsSizePerTask
//	 iii)	d_outputCountPerTask
//7, allocate output memory on device memory
//8, start reduce
//9, copy output data to Spect_t structure
//10,free allocated memory
//----------------------------------------------
		
	
void startReduce2(d_global_state *d_g_state)
{	
	cudaThreadSynchronize(); 
	
	d_g_state->d_reduced_keyval_arr_len = d_g_state->d_sorted_keyvals_arr_len;
		
	cudaMalloc((void **)&(d_g_state->d_reduced_keyval_arr), sizeof(keyval_t)*d_g_state->d_reduced_keyval_arr_len);
	printf("Start Reducer2:   Keyval_arr_len %d\n",d_g_state->d_sorted_keyvals_arr_len);
	cudaThreadSynchronize(); 
	
	Reducer2<<<NUM_BLOCKS,NUM_THREADS>>>(*d_g_state);
	printf("Reducer2   DONE\n");
	cudaThreadSynchronize(); 
}//void

void startReduce(Spec_t* spec)
{

	Spec_t* g_spec = spec;

	if (g_spec->interKeys == NULL) {DoLog( "Error: no any intermediate keys"); exit(0);}
	if (g_spec->interVals == NULL) {DoLog( "Error: no any intermediate values"); exit(0);}
	if (g_spec->interOffsetSizes == NULL) {DoLog( "Error: no any intermediate pointer info");exit(0);}
	if (g_spec->interRecordCount == 0) {DoLog( "Error: invalid intermediate record count");exit(0);}
	if (g_spec->interKeyListRange == NULL) { DoLog( "Error: no any key list range");exit(0);}
	if (g_spec->interDiffKeyCount == 0) { DoLog( "Error: invalid intermediate diff key count");exit(0);}
	
}//void

void MapReduce2(d_global_state *state){

	d_global_state* d_g_state = state;
	
	//-------------------------------------------
	//1, init device
	//-------------------------------------------
	DoLog( "\t\t-----------init GPU-----------");
	//CUT_DEVICE_INIT();

	DoLog("\t\t------------init mapreduce-----");
	InitMapReduce2(d_g_state);

	//-------------------------------------------
	//2, start map
	//-------------------------------------------
	DoLog( "\t\t----------start map-----------" );

	startMap2(d_g_state);
	
	//-------------------------------------------
	//3, start group
	//-------------------------------------------
	startGroup2(d_g_state);

	//-------------------------------------------
	//4, start reduce
	//-------------------------------------------
	DoLog( "\t\t----------start reduce--------");
	//TimeVal_t reduceTimer;
	//startTimer(&reduceTimer);
	
	startReduce2(d_g_state);
	cudaThreadSynchronize(); 
	//endTimer("Reduce", &reduceTimer);

EXIT_MR:
	

}


//----------------------------------------------
//start main map reduce procedure
//1, init device
//2, start map
//3, start reduce
//
//param : spec
//----------------------------------------------

void MapReduce(Spec_t *spec, d_global_state *state)
{
	assert(NULL != spec);
	Spec_t* g_spec = spec;
	d_global_state* d_g_state = state;
}

//------------------------------------------
//the last step
//
//1, free global variables' memory
//2, close log file's file pointer
//------------------------------------------

void FinishMapReduce(Spec_t* spec)
{
	DoLog( "=====finish map/reduce=====");
}//void

void FinishMapReduce2(d_global_state* state)
{
	DoLog( "=====finish map/reduce=====");

}//void


#endif //__PANDALIB_CU__