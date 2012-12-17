/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.3
	File: Global.h 
	Time: 2012-12-09 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */

#ifndef __USER_H__
#define __USER_H__


#include "Panda.h"

#define MATRIX_BLOCK_SIZE 64

#define ENABLE_MDL 0
#define CPU_ONLY 0
#define DEVICE 0

// number of clusters
#define NUM_CLUSTERS 5

// number of dimensions
#define NUM_DIMENSIONS 54

// number of elements
#define NUM_EVENTS 85399

// input file delimiter (normally " " or "," or "\t")
#define DELIMITER ",\t "
#define LINE_LABELS 0

// Parameters
#define FUZZINESS 2
#define FUZZINESS_SQUARE 1
#define THRESHOLD 0.0001
#define K1 1.0
#define K2 0.01
#define K3 1.5
#define MEMBER_THRESH 0.05
#define TABU_ITER 100
#define TABU_TENURE 5
#define VOLUME_TYPE $VOLUME_TYPE$
#define DISTANCE_MEASURE 0
#define MIN_ITERS 0
#define MAX_ITERS 20
#define RANDOM_SEED 1

#define Q_THREADS 192 // number of threads per block building Q
//#define NUM_THREADS $NUM_THREADS$  // number of threads per block
#define NUM_THREADS_DISTANCE 256
#define NUM_THREADS_MEMBERSHIP 512
#define NUM_THREADS_UPDATE 256
//#define NUM_BLOCKS NUM_CLUSTERS
#define NUM_NUM NUM_THREADS
#define PI (3.1415926)

// Number of cluster memberships computed by each thread in UpdateCenters
#define NUM_CLUSTERS_PER_BLOCK 4

__device__ void gpu_core_map(void *KEY, void*VAL, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx);

__device__ void gpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, gpu_context *d_g_state, int map_task_idx);

__device__ void gpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, gpu_context d_g_state);

__device__ int gpu_compare(const void *d_a, int len_a, const void *d_b, int len_b);


void cpu_map(void *KEY, void*VAL, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx);

void cpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, cpu_context* d_g_state);

int cpu_compare(const void *d_a, int len_a, const void *d_b, int len_b);

void cpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, cpu_context *d_g_state, int map_task_idx);

void cmeans_cpu_map_cpp(void *key, void *val, int keySize, int valSize);
void cmeans_cpu_reduce_cpp(void *key, val_t* vals, int keySize, int valCount);


typedef struct
{
        //int point_id;
		int local_map_id;
        int dim;
        int K;
        
        int start;
        int end;
        int global_map_id;

		//int* ptrClusterId;

} KM_KEY_T;

typedef struct
{
        //int* ptrPoints;
        //int* ptrClusters;
        //int* ptrChange;

        float *d_tempClusters;
        float *d_tempDenominators;
        float *d_Clusters;
        float *d_Points;
		float* d_distanceMatrix;


} KM_VAL_T;


#endif
