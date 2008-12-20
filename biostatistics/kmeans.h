#ifndef _KMEANS_H_
#define _KMEANS_H_


#define NUM_CLUSTERS 3 // number of clusters
#define ALL_DIMENSIONS 3 // number of dimensions
#define NUM_THREADS 384 // number of threads per block
#define NUM_BLOCKS 16
#define NUM_EVENTS 19660800//4915200 // number of elements
#define CPU_ONLY 0
#define FLOATS_PER_BLOCK 4000
#define MAX_DIMENSIONS FLOATS_PER_BLOCK/NUM_THREADS
#define FUZZINESS 2

//typedef float	clusters		[NUM_CLUSTERS][NUM_DIMENSIONS] ;
//typedef float	events			[NUM_EVENTS][NUM_DIMENSIONS] ;
//typedef float	currentEvents	[NUM_THREADS][NUM_DIMENSIONS]; // this can't stay the way it is in the general case => should be NUM_DIM/C = RAM SIZE
//typedef int		clusterMembers	[NUM_EVENTS];


#endif
