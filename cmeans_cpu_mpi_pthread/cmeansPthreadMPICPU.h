//###############################################################################
// head file  for Pthreads implementation of cmeans using multi-core on Delta
// lihui@indiana.edu   last update 6/11/2012
//###############################################################################

#ifndef _CMEANS_H_
#define _CMEANS_H_

#include <time.h>
#define ENABLE_MDL 0

// number of clusters
#define NUM_CLUSTERS 10

// number of dimensions
#define NUM_DIMENSIONS 100

// number of elements
// #define NUM_EVENTS 16

// input file delimiter (normally " " or "," or "\t")
#define DELIMITER ","
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
#define MAX_ITERS 100
#define RANDOM_SEED 1

// Naive O(M^2) or optimized O(M) version of membership computation
#define LINEAR 1

// Prints verbose output during the algorithm, enables DEBUG macro
#define ENABLE_DEBUG 1

// Used to enable regular print outs (such as the Rissanen scores, clustering results)
// This should be enabled for general use and disabled for performance evaluations
#define ENABLE_PRINT 1

// Used to enable output of cluster results to .results and .summary files
#define ENABLE_OUTPUT 0

#if ENABLE_DEBUG
#define DEBUG(fmt,args...) printf(fmt, ##args)
#else
#define DEBUG(fmt,args...)
#endif

#if ENABLE_PRINT
#define PRINT(fmt,args...) printf(fmt, ##args)
#else
#define PRINT(fmt,args...)
#endif

#define Q_THREADS 192 // number of threads per block building Q
#define PI (3.1415926)


// Amount of loop unrolling for the distance and membership calculations accross dimensions
#define UNROLL_FACTOR 1

// function definitions

void generateInitialClusters(float* clusters, float* events);

float MembershipValue(const float* clusters, const float* events, int clusterIndex, int eventIndex);
float MembershipValueDist(const float* clusters, const float* events, int eventIndex, float distance);
float MembershipValueReduced(const float* clusters, const float* events, int clusterIndex, int eventIndex, int);
void* doUpdateClusterCentersCPU_Linear(void *ptr);
float* ParseSampleInput(char* filename);

float FindScoreGPU(float* d_matrix, long config);
float* BuildQGPU(float* d_events, float* d_clusters, float* distanceMatrix, float* mdlTime);
long TabuSearchGPU(float* d_matrix);
void FreeMatrix(float* d_matrix);
int bitCount (int* n);

void ReportResults(float* events, float* clusters, int count, char* inFileName);
void ReportSummary(float* clusters, int count, char* inFileName);

#endif
