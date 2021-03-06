#ifndef _CMEANS_H_
#define _CMEANS_H_

#include <time.h>

// CPU vs GPU
#define ENABLE_MDL 0
#define CPU_ONLY 0

// Which GPU device to use
#define DEVICE 0

// number of clusters
#define NUM_CLUSTERS 2

// number of dimensions
#define NUM_DIMENSIONS 4

// number of elements
#define NUM_EVENTS 16

// input file delimiter (normally " " or "," or "\t")
#define DELIMITER ","
#define LINE_LABELS 0

// If 1, uses random events as initial cluster centers
// If 0, uses uniformly distributed events as initial cluster centers
#define RANDOM_SEED 1

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

// Enables optimized O(M*N) membership kernel
// Standard O(M^2*N) if 0
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

// number of Threads and blocks
#define Q_THREADS 192 // number of threads per block building Q
#define NUM_THREADS $NUM_THREADS$  // number of threads per block
#define NUM_THREADS_DISTANCE 512
#define NUM_THREADS_MEMBERSHIP 512
#define NUM_THREADS_UPDATE 256
#define NUM_CLUSTERS_PER_BLOCK 4
#define NUM_BLOCKS NUM_CLUSTERS
#define NUM_NUM NUM_THREADS
#define PI (3.1415926)

// Amount of loop unrolling for the distance and membership calculations accross dimensions
#define UNROLL_FACTOR 1

// function definitions

void kmeans_distance_cpu(const float* allClusters, const float* allEvents, int* cM);
float CalcDistCPU(const float* refVecs, const float* events, int eventIndex, int clusterIndex);
float* generateEvents();
bool UpdateCenters(const float* oldClusters, const float* events, int* cMs, float* newClusters);
int* AllocateCM(int* cMs);
float* AllocateClusters(float* clust);
float* AllocateEvents(float* evs);
void generateInitialClusters(float* clusters, float* events);

float CalculateDistanceCPU(const float* clusters, const float* events, int clusterIndex, int eventIndex);
float MembershipValue(const float* clusters, const float* events, int clusterIndex, int eventIndex);
float MembershipValueDist(const float* clusters, const float* events, int eventIndex, float distance);
float MembershipValueReduced(const float* clusters, const float* events, int clusterIndex, int eventIndex, int);
void UpdateClusterCentersCPU_Naive(const float* oldClusters, const float* events, float* newClusters);
void UpdateClusterCentersCPU_Optimized(const float* oldClusters, const float* events, float* newClusters);
void UpdateClusterCentersCPU_Linear(const float* oldClusters, const float* events, float* newClusters);

float* ParseSampleInput(char* filename);

float FindScoreGPU(float* d_matrix, long config);
float* BuildQGPU(float* d_events, float* d_clusters, float* distanceMatrix, float* mdlTime);
long TabuSearchGPU(float* d_matrix);
void FreeMatrix(float* d_matrix);
int bitCount (int* n);

void FindSphereCharacteristics(float* events, float* clusters, int finalClusterCount, float* volume, float* density, float* occupancy, int includeIndex);
void FindBoxCharacteristics(float* events, float* clusters, int finalClusterCount, float* volume, float* density, float* occupancy, int includeIndex);

void FindCharacteristics(float* events, float* clusters, int finalClusterCount, char* inFileName);
void ReportResults(float* events, float* clusters, int count, char* inFileName);
void ReportSummary(float* clusters, int count, char* inFileName);


#endif
