/********************************************************************
*  sample.cu
*  This is a example of the CUDA program.
*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <kmeans.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <kmeans_kernel.cu>

/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/
#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else



bool InitCUDA(void)
{
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
	cudaSetDevice(i);

	printf("CUDA initialized.\n");
	return true;
}

#endif

void kmeans_distance_cpu(const float* allClusters, const float* allEvents, int* cM);
float CalcDistCPU(const float* refVecs, const float* events, int eventIndex, int clusterIndex);
void generateEvents(float* allEvents);
bool UpdateCenters(const float* oldClusters, const float* events, int* cMs, float* newClusters);
int* AllocateCM(int* cMs);
float* AllocateClusters(float* clust);
float* AllocateEvents(float* evs);
void generateInitialClusters(float* clusters, float* events);

/************************************************************************/
/* HelloCUDA                                                            */
/************************************************************************/
int main(int argc, char* argv[])
{

	if(!InitCUDA()) {
		return 0;
	}
	CUT_DEVICE_INIT(argc, argv);
	srand((unsigned)(time(0)));
	float* myEvents = (float*)malloc(sizeof(float)*NUM_EVENTS*ALL_DIMENSIONS);
	generateEvents(myEvents);
	
	float* myClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS);
	generateInitialClusters(myClusters, myEvents);
	
	
	
	

	//const events allEvents;
	int* cM = (int*)malloc(sizeof(int)*NUM_EVENTS);
	bool updated;
	float* newClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS);

	float* d_E;// = AllocateEvents(myEvents);
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_E, sizeof(float)*NUM_EVENTS*ALL_DIMENSIONS));
		float* d_C;// = AllocateClusters(myClusters);
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_C, sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS));
		int* d_cM;// = AllocateCM(cM);
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_cM, sizeof(int)*NUM_EVENTS));
		int size = sizeof(float)*ALL_DIMENSIONS*NUM_EVENTS;
		CUDA_SAFE_CALL(cudaMemcpy(d_E, myEvents, size, cudaMemcpyHostToDevice));
		size = sizeof(float)*ALL_DIMENSIONS*NUM_CLUSTERS;
		CUDA_SAFE_CALL(cudaMemcpy(d_C, myClusters, size, cudaMemcpyHostToDevice));

	do{
#if CPU_ONLY

		clock_t cpu_start, cpu_stop;
		cpu_start = clock();

		kmeans_distance_cpu(myClusters, myEvents, cM);

		cpu_stop = clock();
		printf("Processing time for GPU: %f (ms) \n", (float)(cpu_stop - cpu_start)/(float)(CLOCKS_PER_SEC)*(float)1e3);
#else
		
		unsigned int timer = 0;
		CUT_SAFE_CALL(cutCreateTimer(&timer));
		CUT_SAFE_CALL(cutStartTimer(timer));

		size = sizeof(float)*ALL_DIMENSIONS*NUM_CLUSTERS;
		CUDA_SAFE_CALL(cudaMemcpy(d_C, myClusters, size, cudaMemcpyHostToDevice));
		

		kmeans_distance<<< NUM_BLOCKS, NUM_THREADS >>>(d_C, d_E, d_cM);

		CUDA_SAFE_CALL(cudaMemcpy(cM, d_cM, sizeof(int)*NUM_EVENTS, cudaMemcpyDeviceToHost));
		
		CUT_SAFE_CALL(cutStopTimer(timer));
		printf("Processing time for GPU: %f (ms) \n", cutGetTimerValue(timer));
		CUT_SAFE_CALL(cutDeleteTimer(timer));

#endif
		updated = UpdateCenters(myClusters, myEvents, cM, newClusters);
	
		
		for(int i=0; i < NUM_CLUSTERS; i++){
			
			for(int k = 0; k < ALL_DIMENSIONS; k++)
				myClusters[i*ALL_DIMENSIONS + k] = newClusters[i*ALL_DIMENSIONS + k];
			
		}
	} while(updated); 
		printf("\n");
		for(int i=0; i < NUM_CLUSTERS; i++){
			for(int k = 0; k < ALL_DIMENSIONS; k++)
				printf("%f\t", myClusters[i*ALL_DIMENSIONS + k]);
			printf("\n");
		}
	CUT_EXIT(argc, argv);
	return 0;
}

void generateEvents(float* allEvents){
	//generateEvents around (10,10,10), (20, 10, 50), and (50, 50, 0)
	int i, j;
	for(i = 0; i < NUM_EVENTS; i++){
		for(j =0; j < 3; j++){
				
		if(i < NUM_EVENTS/3){
			allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 7;
		}
		else if(i < NUM_EVENTS*2/3){
			switch(j){
			 	case 0: allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 17; break;
			 	case 1: allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 7; break;
				case 2: allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 47; break;
				default: printf("error!\n");
			}
		}
		else {
			switch(j){
			 	case 0: allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 47; break;
			 	case 1: allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 47; break;
				case 2: allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*3 ; break;
				default: printf("error!\n");
			}

		}
		}
	}

}

void kmeans_distance_cpu(const float* allClusters, const float* allEvents, int* cMs){
	int i,j, min_clust;
	float tmp, min;
	
	for(i = 0; i < NUM_EVENTS; i++){ 
		min = FLT_MAX;
		for(j = 0; j < NUM_CLUSTERS; j++){ 
			tmp = CalcDistCPU(allClusters, allEvents, i, j);
			
			if(tmp < min){
				min = tmp;
				min_clust = j;
			}
			
		}
		cMs[i] = min_clust;//min_clust;
		
	}



}

float CalcDistCPU(const float* refVecs, const float* events, int eventIndex, int clusterIndex){
	float sum = 0;
	int i;
	for(i = 0; i < ALL_DIMENSIONS; i++){
		float tmp = events[eventIndex*ALL_DIMENSIONS + i] - refVecs[clusterIndex*ALL_DIMENSIONS + i];
		sum += tmp*tmp;
	}
	return sqrt(sum);
}

bool UpdateCenters(const float* oldClusters, const float* events, int* cMs, float* newClusters){
	int* num = (int *)malloc(sizeof(int)*NUM_CLUSTERS);
	bool retVal = false;
	for(int i = 0; i < NUM_CLUSTERS; i++){
		num[i] = 0;
		for(int k = 0; k < ALL_DIMENSIONS; k++){
			newClusters[i*ALL_DIMENSIONS + k] = (float)0.0;
		}
	}
	for(int i = 0; i < NUM_EVENTS; i++){
		int tmp = cMs[i];	

		for(int k = 0; k < ALL_DIMENSIONS; k++){
			
			
			newClusters[tmp*ALL_DIMENSIONS + k] += events[i*ALL_DIMENSIONS + k];
			
			
		}
		num[tmp]++;
	}
	for(int i = 0; i < NUM_CLUSTERS; i++){
		for(int k = 0; k < ALL_DIMENSIONS; k++){
			if(num[i] != 0)
				newClusters[i*ALL_DIMENSIONS + k] /= (float)(num[i]*1.0);
			
			if(!retVal && (newClusters[i*ALL_DIMENSIONS + k] != oldClusters[i*ALL_DIMENSIONS + k]))
				retVal = true;
		}
	}
	return retVal;
}



void generateInitialClusters(float* clusters, float* events){
	int seed;
	for(int i = 0; i < NUM_CLUSTERS; i++){
		seed = rand() % NUM_EVENTS;
		for(int j = 0; j < ALL_DIMENSIONS; j++){
			clusters[i*ALL_DIMENSIONS + j] = events[seed*ALL_DIMENSIONS + j];
		}
	}
	
}