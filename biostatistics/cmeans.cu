/********************************************************************
*  sample.cu
*  This is a example of the CUDA program.
*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <cmeans.h>
#include <cmeanscu.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cmeans_kernel.cu>
#include "timers.h"


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



unsigned int timer_io; // Timer for I/O, such as reading FCS file and outputting result files
unsigned int timer_memcpy; // Timer for GPU <---> CPU memory copying
unsigned int timer_cpu; // Timer for processing on CPU
unsigned int timer_gpu; // Timer for kernels on the GPU
unsigned int timer_total; // Total time

/************************************************************************/
/* C-means Main                                                            */
/************************************************************************/
int main(int argc, char* argv[])
{
   
    CUT_SAFE_CALL(cutCreateTimer(&timer_io));
    CUT_SAFE_CALL(cutCreateTimer(&timer_memcpy));
    CUT_SAFE_CALL(cutCreateTimer(&timer_cpu));
    CUT_SAFE_CALL(cutCreateTimer(&timer_gpu));
    CUT_SAFE_CALL(cutCreateTimer(&timer_total));
    
    CUT_SAFE_CALL(cutStartTimer(timer_total));
    CUT_SAFE_CALL(cutStartTimer(timer_io));
    
    // [program name]  [data file]
    if(argc != 2){
        printf("Usage Error: must supply data file. e.g. programe_name @opt(flags) file.in\n");
        //char tmp45[8];
        //scanf(tmp45, "%s");
        return 1;
    }

    float* myEvents = ParseSampleInput(argv[1]);
#if FAKE
    free(myEvents);
    myEvents = generateEvents();
#endif
    if(myEvents == NULL){
        return 1;
    }
     
    printf("Parsed file\n");
    
    if(!InitCUDA()) {
        return 0;
    }
    
    CUT_DEVICE_INIT(argc, argv);
    srand((unsigned)(time(0)));
    
    
    float* myClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS);
    float* newClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS);
    
    CUT_SAFE_CALL(cutStopTimer(timer_io));
    CUT_SAFE_CALL(cutStartTimer(timer_cpu));
    
    clock_t total_start;
    total_start = clock();
    
    // Select random cluster centers
    generateInitialClusters(myClusters, myEvents);
    
    int iterations = 0;
    
    CUT_SAFE_CALL(cutStopTimer(timer_cpu));
    
#if !CPU_ONLY    
    CUT_SAFE_CALL(cutStartTimer(timer_memcpy));

    float* d_E;// = AllocateEvents(myEvents);
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_E, sizeof(float)*NUM_EVENTS*ALL_DIMENSIONS));
    float* d_C;// = AllocateClusters(myClusters);
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_C, sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS));
    float* d_nC;// = AllocateCM(cM);
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nC, sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS));
    int size = sizeof(float)*ALL_DIMENSIONS*NUM_EVENTS;
    CUDA_SAFE_CALL(cudaMemcpy(d_E, myEvents, size, cudaMemcpyHostToDevice));
    size = sizeof(float)*ALL_DIMENSIONS*NUM_CLUSTERS;
    CUDA_SAFE_CALL(cudaMemcpy(d_C, myClusters, size, cudaMemcpyHostToDevice));

    CUT_SAFE_CALL(cutStopTimer(timer_memcpy));
#endif
    float diff;
    clock_t cpu_start, cpu_stop;
    cpu_start = clock();
    printf("Starting C-means\n");
    float averageTime = 0;
    do{
#if CPU_ONLY
        CUT_SAFE_CALL(cutStartTimer(timer_cpu));
        clock_t cpu_start, cpu_stop;
        cpu_start = clock();

        UpdateClusterCentersCPU(myClusters, myEvents, newClusters);

        cpu_stop = clock();
        printf("Processing time for CPU: %f (ms) \n", (float)(cpu_stop - cpu_start)/(float)(CLOCKS_PER_SEC)*(float)1e3);
        averageTime += (float)(cpu_stop - cpu_start)/(float)(CLOCKS_PER_SEC)*(float)1e3;
        CUT_SAFE_CALL(cutStopTimer(timer_cpu));
#else
        
        unsigned int timer = 0;
        CUT_SAFE_CALL(cutCreateTimer(&timer));
        CUT_SAFE_CALL(cutStartTimer(timer));

        size = sizeof(float)*ALL_DIMENSIONS*NUM_CLUSTERS;

        CUT_SAFE_CALL(cutStartTimer(timer_memcpy));
        CUDA_SAFE_CALL(cudaMemcpy(d_C, myClusters, size, cudaMemcpyHostToDevice));
        CUT_SAFE_CALL(cutStopTimer(timer_memcpy));
        
        dim3 BLOCK_DIM(1, NUM_THREADS, 1);

        CUT_SAFE_CALL(cutStartTimer(timer_gpu));
        UpdateClusterCentersGPU<<< NUM_BLOCKS, BLOCK_DIM >>>(d_C, d_E, d_nC);
	cudaThreadSynchronize();
        CUT_SAFE_CALL(cutStopTimer(timer_gpu));

        CUT_SAFE_CALL(cutStartTimer(timer_memcpy));
        CUDA_SAFE_CALL(cudaMemcpy(newClusters, d_nC, sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS, cudaMemcpyDeviceToHost));
        CUT_SAFE_CALL(cutStopTimer(timer_memcpy));
        
        CUT_SAFE_CALL(cutStopTimer(timer));
        float thisTime = cutGetTimerValue(timer);
        printf("Processing time for GPU: %f (ms) \n", thisTime);
        averageTime += thisTime;
        CUT_SAFE_CALL(cutDeleteTimer(timer));

#endif

        CUT_SAFE_CALL(cutStartTimer(timer_cpu));
        
        diff = 0.0;
        for(int i=0; i < NUM_CLUSTERS; i++){
            
            for(int k = 0; k < ALL_DIMENSIONS; k++){
                diff += myClusters[i*ALL_DIMENSIONS + k] - newClusters[i*ALL_DIMENSIONS + k];
                myClusters[i*ALL_DIMENSIONS + k] = newClusters[i*ALL_DIMENSIONS + k];
            }
        }
        printf("Diff = %f\n", diff);

        iterations++;
        
        CUT_SAFE_CALL(cutStopTimer(timer_cpu));

    } while(abs(diff) > THRESHOLD && iterations < 150); 
    
    if(iterations == 150){
        printf("Warning: c-means did not converge to the %f threshold provided\n", THRESHOLD);
    }
    cpu_stop = clock();
    
    CUT_SAFE_CALL(cutStartTimer(timer_io));
    
    averageTime /= iterations;
    printf("\nTotal Processing time: %f (s) \n", (float)(cpu_stop - cpu_start)/(float)(CLOCKS_PER_SEC));
    printf("C-means complete\n");
    printf("\n");
    for(int i=0; i < NUM_CLUSTERS; i++){
        for(int k = 0; k < ALL_DIMENSIONS; k++)
            printf("%f\t", myClusters[i*ALL_DIMENSIONS + k]);
        printf("\n");
    }

    CUT_SAFE_CALL(cutStopTimer(timer_io));
    
    int* finalClusterConfig;
    float mdlTime = 0;
    
#if !MDL_on_GPU
    finalClusterConfig = MDL(myEvents, myClusters, &mdlTime, argv[1]);
#else
    finalClusterConfig = MDLGPU(d_E, d_nC, &mdlTime, argv[1]);
    mdlTime /= 1000.0; // CUDA timer returns time in milliseconds, normalize to seconds
#endif

    CUT_SAFE_CALL(cutStartTimer(timer_io));

    printf("Final Clusters are:\n");
    int newCount = 0;
    for(int i = 0; i < NUM_CLUSTERS; i++){
        if(finalClusterConfig[i]){
            for(int j = 0; j < ALL_DIMENSIONS; j++){
                newClusters[newCount * ALL_DIMENSIONS + j] = myClusters[i*ALL_DIMENSIONS + j];
                printf("%f\t", myClusters[i*ALL_DIMENSIONS + j]);
            }
            newCount++;
            printf("\n");
        }
    }
    
    CUT_SAFE_CALL(cutStopTimer(timer_io));

    FindCharacteristics(myEvents, newClusters, newCount, averageTime, mdlTime, iterations, argv[1], total_start);
    
    free(newClusters);
    free(myClusters);
    free(myEvents);
#if !CPU_ONLY
    CUDA_SAFE_CALL(cudaFree(d_E));
    CUDA_SAFE_CALL(cudaFree(d_C));
    CUDA_SAFE_CALL(cudaFree(d_nC));
#endif

    CUT_SAFE_CALL(cutStopTimer(timer_total));
    printf("\n\n"); 
    printf("Total Time (ms): %f\n.",cutGetTimerValue(timer_total));
    printf("I/O Time (ms): %f\n.",cutGetTimerValue(timer_io));
    printf("GPU memcpy Time (ms): %f\n.",cutGetTimerValue(timer_memcpy));
    printf("CPU processing Time (ms): %f\n.",cutGetTimerValue(timer_cpu));
    printf("GPU processing Time (ms): %f\n.",cutGetTimerValue(timer_gpu));
    printf("\n\n"); 
    
    //CUT_EXIT(argc, argv);
    printf("\n\n");
    return 0;
}

float* generateEvents(){
    float* allEvents = (float*) malloc(NUM_EVENTS*ALL_DIMENSIONS*sizeof(float));
    //generateEvents around (10,10,10), (20, 10, 50), and (50, 50, 0)
    int i, j;
    for(i = 0; i < NUM_EVENTS; i++){
        for(j =0; j < 3; j++){
                
        if(i < NUM_EVENTS/3){
            allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 7;
        }
        else if(i < NUM_EVENTS*2/3){
            switch(j){
                case 0: allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 47; break;
                case 1: allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 27; break;
                case 2: allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 7; break;
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
    return allEvents;
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



__host__ float CalculateDistanceCPU(const float* clusters, const float* events, int clusterIndex, int eventIndex){

    float sum = 0;
#if DISTANCE_MEASURE == 0
    for(int i = 0; i < ALL_DIMENSIONS; i++){
        float tmp = events[eventIndex*ALL_DIMENSIONS + i] - clusters[clusterIndex*ALL_DIMENSIONS + i];
        sum += tmp*tmp;
    }
    sum = sqrt(sum);
#endif
#if DISTANCE_MEASURE == 1
    for(int i = 0; i < ALL_DIMENSIONS; i++){
        float tmp = events[eventIndex*ALL_DIMENSIONS + i] - clusters[clusterIndex*ALL_DIMENSIONS + i];
        sum += abs(tmp);
    }
#endif
#if DISTANCE_MEASURE == 2
    for(int i = 0; i < ALL_DIMENSIONS; i++){
        float tmp = abs(events[eventIndex*ALL_DIMENSIONS + i] - clusters[clusterIndex*ALL_DIMENSIONS + i]);
        if(tmp > sum)
            sum = tmp;
    }
#endif
    return sum;
}


__host__ float MembershipValue(const float* clusters, const float* events, int clusterIndex, int eventIndex){
    float myClustDist = CalculateDistanceCPU(clusters, events, clusterIndex, eventIndex);
    float sum =0;
    float otherClustDist;
    for(int j = 0; j< NUM_CLUSTERS; j++){
        otherClustDist = CalculateDistanceCPU(clusters, events, j, eventIndex); 
        if(otherClustDist < .000001)
            return 0.0;
        sum += pow((float)(myClustDist/otherClustDist),float(2/(FUZZINESS-1)));
    }
    return 1/sum;
}



void UpdateClusterCentersCPU(const float* oldClusters, const float* events, float* newClusters){
    
    
    float membershipValue, denominator;
    float* numerator = (float*)malloc(sizeof(float)*ALL_DIMENSIONS);


    for(int i = 0; i < NUM_CLUSTERS; i++){
      denominator = 0.0;
      for(int j = 0; j < ALL_DIMENSIONS; j++)
        numerator[j] = 0;
      for(int j = 0; j < NUM_EVENTS; j++){
        membershipValue = MembershipValue(oldClusters, events, i, j);
        for(int k = 0; k < ALL_DIMENSIONS; k++){
          numerator[k] += events[j*ALL_DIMENSIONS + k]*membershipValue;
        }
        
        denominator += membershipValue;
      }  
      for(int j = 0; j < ALL_DIMENSIONS; j++){
          newClusters[i*ALL_DIMENSIONS + j] = numerator[j]/denominator;
      }  
    }
    
}




float* ParseSampleInput(const char* filename){
    FILE* myfile = fopen(filename, "r");
    if(myfile == NULL){
        printf("Error: File DNE\n");
        return NULL;
    }
    char myline[1024];
    
    float* retVal = (float*)malloc(sizeof(float)*NUM_EVENTS*ALL_DIMENSIONS);
    myfile = fopen(filename, "r");
#if !LINE_LABELS

    for(int i = 0; i < NUM_EVENTS; i++){
        fgets(myline, 1024, myfile);
        retVal[i*ALL_DIMENSIONS] = (float)atof(strtok(myline, " "));
        for(int j = 1; j < ALL_DIMENSIONS; j++){
            retVal[i*ALL_DIMENSIONS + j] = (float)atof(strtok(NULL, " "));
        }
    }
#else
    fgets(myline, 1024, myfile);
    for(int i = 0; i < NUM_EVENTS; i++){
        fgets(myline, 1024, myfile);
        strtok(myline, " ");
        for(int j = 0; j < ALL_DIMENSIONS; j++){
            retVal[i*ALL_DIMENSIONS + j] = (float)atof(strtok(NULL, " "));
        }
    }
#endif
    
    fclose(myfile);
    
    
    return retVal;
}

void FreeMatrix(float* d_matrix){
    CUDA_SAFE_CALL(cudaFree(d_matrix));
}

float* BuildQGPU(float* d_events, float* d_clusters, float* mdlTime){
    float* d_matrix;
    int size = sizeof(float) * NUM_CLUSTERS*NUM_CLUSTERS;

    unsigned int timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    CUT_SAFE_CALL(cutStartTimer(timer));
    CUT_SAFE_CALL(cutStartTimer(timer_memcpy));


    CUDA_SAFE_CALL(cudaMalloc((void**)&d_matrix, size));
    //CalculateQMatrixGPU<<<NUM_CLUSTERS,Q_THREADS>>>(d_events, d_clusters, d_matrix);

    CUT_SAFE_CALL(cutStopTimer(timer_memcpy));
    CUT_SAFE_CALL(cutStartTimer(timer_gpu));

    dim3 grid(NUM_CLUSTERS, NUM_CLUSTERS);

    CalculateQMatrixGPUUpgrade<<<grid, Q_THREADS>>>(d_events, d_clusters, d_matrix);
    cudaThreadSynchronize();
    
    CUT_SAFE_CALL(cutStopTimer(timer_gpu));
    

    CUT_SAFE_CALL(cutStartTimer(timer_memcpy));
    float* matrix = (float*)malloc(size);
    cudaError_t error = cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    CUT_SAFE_CALL(cutStopTimer(timer_memcpy));

    CUT_SAFE_CALL(cutStopTimer(timer));
    *mdlTime = cutGetTimerValue(timer);
    printf("Processing time for GPU: %f (ms) \n", *mdlTime);
    CUT_SAFE_CALL(cutDeleteTimer(timer));
        
    const char * message = cudaGetErrorString(error);
    printf("Error: %s\n", message);

    
    FreeMatrix(d_matrix);
    return matrix;
}

/*float FindScoreGPU(float* d_matrix, long config){
    float* d_score;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_score, sizeof(float)));
    EvaluateSolutionGPU<<<1, 1>>>(d_matrix, config, d_score);
    float* score = (float*)malloc(sizeof(float));
    CUDA_SAFE_CALL(cudaMemcpy(score, d_score, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_score));
    return *score;
}*/

