#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <cmeansMultiGPU.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cmeansMultiGPU_kernel.cu>
#include "MDL.h"

void printCudaError() {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("%s\n",cudaGetErrorString(error));
    }
}

typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
    float* et;
} cudaTimer_t;

void createTimer(cudaTimer_t* timer) {
    #pragma omp critical (create_timer) 
    {
        cudaEventCreate(&(timer->start));
        cudaEventCreate(&(timer->stop));
        timer->et = (float*) malloc(sizeof(float));
        *(timer->et) = 0.0f;
    }
}

void deleteTimer(cudaTimer_t timer) {
    #pragma omp critical (delete_timer) 
    {
        cudaEventDestroy(timer.start);
        cudaEventDestroy(timer.stop);
        free(timer.et);
    }
}

void startTimer(cudaTimer_t timer) {
    cudaEventRecord(timer.start,0);
}

void stopTimer(cudaTimer_t timer) {
    cudaEventRecord(timer.stop,0);
    cudaEventSynchronize(timer.stop);
    float tmp;
    cudaEventElapsedTime(&tmp,timer.start,timer.stop);
    *(timer.et) += tmp;
}

float getTimerValue(cudaTimer_t timer) {
    return *(timer.et);
}

/************************************************************************/
/* C-means Main                                                            */
/************************************************************************/
int main(int argc, char* argv[])
{
    unsigned int timer_io; // Timer for I/O, such as reading FCS file and outputting result files
    unsigned int timer_total; // Total time
   
    cutCreateTimer(&timer_io);
    cutCreateTimer(&timer_total);
    
    cutStartTimer(timer_total);
    cutStartTimer(timer_io);
    
    // [program name]  [data file]
    if(argc != 2){
        printf("Usage Error: must supply data file. e.g. programe_name @opt(flags) file.in\n");
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
    
    int num_gpus = 0;       // number of CUDA GPUs

    // determine the number of CUDA capable GPUs
    cudaGetDeviceCount(&num_gpus);
    if(num_gpus < 1)
    {
        printf("no CUDA capable devices were detected\n");
        return 1;
    }

    // display CPU and GPU configuration
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", num_gpus);
    for(int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s\n", i, dprop.name);
    }
    printf("---------------------------\n");
    
    //srand((unsigned)(time(0)));
    srand(42);
    
    cutStopTimer(timer_io);
    
    // Allocate arrays for the cluster centers
    float* myClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);
    float* newClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);

    // Select random cluster centers
    generateInitialClusters(myClusters, myEvents);

    // Create an array of arrays for temporary cluster centers from each GPU
    float** tempClusters = (float**) malloc(sizeof(float*)*num_gpus);
    float** tempDenominators = (float**) malloc(sizeof(float*)*num_gpus);
    for(int i=0; i < num_gpus; i++) {
        tempClusters[i] = (float*) malloc(sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);
        tempDenominators[i] = (float*) malloc(sizeof(float)*NUM_CLUSTERS);
        memcpy(tempClusters[i],myClusters,sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);
    }
    // Create an array of arrays for temporary Q matrix pieces from each GPU
    float** q_matrices = (float**) malloc(sizeof(float*)*num_gpus);
    // Create an array for the final Q matrix
    float* q_matrix = (float*) malloc(sizeof(float)*NUM_CLUSTERS*NUM_CLUSTERS);
    
    float diff; // used to track difference in cluster centers between iterations

    // Transpose the events matrix
    float* transposedEvents = (float*)malloc(sizeof(float)*NUM_EVENTS*NUM_DIMENSIONS);
    for(int i=0; i<NUM_EVENTS; i++) {
        for(int j=0; j<NUM_DIMENSIONS; j++) {
            transposedEvents[j*NUM_EVENTS+i] = myEvents[i*NUM_DIMENSIONS+j];
        }
    }

    float* memberships = (float*) malloc(sizeof(float)*NUM_CLUSTERS*NUM_EVENTS);
    int* finalClusterConfig;
   
    ////////////////////////////////////////////////////////////////
    // run as many CPU threads as there are CUDA devices
    //   each CPU thread controls a different device, processing its
    //   portion of the data.  It's possible to use more CPU threads
    //   than there are CUDA devices, in which case several CPU
    //   threads will be allocating resources and launching kernels
    //   on the same device.  For example, try omp_set_num_threads(2*num_gpus);
    //   Recall that all variables declared inside an "omp parallel" scope are
    //   local to each CPU thread
    //
    //num_gpus = 1;
    omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices
    //omp_set_num_threads(2*num_gpus);// create twice as many CPU threads as there are CUDA devices
    #pragma omp parallel shared(myClusters,diff,tempClusters,tempDenominators,memberships,finalClusterConfig)
    {
        cudaTimer_t timer_memcpy; // Timer for GPU <---> CPU memory copying
        cudaTimer_t timer_cpu; // Timer for processing on CPU
        cudaTimer_t timer_gpu; // Timer for kernels on the GPU
        
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        printf("hello from thread %d of %d\n",cpu_thread_id,num_cpu_threads);

        // set and check the CUDA device for this CPU thread
        int gpu_id = -1;
        cudaSetDevice(cpu_thread_id % num_gpus);        // "% num_gpus" allows more CPU threads than GPU devices
        cudaGetDevice(&gpu_id);
       
        #pragma omp barrier
 
        createTimer(&timer_memcpy);
        createTimer(&timer_cpu);
        createTimer(&timer_gpu);

        printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);
        
        startTimer(timer_memcpy);
        float* d_distanceMatrix;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_distanceMatrix, sizeof(float)*NUM_EVENTS*NUM_CLUSTERS));
        float* d_memberships;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_memberships, sizeof(float)*NUM_EVENTS*NUM_CLUSTERS));
        float* d_E;// = AllocateEvents(myEvents);
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_E, sizeof(float)*NUM_EVENTS*NUM_DIMENSIONS));
        float* d_C;// = AllocateClusters(myClusters);
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_C, sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS));
        float* d_nC;// = AllocateCM(cM);
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_nC, sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS));
        float* d_denoms;// = AllocateCM(cM);
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_denoms, sizeof(float)*NUM_CLUSTERS));
        int size = sizeof(float)*NUM_DIMENSIONS*NUM_EVENTS;
        //CUDA_SAFE_CALL(cudaMemcpy(d_E, myEvents, size, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_E, transposedEvents, size, cudaMemcpyHostToDevice));
        size = sizeof(float)*NUM_DIMENSIONS*NUM_CLUSTERS;
        CUDA_SAFE_CALL(cudaMemcpy(d_C, myClusters, size, cudaMemcpyHostToDevice));
        stopTimer(timer_memcpy);
        
        printf("Starting C-means\n");
        int iterations = 0;
        
        // Compute starting/finishing indexes for the events for each gpu
        int events_per_gpu = NUM_EVENTS / num_gpus;
        int start = cpu_thread_id*events_per_gpu;
        int finish = (cpu_thread_id+1)*events_per_gpu;
        if(cpu_thread_id == (num_gpus-1)) {
            finish = NUM_EVENTS;
        }
        int my_num_events = finish-start+1;
        printf("GPU %d, Starting Event: %d, Ending Event: %d\n",cpu_thread_id,start,finish);

        do{
            cudaTimer_t timer;
            createTimer(&timer);
            startTimer(timer);

            size = sizeof(float)*NUM_DIMENSIONS*NUM_CLUSTERS;

            // Copy the cluster centers to the GPU
            startTimer(timer_memcpy);
            CUDA_SAFE_CALL(cudaMemcpy(d_C, myClusters, size, cudaMemcpyHostToDevice));
            stopTimer(timer_memcpy);
            
            dim3 BLOCK_DIM(1, NUM_THREADS, 1);

            startTimer(timer_gpu);
            DEBUG("Launching ComputeDistanceMatrix kernel\n");
            ComputeDistanceMatrix<<< NUM_CLUSTERS, NUM_THREADS_DISTANCE  >>>(d_C, d_E, d_distanceMatrix, start, finish);
            cudaThreadSynchronize();
            printCudaError();
            
            DEBUG("Launching ComputeMembershipMatrix kernel\n");
            ComputeMembershipMatrix<<< NUM_CLUSTERS, NUM_THREADS_MEMBERSHIP  >>>(d_distanceMatrix, d_memberships, start, finish);
            cudaThreadSynchronize();
            printCudaError();

            DEBUG("Launching UpdateClusterCentersGPU kernel\n");
            //UpdateClusterCentersGPU<<< NUM_BLOCKS, NUM_THREADS >>>(d_C, d_E, d_nC, d_distanceMatrix, d_denoms, start, finish);
            UpdateClusterCentersGPU2<<< dim3(NUM_CLUSTERS,NUM_DIMENSIONS), NUM_THREADS_UPDATE >>>(d_C, d_E, d_nC, d_memberships, d_denoms, start, finish);
            cudaThreadSynchronize();
            printCudaError();
            
            stopTimer(timer_gpu);
            
            // Copy partial centers and denominators to host
            startTimer(timer_memcpy);
            cudaMemcpy(tempClusters[cpu_thread_id], d_nC, sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS, cudaMemcpyDeviceToHost);
            cudaMemcpy(tempDenominators[cpu_thread_id], d_denoms, sizeof(float)*NUM_CLUSTERS, cudaMemcpyDeviceToHost);
            printCudaError();
            stopTimer(timer_memcpy);
            
            stopTimer(timer);
            float thisTime = getTimerValue(timer);
            DEBUG("Processing time for GPU %d: %f (ms) \n", cpu_thread_id, thisTime);
            deleteTimer(timer);

        
            #pragma omp barrier
            startTimer(timer_cpu);
            if(cpu_thread_id == 0) {
                // Sum up the partial cluster centers (numerators)
                for(int i=1; i < num_gpus; i++) {
                    for(int c=0; c < NUM_CLUSTERS; c++) {
                        for(int d=0; d < NUM_DIMENSIONS; d++) {
                            tempClusters[0][c*NUM_DIMENSIONS+d] += tempClusters[i][c*NUM_DIMENSIONS+d];
                        }
                    }
                }

                // Sum up the denominator for each cluster
                for(int i=1; i < num_gpus; i++) {
                    for(int c=0; c < NUM_CLUSTERS; c++) {
                        tempDenominators[0][c] += tempDenominators[i][c];
                    }
                }

                // Divide to get the final clusters
                for(int c=0; c < NUM_CLUSTERS; c++) {
                    for(int d=0; d < NUM_DIMENSIONS; d++) {
                        tempClusters[0][c*NUM_DIMENSIONS+d] /= tempDenominators[0][c];
                    }
                }
                diff = 0.0;
                for(int i=0; i < NUM_CLUSTERS; i++){
                    DEBUG("GPU %d, Cluster %d: ",cpu_thread_id,i);
                    for(int k = 0; k < NUM_DIMENSIONS; k++){
                        DEBUG("%f ",tempClusters[cpu_thread_id][i*NUM_DIMENSIONS + k]);
                        diff += fabs(myClusters[i*NUM_DIMENSIONS + k] - tempClusters[cpu_thread_id][i*NUM_DIMENSIONS + k]);
                    }
                    DEBUG("\n");
                }
                memcpy(myClusters,tempClusters[cpu_thread_id],sizeof(float)*NUM_DIMENSIONS*NUM_CLUSTERS);
                DEBUG("Diff = %f\n", diff);
                DEBUG("Done with iteration #%d\n", iterations);
                fflush(stdout);
            }
            stopTimer(timer_cpu);
            #pragma omp barrier
            iterations++;
            DEBUG("\n");
        } while(iterations < MIN_ITERS || (abs(diff) > THRESHOLD && iterations < MAX_ITERS)); 

        ComputeNormalizedMembershipMatrix<<< NUM_CLUSTERS, NUM_THREADS_MEMBERSHIP  >>>(d_distanceMatrix, d_memberships, start, finish);
        // Copy memberships from the GPU
        float* temp_memberships = (float*) malloc(sizeof(float)*NUM_EVENTS*NUM_CLUSTERS);
        cudaMemcpy(temp_memberships,d_memberships,sizeof(float)*NUM_EVENTS*NUM_CLUSTERS,cudaMemcpyDeviceToHost);
        for(int c=0; c < NUM_CLUSTERS; c++) {
            memcpy(&(memberships[c*NUM_EVENTS+cpu_thread_id*events_per_gpu]),&(temp_memberships[c*NUM_EVENTS+cpu_thread_id*events_per_gpu]),sizeof(float)*my_num_events);
        }
        free(temp_memberships);

        if(cpu_thread_id == 0) {        
            if(abs(diff) > THRESHOLD){
                PRINT("Warning: c-means did not converge to the %f threshold provided\n", THRESHOLD);
            }
            PRINT("C-means complete\n");
        }
        
        #pragma omp barrier // sync threads 
       
        #if !ENABLE_MDL
            if(cpu_thread_id == 0) {
                // Don't attempt MDL, save all clusters 
                finalClusterConfig = (int*) malloc(sizeof(int)*NUM_CLUSTERS);
                memset(finalClusterConfig,1,sizeof(int)*NUM_CLUSTERS);
            }
        #else
            PRINT("Calculating Q Matrix Section %d\n",cpu_thread_id);
           
            // Copy the latest clusters to the device 
            //  (the current ones on the device are 1 iteration old) 
            startTimer(timer_memcpy);
            CUDA_SAFE_CALL(cudaMemcpy(d_C, myClusters, size, cudaMemcpyHostToDevice));
            stopTimer(timer_memcpy);
            
            // Build Q matrix, each gpu handles NUM_DIMENSIONS/num_gpus rows of the matrix
            q_matrices[cpu_thread_id] = BuildQGPU(d_E, d_C, d_distanceMatrix, &mdlTime, cpu_thread_id, num_gpus);
            
            #pragma omp barrier // sync threads
            
            if(cpu_thread_id == 0) {
                // Combine the partial matrices
                int num_matrix_elements = NUM_CLUSTERS*(NUM_CLUSTERS/num_gpus);
                for(int i=0; i < num_gpus; i++) {
                    float* q_matrix_ptr = (float*) q_matrix+i*num_matrix_elements;
                    float* q_matrices_ptr = (float*) q_matrices[i]+i*num_matrix_elements;
                    memcpy(q_matrix_ptr,q_matrices_ptr,sizeof(float)*num_matrix_elements);   
                    free(q_matrices[i]);
                }
                startTimer(timer_cpu);
                DEBUG("Searching for optimal configuration...\n");
                finalClusterConfig = TabuSearch(q_matrix, argv[1]);
                stopTimer(timer_cpu);

                DEBUG("Q Matrix:\n");
                for(int row=0; row < NUM_CLUSTERS; row++) {
                    for(int col=0; col < NUM_CLUSTERS; col++) {
                        DEBUG("%.2e ",q_matrix[row*NUM_CLUSTERS+col]);
                    }
                    DEBUG("\n");
                }
                
                free(q_matrix);
            }
            mdlTime /= 1000.0; // CUDA timer returns time in milliseconds, normalize to seconds
        #endif

        fflush(stdout);
        #pragma omp barrier
 
        printf("\n\n"); 
        printf("Thread %d: GPU memcpy Time (ms): %f\n",cpu_thread_id,getTimerValue(timer_memcpy));
        printf("Thread %d: CPU processing Time (ms): %f\n",cpu_thread_id,getTimerValue(timer_cpu));
        printf("Thread %d: GPU processing Time (ms): %f\n",cpu_thread_id,getTimerValue(timer_gpu));
        
        #if !CPU_ONLY
            CUDA_SAFE_CALL(cudaFree(d_E));
            CUDA_SAFE_CALL(cudaFree(d_C));
            CUDA_SAFE_CALL(cudaFree(d_nC));
        #endif
    
        #pragma omp barrier
        DEBUG("Thread %d done.\n",cpu_thread_id);
    } // end of omp_parallel block
    
    cutStartTimer(timer_io);

    PRINT("Final Clusters are:\n");
    int newCount = 0;
    for(int i = 0; i < NUM_CLUSTERS; i++){
        if(finalClusterConfig[i]){
            for(int j = 0; j < NUM_DIMENSIONS; j++){
                newClusters[newCount * NUM_DIMENSIONS + j] = myClusters[i*NUM_DIMENSIONS + j];
                PRINT("%.2f\t", myClusters[i*NUM_DIMENSIONS + j]);
            }
            newCount++;
            PRINT("\n");
        }
    }
    
    #if ENABLE_OUTPUT 
        ReportSummary(newClusters, newCount, argv[1]);
        ReportResults(myEvents, memberships, newCount, argv[1]);
    #endif
    cutStopTimer(timer_io);
    cutStopTimer(timer_total);
    
    printf("Total Time (ms): %f\n",cutGetTimerValue(timer_total));
    printf("I/O Time (ms): %f\n",cutGetTimerValue(timer_io));
    printf("\n\n"); 
    
    free(newClusters);
    free(myClusters);
    free(myEvents);
    free(transposedEvents);
    return 0;
}


void generateInitialClusters(float* clusters, float* events){
    int seed;
    for(int i = 0; i < NUM_CLUSTERS; i++){
        seed = i * NUM_EVENTS / NUM_CLUSTERS;
        //seed = rand() % NUM_EVENTS;
        for(int j = 0; j < NUM_DIMENSIONS; j++){
            clusters[i*NUM_DIMENSIONS + j] = events[seed*NUM_DIMENSIONS + j];
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
    
    float* retVal = (float*)malloc(sizeof(float)*NUM_EVENTS*NUM_DIMENSIONS);
    myfile = fopen(filename, "r");
    #if LINE_LABELS
        fgets(myline, 1024, myfile);
        for(int i = 0; i < NUM_EVENTS; i++){
            fgets(myline, 1024, myfile);
            retVal[i*NUM_DIMENSIONS] = (float)atof(strtok(myline, DELIMITER));
            for(int j = 1; j < NUM_DIMENSIONS; j++){
                retVal[i*NUM_DIMENSIONS + j] = (float)atof(strtok(NULL, DELIMITER));
            }
        }
    #else
        for(int i = 0; i < NUM_EVENTS; i++){
            fgets(myline, 1024, myfile);
            retVal[i*NUM_DIMENSIONS] = (float)atof(strtok(myline, DELIMITER));
            for(int j = 1; j < NUM_DIMENSIONS; j++){
                retVal[i*NUM_DIMENSIONS + j] = (float)atof(strtok(NULL, DELIMITER));
            }
        }
    #endif
    
    fclose(myfile);
    return retVal;
}

void FreeMatrix(float* d_matrix){
    CUDA_SAFE_CALL(cudaFree(d_matrix));
}

float* BuildQGPU(float* d_events, float* d_clusters, float* distanceMatrix, float* mdlTime, int gpu_id, int num_gpus){
    float* d_matrix;
    int size = sizeof(float) * NUM_CLUSTERS*NUM_CLUSTERS;

    cudaTimer_t timer_gpu;
    cudaTimer_t timer_memcpy;
    createTimer(&timer_gpu);
    createTimer(&timer_memcpy);
    
    startTimer(timer_memcpy);
    cudaMalloc((void**)&d_matrix, size);
    printCudaError();
    stopTimer(timer_memcpy);
    
    startTimer(timer_gpu);
    dim3 grid(NUM_CLUSTERS / num_gpus, NUM_CLUSTERS);
    int start_row = gpu_id*(NUM_CLUSTERS/num_gpus);
    printf("GPU %d: Starting row for Q Matrix: %d\n",gpu_id,start_row);

    printf("Launching Q Matrix Kernel\n");
    CalculateQMatrixGPUUpgrade<<<grid, Q_THREADS>>>(d_events, d_clusters, d_matrix, distanceMatrix, start_row);
    cudaThreadSynchronize();
    printCudaError();
    stopTimer(timer_gpu);

    startTimer(timer_memcpy);
    float* matrix = (float*)malloc(size);
    printf("Copying results to CPU\n");
    cudaError_t error = cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");
    stopTimer(timer_memcpy);

    stopTimer(timer_gpu);
    *mdlTime = getTimerValue(timer_gpu);
    printf("Processing time for MDL GPU: %f (ms) \n", *mdlTime);
    printf("Memcpy time for MDL GPU: %f (ms) \n", getTimerValue(timer_memcpy));
    
    deleteTimer(timer_gpu);
    deleteTimer(timer_memcpy);
        
    printCudaError();
    
    FreeMatrix(d_matrix);
    return matrix;
}

