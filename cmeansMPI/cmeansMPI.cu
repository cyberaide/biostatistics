#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <cmeansMPI.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cmeansMPI_kernel.cu>
#include "MDL.h"
#include <mpi.h>

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
    int rank, num_nodes, len, provided;
    char name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);
    MPI_Comm_size(MPI_COMM_WORLD,&num_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Get_processor_name(name, &len);
    printf("Hello world from node %d of %d on %s\n",rank,num_nodes,name);

    unsigned int timer_io; // Timer for I/O, such as reading FCS file and outputting result files
    unsigned int timer_total; // Total time
    unsigned int timer_main_cpu; // Total time
   
    cutCreateTimer(&timer_io);
    cutCreateTimer(&timer_total);
    cutCreateTimer(&timer_main_cpu);
    
    cutStartTimer(timer_total);
    
    // [program name]  [data file]
    if(argc != 2){
        printf("Usage Error: must supply data file. e.g. programe_name @opt(flags) file.in\n");
        return 1;
    }

    cutStartTimer(timer_io);
    float* myEvents;
    if(rank == 0) {
        myEvents = ParseSampleInput(argv[1]);
        MPI_Bcast(myEvents,NUM_EVENTS*NUM_DIMENSIONS,MPI_FLOAT,0,MPI_COMM_WORLD);
    } else {
        myEvents = (float*) malloc(sizeof(float)*NUM_DIMENSIONS*NUM_EVENTS);
        MPI_Bcast(myEvents,NUM_EVENTS*NUM_DIMENSIONS,MPI_FLOAT,0,MPI_COMM_WORLD);
    }
     
    printf("Parsed file\n");
    
    cutStopTimer(timer_io);
    cutStartTimer(timer_main_cpu);
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
    
    srand((unsigned)(time(0)));
    //srand(42);
    
    // Allocate arrays for the cluster centers
    float* myClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);
    float* newClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);

    // Select random cluster centers
    generateInitialClusters(myClusters, myEvents);

    int total_num_gpus = num_gpus * num_nodes;

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
    cutStopTimer(timer_main_cpu);
   
    ////////////////////////////////////////////////////////////////
    // run as many CPU threads as there are CUDA devices
    //num_gpus = 1;
    //omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices
    #pragma omp parallel shared(myClusters,diff,tempClusters,tempDenominators,memberships,finalClusterConfig)
    {
        cudaTimer_t timer_memcpy; // Timer for GPU <---> CPU memory copying
        cudaTimer_t timer_cpu; // Timer for processing on CPU
        cudaTimer_t timer_gpu; // Timer for kernels on the GPU
        cudaTimer_t timer_mpi; // Timer for MPI
        
        unsigned int tid = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        int gpu_num = rank*num_gpus+tid;
        printf("hello from thread %d of %d\n",tid,num_cpu_threads);

        // set and check the CUDA device for this CPU thread
        int gpu_id = -1;
        cudaSetDevice(tid % num_gpus);        // "% num_gpus" allows more CPU threads than GPU devices
        cudaGetDevice(&gpu_id);
       
        #pragma omp barrier
 
        createTimer(&timer_memcpy);
        createTimer(&timer_cpu);
        createTimer(&timer_gpu);
        createTimer(&timer_mpi);

        printf("CPU thread %d (of %d) uses CUDA device %d\n", tid, num_cpu_threads, gpu_id);
        
        // Compute starting/finishing indexes for the events for each gpu
        int events_per_gpu = NUM_EVENTS / total_num_gpus;
        int my_num_events = events_per_gpu;
        if(gpu_num == (total_num_gpus-1)) {
            my_num_events += NUM_EVENTS % total_num_gpus;
        }

        startTimer(timer_memcpy);
        float* d_distanceMatrix;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_distanceMatrix, sizeof(float)*my_num_events*NUM_CLUSTERS));
        #if !LINEAR
            float* d_memberships;
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_memberships, sizeof(float)*my_num_events*NUM_CLUSTERS));
        #endif
        float* d_E;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_E, sizeof(float)*my_num_events*NUM_DIMENSIONS));
        float* d_C;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_C, sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS));
        float* d_nC;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_nC, sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS));
        float* d_denoms;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_denoms, sizeof(float)*NUM_CLUSTERS));

        int size = sizeof(float)*NUM_DIMENSIONS*my_num_events;

        // Copying the transposed data is trickier since it's not all contigious for the relavant events
        float* temp_fcs_data = (float*) malloc(size);
        for(int d=0; d < NUM_DIMENSIONS; d++) {
            memcpy(&temp_fcs_data[d*my_num_events],&transposedEvents[d*NUM_EVENTS + gpu_num*events_per_gpu],sizeof(float)*my_num_events);
        }
        CUDA_SAFE_CALL(cudaMemcpy( d_E, temp_fcs_data, size,cudaMemcpyHostToDevice) );
        cudaThreadSynchronize();
        free(temp_fcs_data);

        size = sizeof(float)*NUM_DIMENSIONS*NUM_CLUSTERS;
        CUDA_SAFE_CALL(cudaMemcpy(d_C, myClusters, size, cudaMemcpyHostToDevice));
        stopTimer(timer_memcpy);
        
        printf("Starting C-means\n");
        int iterations = 0;
        

        int num_blocks_distance = my_num_events / NUM_THREADS_DISTANCE;
        if(my_num_events % NUM_THREADS_DISTANCE) {
            num_blocks_distance++;
        }
        int num_blocks_membership = my_num_events / NUM_THREADS_MEMBERSHIP;
        if(my_num_events % NUM_THREADS_DISTANCE) {
            num_blocks_membership++;
        }
        int num_blocks_update = NUM_CLUSTERS / NUM_CLUSTERS_PER_BLOCK;
        if(NUM_CLUSTERS % NUM_CLUSTERS_PER_BLOCK) {
            num_blocks_update++;
        }

        do{
            cudaTimer_t timer;
            createTimer(&timer);
            startTimer(timer);

            size = sizeof(float)*NUM_DIMENSIONS*NUM_CLUSTERS;

            // Copy the cluster centers to the GPU
            startTimer(timer_memcpy);
            CUDA_SAFE_CALL(cudaMemcpy(d_C, myClusters, size, cudaMemcpyHostToDevice));
            stopTimer(timer_memcpy);
            

            startTimer(timer_gpu);
            DEBUG("Launching ComputeDistanceMatrix kernel\n");
            ComputeDistanceMatrix<<< dim3(num_blocks_distance,NUM_CLUSTERS), NUM_THREADS_DISTANCE  >>>(d_C, d_E, d_distanceMatrix, my_num_events);
            #if LINEAR
                // O(M) membership kernel
                DEBUG("Launching ComputeMembershipMatrixLinear kernel\n");
                ComputeMembershipMatrixLinear<<< num_blocks_membership, NUM_THREADS_MEMBERSHIP  >>>(d_distanceMatrix, my_num_events);
                DEBUG("Launching UpdateClusterCentersGPU kernel\n");
                //UpdateClusterCentersGPU<<< dim3(NUM_CLUSTERS,NUM_DIMENSIONS), NUM_THREADS_UPDATE >>>(d_C, d_E, d_nC, d_distanceMatrix, d_denoms, my_num_events);
                //UpdateClusterCentersGPU2<<< dim3(num_blocks_update,NUM_DIMENSIONS), NUM_THREADS_UPDATE >>>(d_C, d_E, d_nC, d_distanceMatrix, my_num_events);
                UpdateClusterCentersGPU3<<< dim3(NUM_DIMENSIONS,num_blocks_update), NUM_THREADS_UPDATE >>>(d_C, d_E, d_nC, d_distanceMatrix, my_num_events);
                ComputeClusterSizes<<< NUM_CLUSTERS, 512 >>>( d_distanceMatrix, d_denoms, my_num_events);
            #else
                // O(M^2) membership kernel
                DEBUG("Launching ComputeMembershipMatrix kernel\n");
                ComputeMembershipMatrix<<< dim3(num_blocks_membership,NUM_CLUSTERS), NUM_THREADS_MEMBERSHIP  >>>(d_distanceMatrix, d_memberships, my_num_events);
                DEBUG("Launching UpdateClusterCentersGPU kernel\n");
                //UpdateClusterCentersGPU<<< dim3(NUM_CLUSTERS,NUM_DIMENSIONS), NUM_THREADS_UPDATE >>>(d_C, d_E, d_nC, d_memberships, d_denoms, my_num_events);
                //UpdateClusterCentersGPU2<<< dim3(num_blocks_update,NUM_DIMENSIONS), NUM_THREADS_UPDATE >>>(d_C, d_E, d_nC, d_memberships, my_num_events);
                UpdateClusterCentersGPU3<<< dim3(NUM_DIMENSIONS,num_blocks_update), NUM_THREADS_UPDATE >>>(d_C, d_E, d_nC, d_memberships, my_num_events);
                ComputeClusterSizes<<< NUM_CLUSTERS, 512 >>>( d_memberships, d_denoms, my_num_events );
            #endif

            cudaThreadSynchronize();
            printCudaError();
            
            stopTimer(timer_gpu);
            
            // Copy partial centers and denominators to host
            startTimer(timer_memcpy);
            cudaMemcpy(tempClusters[tid], d_nC, sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS, cudaMemcpyDeviceToHost);
            cudaMemcpy(tempDenominators[tid], d_denoms, sizeof(float)*NUM_CLUSTERS, cudaMemcpyDeviceToHost);
            printCudaError();
            stopTimer(timer_memcpy);
            
            stopTimer(timer);
            float thisTime = getTimerValue(timer);
            DEBUG("Processing time for GPU %d: %f (ms) \n", tid, thisTime);
            deleteTimer(timer);

        
            #pragma omp barrier
            #pragma omp master
            {
                startTimer(timer_cpu);
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
                stopTimer(timer_cpu);

                DEBUG("Reducing cluster values\n");
                startTimer(timer_mpi);
                if(rank == 0) {
                    MPI_Reduce(MPI_IN_PLACE,tempClusters[0],NUM_DIMENSIONS*NUM_CLUSTERS,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
                    //MPI_Reduce(tempClusters[0],tempClusters[1],NUM_DIMENSIONS*NUM_CLUSTERS,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
                    //memcpy(tempClusters[0],tempClusters[1],sizeof(float)*NUM_DIMENSIONS*NUM_CLUSTERS);
                    MPI_Reduce(MPI_IN_PLACE,tempDenominators[0],NUM_CLUSTERS,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
                    //MPI_Reduce(tempDenominators[0],tempDenominators[1],NUM_CLUSTERS,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
                    //memcpy(tempDenominators[0],tempDenominators[1],sizeof(float)*NUM_CLUSTERS);
                } else {
                    MPI_Reduce(tempClusters[0],0,NUM_DIMENSIONS*NUM_CLUSTERS,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
                    MPI_Reduce(tempDenominators[0],0,NUM_CLUSTERS,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
                }
                MPI_Barrier(MPI_COMM_WORLD); // not sure if neccesary...
                stopTimer(timer_mpi);

                startTimer(timer_cpu);
                // Divide to get the final clusters
                if(rank == 0) {
                    for(int c=0; c < NUM_CLUSTERS; c++) {
                        for(int d=0; d < NUM_DIMENSIONS; d++) {
                            tempClusters[0][c*NUM_DIMENSIONS+d] /= tempDenominators[0][c];
                        }
                    }
                }
                stopTimer(timer_cpu);
                startTimer(timer_mpi);
                DEBUG("Broadcasting Cluster Values\n");
                MPI_Bcast(tempClusters[0],NUM_DIMENSIONS*NUM_CLUSTERS,MPI_FLOAT,0,MPI_COMM_WORLD);
                MPI_Barrier(MPI_COMM_WORLD);
                stopTimer(timer_mpi);

                startTimer(timer_cpu);
                diff = 0.0;
                for(int i=0; i < NUM_CLUSTERS; i++){
                    DEBUG("GPU %d, Cluster %d: ",tid,i);
                    for(int k = 0; k < NUM_DIMENSIONS; k++){
                        DEBUG("%f ",tempClusters[tid][i*NUM_DIMENSIONS + k]);
                        diff += fabs(myClusters[i*NUM_DIMENSIONS + k] - tempClusters[tid][i*NUM_DIMENSIONS + k]);
                    }
                    DEBUG("\n");
                }
                memcpy(myClusters,tempClusters[tid],sizeof(float)*NUM_DIMENSIONS*NUM_CLUSTERS);
                DEBUG("Diff = %f\n", diff);
                DEBUG("Done with iteration #%d\n", iterations);
                stopTimer(timer_cpu);
            }
            #pragma omp barrier
            iterations++;
            DEBUG("\n");
        } while(iterations < MIN_ITERS || (abs(diff) > THRESHOLD && iterations < MAX_ITERS)); 

        #pragma omp master
        {
            if(rank == 0) {
                printf("Iterations: %d\n",iterations);
            }
        }
        #if ENABLE_OUTPUT
            // Compute final membership vaues
            startTimer(timer_gpu);
            #if LINEAR
                // O(M)
                ComputeDistanceMatrix<<< dim3(num_blocks_distance,NUM_CLUSTERS), NUM_THREADS_DISTANCE  >>>(d_C, d_E, d_distanceMatrix, my_num_events);
                ComputeNormalizedMembershipMatrixLinear<<< num_blocks_membership, NUM_THREADS_MEMBERSHIP >>>(d_distanceMatrix,my_num_events);
            #else
                // O(M^2)
                ComputeNormalizedMembershipMatrix<<< dim3(num_blocks_membership,NUM_CLUSTERS), NUM_THREADS_MEMBERSHIP  >>>(d_distanceMatrix, d_memberships, my_num_events);
            #endif
            stopTimer(timer_gpu);

            // Copy memberships from the GPU
            float* temp_memberships = (float*) malloc(sizeof(float)*my_num_events*NUM_CLUSTERS);
            startTimer(timer_memcpy);
            #if LINEAR
                cudaMemcpy(temp_memberships,d_distanceMatrix,sizeof(float)*my_num_events*NUM_CLUSTERS,cudaMemcpyDeviceToHost);
            #else
                cudaMemcpy(temp_memberships,d_memberships,sizeof(float)*my_num_events*NUM_CLUSTERS,cudaMemcpyDeviceToHost);
            #endif
            stopTimer(timer_memcpy);

            startTimer(timer_cpu);
            for(int c=0; c < NUM_CLUSTERS; c++) {
                memcpy(&(memberships[c*NUM_EVENTS+gpu_num*events_per_gpu]),&(temp_memberships[c*my_num_events]),sizeof(float)*my_num_events);
            }
            stopTimer(timer_cpu);
            #pragma omp barrier
            #pragma omp master
            {
                startTimer(timer_cpu);
                // First transpose the memberships, makes it easier to gather the results between nodes
                float* temp = (float*) malloc(sizeof(float)*NUM_EVENTS*NUM_CLUSTERS);
                for(int e=0; e < NUM_EVENTS; e++) {
                    for(int c=0; c < NUM_CLUSTERS; c++) {
                        temp[e*NUM_CLUSTERS+c] = memberships[c*NUM_EVENTS+e];
                    }
                }
                memcpy(memberships,temp,sizeof(float)*NUM_EVENTS*NUM_CLUSTERS);
                stopTimer(timer_cpu);
                // Gather memberships on root        
                startTimer(timer_mpi);
                int memberships_being_sent, memberships_per_node;
                memberships_per_node = events_per_gpu*num_gpus*NUM_CLUSTERS;
                if(rank == 0) {
                    for(int i=1; i < num_nodes; i++) {
                        memberships_being_sent = memberships_per_node;
                        if(i == num_nodes-1) { // boundary condition
                            memberships_being_sent += (NUM_EVENTS % total_num_gpus)*NUM_CLUSTERS;
                        }
                        MPI_Status s;
                        MPI_Recv(&(temp[memberships_per_node*i]),memberships_being_sent,MPI_FLOAT,i,1,MPI_COMM_WORLD,&s);
                    }
                } else {
                    memberships_being_sent = memberships_per_node;
                    if(rank == num_nodes-1) { // boundary condition
                        memberships_being_sent += (NUM_EVENTS % total_num_gpus)*NUM_CLUSTERS;
                    }
                    MPI_Send(&(memberships[memberships_per_node*rank]),memberships_being_sent,MPI_FLOAT,0,1,MPI_COMM_WORLD);
                }
                MPI_Barrier(MPI_COMM_WORLD);
                stopTimer(timer_mpi);
                // Tranpose the memberships again to get original ordering
                startTimer(timer_cpu);
                if(rank == 0) {
                    for(int e=0; e < NUM_EVENTS; e++) {
                        for(int c=0; c<NUM_CLUSTERS; c++) {
                            memberships[c*NUM_EVENTS+e] = temp[e*NUM_CLUSTERS+c];
                        }
                    }    
                }
                free(temp);
                stopTimer(timer_cpu);
            }
            #pragma omp barrier
            free(temp_memberships);
        #endif // #if ENABLE_OUTPUT

        if(tid == 0) {        
            if(abs(diff) > THRESHOLD){
                PRINT("Warning: c-means did not converge to the %f threshold provided\n", THRESHOLD);
            }
            PRINT("C-means complete\n");
        }
        
        #pragma omp barrier // sync threads 
       
        #if !ENABLE_MDL
            if(tid == 0) {
                // Don't attempt MDL, save all clusters 
                finalClusterConfig = (int*) malloc(sizeof(int)*NUM_CLUSTERS);
                memset(finalClusterConfig,1,sizeof(int)*NUM_CLUSTERS);
            }
        #else
            PRINT("Calculating Q Matrix Section %d\n",tid);
           
            // Copy the latest clusters to the device 
            //  (the current ones on the device are 1 iteration old) 
            startTimer(timer_memcpy);
            CUDA_SAFE_CALL(cudaMemcpy(d_C, myClusters, size, cudaMemcpyHostToDevice));
            stopTimer(timer_memcpy);
            
            // Build Q matrix, each gpu handles NUM_DIMENSIONS/num_gpus rows of the matrix
            q_matrices[tid] = BuildQGPU(d_E, d_C, d_distanceMatrix, &mdlTime, tid, num_gpus, my_num_events);
            
            #pragma omp barrier // sync threads
            
            if(tid == 0) {
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
        printf("Node %d: Thread %d: GPU memcpy Time (ms): %f\n",rank,tid,getTimerValue(timer_memcpy));
        printf("Node %d: Thread %d: CPU processing Time (ms): %f\n",rank,tid,getTimerValue(timer_cpu));
        printf("Node %d: Thread %d: GPU processing Time (ms): %f\n",rank,tid,getTimerValue(timer_gpu));
        printf("Node %d: Thread %d: MPI Time (ms): %f\n",rank,tid,getTimerValue(timer_mpi));
        
        #if !CPU_ONLY
            CUDA_SAFE_CALL(cudaFree(d_E));
            CUDA_SAFE_CALL(cudaFree(d_C));
            CUDA_SAFE_CALL(cudaFree(d_nC));
        #endif
    
        #pragma omp barrier
        DEBUG("Thread %d done.\n",tid);
    } // end of omp_parallel block
    
    cutStartTimer(timer_io);

    if(rank == 0) {
        PRINT("Final Clusters are:\n");
        int newCount = 0;
        for(int i = 0; i < NUM_CLUSTERS; i++){
            if(finalClusterConfig[i]){
                for(int j = 0; j < NUM_DIMENSIONS; j++){
                    newClusters[newCount * NUM_DIMENSIONS + j] = myClusters[i*NUM_DIMENSIONS + j];
                    PRINT("%.3f\t", myClusters[i*NUM_DIMENSIONS + j]);
                }
                newCount++;
                PRINT("\n");
            }
        }
        
        #if ENABLE_OUTPUT 
            ReportSummary(newClusters, newCount, argv[1]);
            ReportResults(myEvents, memberships, newCount, argv[1]);
        #endif
    }

    cutStopTimer(timer_io);
    cutStopTimer(timer_total);
    
    printf("Total Time (ms): %f\n",cutGetTimerValue(timer_total));
    printf("I/O Time (ms): %f\n",cutGetTimerValue(timer_io));
    printf("Main Thread CPU Time (ms): %f\n",cutGetTimerValue(timer_main_cpu));
    printf("\n\n"); 
    
    free(newClusters);
    free(myClusters);
    free(myEvents);
    free(transposedEvents);
    MPI_Finalize();
    return 0;
}


void generateInitialClusters(float* clusters, float* events){
    int seed;
    for(int i = 0; i < NUM_CLUSTERS; i++){
        //seed = i * NUM_EVENTS / NUM_CLUSTERS;
        seed = rand() % NUM_EVENTS;
        for(int j = 0; j < NUM_DIMENSIONS; j++){
            clusters[i*NUM_DIMENSIONS + j] = events[seed*NUM_DIMENSIONS + j];
        }
    }
    
}

float* readBIN(char* f) {
    FILE* fin = fopen(f,"rb");
    int nevents,ndims;
    fread(&nevents,4,1,fin);
    fread(&ndims,4,1,fin);
    int num_elements = (ndims)*(nevents);
    printf("Number of rows: %d\n",nevents);
    printf("Number of cols: %d\n",ndims);
    float* data = (float*) malloc(sizeof(float)*num_elements);
    fread(data,sizeof(float),num_elements,fin);
    fclose(fin);
    return data;
}


float* readCSV(char* filename) {
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

float* ParseSampleInput(char* f){
    int length = strlen(f);
    printf("File Extension: %s\n",f+length-3);
    if(strcmp(f+length-3,"bin") == 0) {
        return readBIN(f);
    } else {
        return readCSV(f);
    }
}


void FreeMatrix(float* d_matrix){
    CUDA_SAFE_CALL(cudaFree(d_matrix));
}

float* BuildQGPU(float* d_events, float* d_clusters, float* distanceMatrix, float* mdlTime, int gpu_id, int num_gpus, int my_num_events){
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
    CalculateQMatrixGPUUpgrade<<<grid, Q_THREADS>>>(d_events, d_clusters, d_matrix, distanceMatrix, start_row, my_num_events);
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

