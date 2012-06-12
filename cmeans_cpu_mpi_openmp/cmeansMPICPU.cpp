// ###############################################################################
// source code  for MPI implementation of cmeans using multi-core on Delta
// lihui@indiana.edu   last update 5/27/2012
// ###############################################################################

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include "MDL.h"
#include "cmeansMPICPU.h"

int NUM_EVENTS;  //number of events exists in input file

float CalculateDistanceCPU(const float* clusters, const float* events, int clusterIndex, int eventIndex){
    float sum = 0;
#if DISTANCE_MEASURE == 0
    for(int i = 0; i < NUM_DIMENSIONS; i++){
        float tmp = events[eventIndex*NUM_DIMENSIONS + i] - clusters[clusterIndex*NUM_DIMENSIONS + i];
        sum += tmp*tmp;
    }
    sum = sqrt(sum+1e-30);
#endif
#if DISTANCE_MEASURE == 1
    for(int i = 0; i < NUM_DIMENSIONS; i++){
        float tmp = events[eventIndex*NUM_DIMENSIONS + i] - clusters[clusterIndex*NUM_DIMENSIONS + i];
        sum += abs(tmp)+1e-30;
    }
#endif
#if DISTANCE_MEASURE == 2
    for(int i = 0; i < NUM_DIMENSIONS; i++){
        float tmp = abs(events[eventIndex*NUM_DIMENSIONS + i] - clusters[clusterIndex*NUM_DIMENSIONS + i]);
        if(tmp > sum)
            sum = tmp+1e-30;
    }
#endif
    return sum;
}//

//Function that caluate new cluster based on distribution of points and previous cluster
//Work on different parts of events and run in parallel.

void UpdateClusterCentersCPU_Linear(const float* oldClusters, const float* events, 
    float* tempClusters, float* tempDenominators,
    int start, int end){
	
    float membershipValue, denominator;
    float* numerator = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
    float* distances = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
    float* memberships = (float*)malloc(sizeof(float)*NUM_CLUSTERS);

    for(int i = 0; i < NUM_DIMENSIONS*NUM_CLUSTERS; i++) {
        tempClusters[i] = 0;
    }//for

    for(int i = 0; i < NUM_CLUSTERS; i++) {
        numerator[i] = 0;
        tempDenominators[i] = 0;
    }//for
  
    for(int n = start; n < end; n++){
        denominator = 0.0f;
        for(int c = 0; c < NUM_CLUSTERS; c++){
            distances[c] = CalculateDistanceCPU(oldClusters, events, c, n);
            numerator[c] = powf(distances[c],2.0f/(FUZZINESS-1.0f))+1e-30; 
// prevents divide by zero error if distance is really small and powf makes it underflow
            denominator = denominator + 1.0f/numerator[c];
        }//for

	// Add contribution to numerator and denominator
        for(int c = 0; c < NUM_CLUSTERS; c++){
            membershipValue = 1.0f/powf(numerator[c]*denominator,(float)FUZZINESS);
            for(int d = 0; d < NUM_DIMENSIONS; d++){
                tempClusters[c*NUM_DIMENSIONS+d] += events[n*NUM_DIMENSIONS+d]*membershipValue;
            }
            tempDenominators[c] += membershipValue;
        }//for
    }//for

    free(numerator);
    free(distances);
    free(memberships);

}//void

int main(int argc, char* argv[])
{
    unsigned int timer_io;    // Timer for I/O, such as reading FCS file and outputting result files
    unsigned int timer_total; // Total time
    if(argc != 2){
        printf("Usage Error: must supply data file. e.g. programe_name @opt(flags) file.in\n");
        return 1;
    }//fi

    int rank, num_nodes, len, provided,num_cpus;
    char name[MPI_MAX_PROCESSOR_NAME]; 

    MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);
    MPI_Comm_size(MPI_COMM_WORLD,&num_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Get_processor_name(name, &len);
    num_cpus = omp_get_num_procs();

    float *myEvents;
    clock_t cpu_start;
  
    int events_per_node, events_being_sent;
    struct timespec start, finish; 
    double elapsed; 
    clock_gettime(CLOCK_MONOTONIC, &start); 
    float* myClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);
    if(rank == 0) {
        myEvents = ParseSampleInput(argv[1]);
        if (myEvents == NULL){
	return 1;
	}
	generateInitialClusters(myClusters, myEvents);
    }//fi

    MPI_Bcast(&NUM_EVENTS,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(myClusters, NUM_DIMENSIONS*NUM_CLUSTERS,MPI_FLOAT,0,MPI_COMM_WORLD);
    events_per_node = NUM_EVENTS / num_nodes;

    if(rank == 0){
        MPI_Request* requests = (MPI_Request*) malloc(sizeof(MPI_Request)*num_nodes);
        MPI_Status s;
        // Send everything asynchronously
        for(int i=1; i < num_nodes; i++) {
            events_being_sent = events_per_node;
            if(i == num_nodes-1) { // boundary condition
                events_being_sent += (NUM_EVENTS % num_nodes);
            }
   MPI_Isend(&(myEvents[events_per_node*i*NUM_DIMENSIONS]),events_being_sent*NUM_DIMENSIONS,MPI_FLOAT,i,1,MPI_COMM_WORLD,&requests[i]);
        }
        // Wait for the Isends to complete
        for(int i=1; i < num_nodes; i++) {
            MPI_Wait(&requests[i],&s);
        }
        free(requests);
        events_being_sent = events_per_node; // so that its set properly for the root
    } else {
        MPI_Status s; 
        myEvents = (float*) malloc(sizeof(float)*NUM_DIMENSIONS*NUM_EVENTS);
        events_being_sent = events_per_node;
        if(rank == num_nodes-1) { // boundary condition
            events_being_sent += (NUM_EVENTS % num_nodes);
        }
 MPI_Recv(&(myEvents[events_per_node*rank*NUM_DIMENSIONS]),events_being_sent*NUM_DIMENSIONS,MPI_FLOAT,0,1,MPI_COMM_WORLD,&s);
    }
    
    srand((unsigned)(time(0)));
    
    // Create an array of arrays for temporary cluster centers 
    float** tempClusters = (float**) malloc(sizeof(float*)*num_cpus);
    float** tempDenominators = (float**) malloc(sizeof(float*)*num_cpus);
    for(int i=0; i < num_cpus; i++) {
        tempClusters[i] = (float*) malloc(sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);
        tempDenominators[i] = (float*) malloc(sizeof(float)*NUM_CLUSTERS);
        memcpy(tempClusters[i],myClusters,sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);
    }//for
    
    float diff, max_change; // used to track difference in cluster centers between iterations
    float* memberships = (float*) malloc(sizeof(float)*NUM_CLUSTERS*NUM_EVENTS);
    int* finalClusterConfig;
 
    ////////////////////////////////////////////////////////
    // run as many CPU threads as there are CUDA devices
    omp_set_num_threads(num_cpus);  // create as many CPU threads as there are CUDA devices
    #pragma omp parallel shared(myClusters,diff,tempClusters,tempDenominators,memberships,finalClusterConfig)
    {
        int tid = omp_get_thread_num();
        int num_cpu_threads = omp_get_num_threads();
        #pragma omp barrier
        
        //Compute starting/finishing indexes for the events for each thread
        int events_per_cpu = events_being_sent / num_cpus;
        int start = rank*events_per_node+tid*events_per_cpu;
        int finish = start+events_per_cpu;
        if (tid == num_cpus-1) {
            finish += events_being_sent%num_cpus;
        }//if

        int iterations = 0;
        do{

	    UpdateClusterCentersCPU_Linear(myClusters,myEvents,tempClusters[tid],tempDenominators[tid],start,finish);
 
            #pragma omp barrier
            #pragma omp master
            {
                //Sum up the partial cluster centers (numerators)
                for(int i=1; i < num_cpus; i++) {
                    for(int c=0; c < NUM_CLUSTERS; c++) {
                        for(int d=0; d < NUM_DIMENSIONS; d++) {
                            tempClusters[0][c*NUM_DIMENSIONS+d] += tempClusters[i][c*NUM_DIMENSIONS+d];
                        }//for
                    }//for
                }//for

                // Sum up the denominator for each cluster
                for(int i=1; i < num_cpus; i++) {
                    for(int c=0; c < NUM_CLUSTERS; c++) {
                        tempDenominators[0][c] += tempDenominators[i][c];
                    }
                }

 		if(rank == 0) {
                    MPI_Reduce(MPI_IN_PLACE,tempClusters[0],NUM_DIMENSIONS*NUM_CLUSTERS,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
                    MPI_Reduce(MPI_IN_PLACE,tempDenominators[0],NUM_CLUSTERS,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
                } else {
                    MPI_Reduce(tempClusters[0],0,NUM_DIMENSIONS*NUM_CLUSTERS,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
                    MPI_Reduce(tempDenominators[0],0,NUM_CLUSTERS,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
                }
                MPI_Barrier(MPI_COMM_WORLD); // not sure if neccesary...

                //Divide to get the final clusters

		if(rank==0){
                for(int c=0; c < NUM_CLUSTERS; c++) {
                    for(int d=0; d < NUM_DIMENSIONS; d++) {
                        tempClusters[0][c*NUM_DIMENSIONS+d] /= tempDenominators[0][c];
                    }
                }//for
		}

                DEBUG("Broadcasting Cluster Values\n");
                MPI_Bcast(tempClusters[0],NUM_DIMENSIONS*NUM_CLUSTERS,MPI_FLOAT,0,MPI_COMM_WORLD);
                MPI_Barrier(MPI_COMM_WORLD);

                diff = 0.0;
                max_change = 0.0;
                for(int i=0; i < NUM_CLUSTERS; i++){
                    for(int k = 0; k < NUM_DIMENSIONS; k++){
                      diff += fabs(myClusters[i*NUM_DIMENSIONS + k] - tempClusters[0][i*NUM_DIMENSIONS + k]);
            max_change = fmaxf(max_change,fabs(myClusters[i*NUM_DIMENSIONS + k] - tempClusters[0][i*NUM_DIMENSIONS + k]));
                    }
                }
                memcpy(myClusters,tempClusters[0],sizeof(float)*NUM_DIMENSIONS*NUM_CLUSTERS);
                DEBUG("Iteration %d: Total Change = %e, Max Change = %e\n", iterations, diff, max_change);
                DEBUG("Done with iteration #%d\n", iterations);
		//for (int i=1;i<NUM_CLUSTERS
            }//#pragma omp master
	
            #pragma omp barrier
            iterations++;
        } while(iterations < MIN_ITERS || (iterations < MAX_ITERS)); 

        #pragma omp barrier
        DEBUG("Rank:%d Thread %d done.\n",rank,tid);
    }   //end of omp_parallel block
   
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    float msec = (finish.tv_nsec-start.tv_nsec)/1000000000.0;

    if(rank==0) {
    FILE* fh = fopen("cmeans.log","a");
    fprintf(fh,"num_events:%d  time:%f \n",NUM_EVENTS,elapsed+msec);
    fclose(fh);
    
    int newCount = NUM_CLUSTERS;
    ReportSummary(myClusters, newCount, argv[1]);
    //ReportResults(myEvents, memberships, newCount, argv[1]);
    }
    printf("rank :%d  done........\n",rank); 
    free(myClusters);
    free(myEvents);
    MPI_Finalize();
    return 0;
}


void generateInitialClusters(float* clusters, float* events){
    int seed;
    srand(time(NULL));
    for(int i = 0; i < NUM_CLUSTERS; i++){
        #if RANDOM_SEED
            seed = rand() % NUM_EVENTS;
        #else
            seed = i * NUM_EVENTS / NUM_CLUSTERS;
        #endif
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
    int num_elements = NUM_EVENTS*NUM_DIMENSIONS;
    printf("Number of rows: %d\n",nevents);
    printf("Number of cols: %d\n",ndims);
    float* data = (float*) malloc(sizeof(float)*NUM_EVENTS*NUM_DIMENSIONS);
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
    //float* retVal = (float*)malloc(sizeof(float)*NUM_EVENTS*NUM_DIMENSIONS);
    myfile = fopen(filename, "r");

    NUM_EVENTS = 0;
    while (fgets(myline, 10000, myfile) != NULL)
       NUM_EVENTS ++;

    rewind(myfile);
    float* retVal = (float*)malloc(sizeof(float)*NUM_EVENTS*NUM_DIMENSIONS);

    #if LINE_LABELS
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

