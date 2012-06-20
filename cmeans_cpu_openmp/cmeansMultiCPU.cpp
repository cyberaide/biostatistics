//###############################################################################
// source code  for OpenMP implementation of cmeans using multi-core on Delta
// lihui@indiana.edu   last update 5/23/2012
//###############################################################################


#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include "MDL.h"
#include "cmeansMultiCPU.h"
using namespace std;

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
    //[program name]  [data file]
    if(argc != 3){
        printf("Usage : %s [file.in][num_threads]\n",argv[0]);
        return 1;
    }//fi
    int num_threads;
    sscanf(argv[2], "%d", &num_threads);

    float* myEvents = ParseSampleInput(argv[1]);
    clock_t cpu_start;
   
    struct timespec start, finish; 
    double elapsed; 

    clock_gettime(CLOCK_MONOTONIC, &start); 
    //clock_gettime(CLOCK_MONOTONIC, &finish); 
    //elapsed = (finish.tv_sec - start.tv_sec); 
 
    if(myEvents == NULL){
        return 1;
    }

    
    int num_cpus = omp_get_num_procs();       // number of CUDA GPUs
    num_cpus = num_threads; 
    printf("number of host CPUs:%d  using %d threads\n", omp_get_num_procs(),num_threads);
    srand((unsigned)(time(0)));
    
    // Allocate arrays for the cluster centers
    float* myClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);

    // Select random cluster centers
    generateInitialClusters(myClusters, myEvents);

    // Create an array of arrays for temporary cluster centers 
    float** tempClusters = (float**) malloc(sizeof(float*)*num_cpus);
    float** tempDenominators = (float**) malloc(sizeof(float*)*num_cpus);
    for(int i=0; i < num_cpus; i++) {
        tempClusters[i] = (float*) malloc(sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);
        tempDenominators[i] = (float*) malloc(sizeof(float)*NUM_CLUSTERS);
        // memcpy(tempClusters[i],myClusters,sizeof(float)*NUM_CLUSTERS*NUM_DIMENSIONS);
    }
    
    float diff, max_change; // used to track difference in cluster centers between iterations
    float* memberships = (float*) malloc(sizeof(float)*NUM_CLUSTERS*NUM_EVENTS);
    int* finalClusterConfig;
   
    ////////////////////////////////////////////////////////////////
    // run as many CPU threads as there are CUDA devices
    omp_set_num_threads(num_cpus);  // create as many CPU threads as there are CUDA devices
    #pragma omp parallel shared(myClusters,diff,tempClusters,tempDenominators,memberships,finalClusterConfig)
    {	
        int tid = omp_get_thread_num();
        int num_cpu_threads = omp_get_num_threads();
        #pragma omp barrier
        printf("CPU thread %d (of %d)\n", tid, num_cpu_threads);
        
        //Compute starting/finishing indexes for the events for each thread
        int events_per_cpu = NUM_EVENTS / num_cpus;
        int start = tid*events_per_cpu;
        int finish = (tid+1)*events_per_cpu;
        if (tid == num_cpus-1) {
            finish = NUM_EVENTS;
        }//if

        int iterations = 0;
        do{

    	//UpdateClusterCentersCPU_Linear(t* oldClusters, const float* events,
    	//float* tempClusters, float* tempDenominators,
    	//int start, int end){

	UpdateClusterCentersCPU_Linear(myClusters,myEvents,tempClusters[tid],tempDenominators[tid],start,finish);
 
            #pragma omp barrier
            #pragma omp master
            {
                // Sum up the partial cluster centers (numerators)
                for(int i=1; i < num_cpus; i++) {
                    for(int c=0; c < NUM_CLUSTERS; c++) {
                        for(int d=0; d < NUM_DIMENSIONS; d++) {
                            tempClusters[0][c*NUM_DIMENSIONS+d] += tempClusters[i][c*NUM_DIMENSIONS+d];
                        }
                    }
                }

                // Sum up the denominator for each cluster
                for(int i=1; i < num_cpus; i++) {
                    for(int c=0; c < NUM_CLUSTERS; c++) {
                        tempDenominators[0][c] += tempDenominators[i][c];
                    }
                }

                // Divide to get the final clusters
                for(int c=0; c < NUM_CLUSTERS; c++) {
                    for(int d=0; d < NUM_DIMENSIONS; d++) {
                        tempClusters[0][c*NUM_DIMENSIONS+d] /= tempDenominators[0][c];
                    }
                }//for

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
            }//#pragma omp master

            #pragma omp barrier
            iterations++;
        } while(iterations < MIN_ITERS || (iterations < MAX_ITERS)); 

        #pragma omp barrier
        DEBUG("Thread %d done.\n",tid);
    }   // end of omp_parallel block
   
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
 
    FILE* fh = fopen("cmeans.log","a");
    fprintf(fh,"num_events:%d  time:%f \n",NUM_EVENTS,elapsed+(finish.tv_nsec-start.tv_nsec)/1000000000.0);
    fclose(fh);
    
    int newCount = NUM_CLUSTERS;
    //#if ENABLE_OUTPUT 
    ReportSummary(myClusters, newCount, argv[1]);
    //    ReportResults(myEvents, memberships, newCount, argv[1]);
    //#endif
    
    free(myClusters);
    free(myEvents);
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

float* readFCS (char*filename){

	FILE* myfile = fopen(filename, "r");
    if(myfile == NULL){
        printf("Error: File DNE\n");
        return NULL;
    }//if
    int initialHeaderSize = 59;
    char myline[initialHeaderSize];
    //float* retVal = (float*)malloc(sizeof(float)*NUM_EVENTS*NUM_DIMENSIONS);
    myfile = fopen(filename, "r");
    fgets(myline, initialHeaderSize, myfile);
    //printf("%s\n",myline);

    char *pFCSVersion = strtok(myline, " ");
    printf("%s\n",pFCSVersion);

    char *pToBeginText = strtok(NULL, " ");
    //printf("hi 0.0 :%s\n",pToBeginText);
    int toBeginText = atoi(pToBeginText);

    char *pToEndText = strtok(NULL, " ");
    int toEndText = atoi(pToEndText);
    //printf("hi 0.1 :%s\n",pToEndText);

    char *pToBeginDATA = strtok(NULL, " ");
    int toBeginDATA = atoi(pToBeginDATA);
    //printf("hi 0.2 :%s\n",pToBeginDATA);

    char *pToEndDATA = strtok(NULL, " ");
    //printf("hi 0.3 :%s\n",pToEndDATA);
    int toEndDATA = atoi(pToEndDATA);

    char *pToBeginANAL = strtok(NULL, " ");
    //printf("hi 0.4 :%s\n",pToBeginANAL);
    int toBeginANAL = atoi(pToBeginANAL);

    char *pToEndANAL = strtok(NULL, " ");
    //printf("hi 0.5 :%s\n",pToEndANAL);
    int toEndANAL = atoi(pToEndANAL);

    long remainingHeaderSize = toBeginText - initialHeaderSize;
    char* restOfHeader_Array = (char*)malloc(sizeof(char)*remainingHeaderSize);	// Create an empty character array
   	fgets(restOfHeader_Array, remainingHeaderSize, myfile);		// Populate the character array called restOfHeaderArray.
    //String restOfHeader = new String(restOfHeader_Array);
    //String fullHEADER = header + restOfHeader;

   	long sizeTEXT = toEndText - toBeginText + 1;
	char* primaryTextSection = (char*)malloc(sizeof(char)*sizeTEXT);
   	fgets(primaryTextSection, sizeTEXT, myfile);

   	//char delimiter[2];
   	//sprintf(delimiter,"%c",primaryTextSection[0]);
   	//delimiter[1] = '\0';
   	//printf("size:%d delimiter:%s\n",sizeTEXT,delimiter);
   	//printf("deliter:%c|%d| strlen:%d\n",delimiter[0],delimiter[0],strlen(pStr));

   	char *pch;
   	char *pStr = (char *)(primaryTextSection+1);

   	pch = strtok(pStr," \n\r\f");
   	int beginstext = 0;
   	char *dataType;
   	int tot;
   	int par;

   	while (pch!=NULL){
	   		if (strcmp(pch,"$DATATYPE")==0)
   	   		{
	   		pch = strtok(NULL," \n\r\f");
	   		dataType = pch;
   	   		printf("$DATATYPE:%s\n",dataType);
   	   		}//if


   	   		if (strcmp(pch,"$TOT")==0)
   	   		{
   	   		pch = strtok(NULL," \n\r\f");
   	   		tot = atoi(pch);
   	   		printf("$TOT:%d\n",tot);
   	   		}//if

   	   		if (strcmp(pch,"$PAR")==0)
   	   	   	{
   	   	   		pch = strtok(NULL," \n\r\f");
   	   	   		tot = atoi(pch);
   	   	   		printf("$PAR:%d\n",tot);
   	   	   	}//if

   	   		pch = strtok(NULL," \n\r\f");
   	   		//printf("TOT val:%s\n",pch);
   	   		//tot = atoi(pch);
   	   		//}
   	   		//pch = strtok(NULL," \n\r");
   	   		//printf("pch:%s\n",pch);
   	}//while
   	//printf("reading finished: toBeginDATA:%d\n",toBeginDATA);

   	NUM_EVENTS = tot;
   	float *dataMatrix = NULL;
   	if (strcmp(dataType,"F")==0)									// $DATATYPE = F (float); $PnB = 32
   		{
   		rewind(myfile);
   		fseek (myfile , toBeginDATA , SEEK_SET );

   		//FileWriter fstream = new FileWriter(outputFileName);
   		//BufferedWriter out = new BufferedWriter(fstream);
   		dataMatrix = (float *)malloc(sizeof(float)*tot*par);

   		float v;
   		for (int i = 0; i < tot; i++){
   				for (int j = 0; j < par; j++){
   					fread((void*)(&v), sizeof(v), 1, myfile);
   					dataMatrix[i*tot+j] = v;
   					//out.write(dataMatrix[i][j]+",");
   					}
   					//out.write(dataMatrix[i][par-1]+"\n");
   		}//for
   		fclose(myfile);
   		}//strcmp(dataType,"F")==0)
   	return dataMatrix;
}//float readFCS

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
    }
    if (strcmp(f+length-3,"fcs")==0){
    	   return readFCS(f);
    }
    if (strcmp(f+length-3,"csv")==0){
        return readCSV(f);
    }
    return readCSV(f);
}

