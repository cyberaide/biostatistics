#include "cmeans.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <iostream>
#include <fstream>

#include "timers.h"
#include <cuda_runtime.h>
#include <cutil.h>

using namespace std;

float MembershipValueDist(const float* clusters, const float* events, int eventIndex, float distance){
    float sum =0;
    float otherClustDist;
    for(int j = 0; j< NUM_CLUSTERS; j++){
        otherClustDist = CalculateDistanceCPU(clusters, events, j, eventIndex); 
        if(otherClustDist < .000001)
            return 0.0;
        sum += pow((float)(distance/otherClustDist),float(2/(FUZZINESS-1)));
    }
    return 1/sum;
}




float CalculateQII(float* events, float* clusters, int cluster_index_I){
    float EI = 0;
    float numMem = 0;
    for(int i = 0; i < NUM_EVENTS; i++){
        float distance = CalculateDistanceCPU(clusters, events, cluster_index_I, i);
        float memVal = MembershipValueDist(clusters, events, i, distance);
    
        if(memVal > MEMBER_THRESH){
            EI += pow(memVal, 2) * pow(distance, 2);
            numMem++;
        }
    }
        
    return (float)(((float)K1)*numMem - ((float)K2)*EI - ((float)K3)*ALL_DIMENSIONS);

}

//Qij = (-K1 #events in both Ci and Cj + K2*Eij)/2 i!=j

float CalculateQIJ(float* events, float* clusters, int cluster_index_I, int cluster_index_J){
    
    float EI = 0;
    float EJ = 0;
    float numMem = 0;
    for(int i = 0; i < NUM_EVENTS; i++){
        float distance = CalculateDistanceCPU(clusters, events, cluster_index_I, i);
        float memValI = MembershipValueDist(clusters, events, i, distance);
    
        if(memValI > MEMBER_THRESH){
            EI += pow(memValI, 2) * pow(distance, 2);
            
        }
        
        distance = CalculateDistanceCPU(clusters, events, cluster_index_J, i);
        float memValJ = MembershipValueDist(clusters, events, i, distance);
        if(memValJ > MEMBER_THRESH){
            EJ += pow(memValJ, 2) * pow(distance, 2);
            
        }
        if(memValI > MEMBER_THRESH && memValJ > MEMBER_THRESH){
            numMem++;
        }
        if(cluster_index_I == 1 && cluster_index_J == 0 && i%10 == 0){
        }
        
    }
    return (float)(-1*((float)K1)*numMem + ((float)K2)*((EI > EJ) ? EI : EJ));

}

void CalculateQMatrix(float* events, float* clusters, float* matrix){
    
    for(int i = 0; i < NUM_CLUSTERS; i++){
        for(int j = 0; j < NUM_CLUSTERS; j++){
            if(i == j){
                matrix[i*NUM_CLUSTERS + j] = CalculateQII(events, clusters, i);
            } else{
                matrix[i*NUM_CLUSTERS + j] = CalculateQIJ(events, clusters, i, j);
            }
        }
    
    }

}

float EvaluateSolution(float* matrix, int* config){
    float partial[NUM_CLUSTERS] = {0};
    for(int i = 0; i < NUM_CLUSTERS; i++){
        for(int j = 0; j < NUM_CLUSTERS; j++){
            partial[i] += (config[i] == 0) ? 0 : matrix[i + j*NUM_CLUSTERS];
        }
    } 
    float score = 0;
    for(int i = 0; i < NUM_CLUSTERS; i++){
        score += (config[i] == 0) ? 0 : partial[i];
    }
    return score;
}

int* TabuSearch(float* matrix, char* inputFile){
    //unsigned long config = (((unsigned long)1) << NUM_CLUSTERS) - 1;
    int* config = (int*)malloc(sizeof(int)* NUM_CLUSTERS);
    int* minimumConfig = (int*)malloc(sizeof(int)* NUM_CLUSTERS);
    for(int i = 0; i < NUM_CLUSTERS; i++){
        config[i] = 1;
        minimumConfig[i] = 1;
    }
    int history[NUM_CLUSTERS] = {0};
    float minimumScore = EvaluateSolution(matrix, config);
    
    
    int minimumIndex =0;

    ofstream myfile;
    char logFileName [512];
    sprintf(logFileName, "%s_tabu_search_results_table_%d", inputFile, NUM_CLUSTERS);
    cout << "Tabu Search Results Table filename = " << logFileName << " int_min = "
        << INT_MIN << endl;
    myfile.open(logFileName);


    for(int i = 0; i < TABU_ITER; i++){
        
        
        float currentScore = INT_MIN;//FLT_MAX;

        for(int j = 0; j < NUM_CLUSTERS; j++){
            if(history[j] == 0){ // can change
                int oldVal = config[j];
                if(oldVal)
                    config[j] = 0;
                else
                    config[j] = 1;
                float tmpScore = EvaluateSolution(matrix, config);
                //float tmpScore = EvaluateSolution(matrix, config ^ (long)pow((float)2, (float)(NUM_CLUSTERS - j - 1)));
                //float tmpScore = EvaluateSolution(matrix, config ^ (unsigned long)(((unsigned long)1) << (NUM_CLUSTERS - j - 1)));
                //if(i==0){
                //  myfile << hex << endl;
                //  myfile << j << " " << (1 << (NUM_CLUSTERS - j - 1)) << " "<< (config ^ (unsigned long)(((unsigned long)1) << (NUM_CLUSTERS - j - 1)))  << endl;

                //}
                
                if(tmpScore > currentScore && tmpScore != 0){
                    currentScore = tmpScore;
                    minimumIndex = j;
                }
                config[j] = oldVal;
            }
            else{
                history[j]--;
            }
        }
        
        //config = config ^ (long)pow((float)2, (float)(NUM_CLUSTERS - minimumIndex - 1));
        if(config[minimumIndex])
            config[minimumIndex] = 0;
        else
            config[minimumIndex] = 1;       
    //config[ = config ^ (1 << (NUM_CLUSTERS - minimumIndex - 1));
        history[minimumIndex] = TABU_TENURE;
        
        if(currentScore > minimumScore){
            minimumScore = currentScore;
            for(int i = 0; i < NUM_CLUSTERS; i++){
                minimumConfig[i] = config[i];
            }
        }

    
        
        myfile << i << ", " << bitCount(config) << ", " << currentScore << "," << "\n";

    }
    myfile.close();
    free(config);
    return minimumConfig;

}

int bitCount (int* n)  {
   int count = 0 ;
   for(int i = 0; i < NUM_CLUSTERS; i++){
    count += n[i];
   }
   return count ;
}

int* MDL(float* events, float* clusters, float* mdlTime, char* inputFile){
    CUT_SAFE_CALL(cutStartTimer(timer_cpu));
    float* matrix = (float*) malloc (sizeof(float) * NUM_CLUSTERS * NUM_CLUSTERS);
    
    printf("Starting MDL\n");
    clock_t cpu_start, cpu_stop;
        cpu_start = clock();

        
    CalculateQMatrix(events, clusters, matrix);

    cpu_stop = clock();
    printf("Processing time for CPU: %f (s) \n", (float)(cpu_stop - cpu_start)/(float)(CLOCKS_PER_SEC));
    *mdlTime = (float)(cpu_stop - cpu_start)/(float)(CLOCKS_PER_SEC);
    //for(int i = 0; i < NUM_CLUSTERS; i++){
    //  for(int j = 0; j < NUM_CLUSTERS; j++){
    //      printf("%f\t", matrix[i*NUM_CLUSTERS + j]);
    //  }
    //  printf("\n");
    //}
    printf("Searching...\n");
    
    int* finalConfig = TabuSearch(matrix, inputFile);
    
    CUT_SAFE_CALL(cutStopTimer(timer_cpu));
    
    free(matrix);
    
    return finalConfig;

}


int* MDLGPU(float* d_events, float* d_clusters, float* mdlTime, char* inputFile){
    
    printf("Calculating Q Matrix\n");
    
    float* matrix = BuildQGPU(d_events, d_clusters, mdlTime);
    //for(int i = 0; i < NUM_CLUSTERS; i++){
    //  for(int j = 0; j < NUM_CLUSTERS; j++){
    //      printf("%f\t", matrix[i*NUM_CLUSTERS + j]);
    //  }
    //  printf("\n");
    //}
    printf("Searching...\n");
    
    CUT_SAFE_CALL(cutStartTimer(timer_cpu));
    int* finalConfig = TabuSearch(matrix, inputFile);
    CUT_SAFE_CALL(cutStopTimer(timer_cpu));
    
    //printf("my config = %ld\n", finalConfig);
    
    free(matrix);
    return finalConfig;

}

/*long TabuSearchGPU(float* d_matrix){
    //long config = (long)pow((float)2, (float)(NUM_CLUSTERS )) - 1; // initialize at 111...11
    long config = (((long)1) << NUM_CLUSTERS) - 1;
    int history[NUM_CLUSTERS] = {0};
    float minimumScore = FindScoreGPU(d_matrix, config);
    
    long minimumConfig = config;
    int minimumIndex=0;

    for(int i = 0; i < TABU_ITER; i++){
        if(i%25 == 0) printf("\tOn iteration %d\n", i);

        float currentScore = INT_MIN;//FLT_MAX;

        for(int j = 0; j < NUM_CLUSTERS; j++){
            if(history[j] == 0){ // can change
                //float tmpScore = EvaluateSolution(matrix, config ^ (long)pow((float)2, (float)(NUM_CLUSTERS - j - 1)));
                float tmpScore = FindScoreGPU(d_matrix, config ^ (1 << (NUM_CLUSTERS - j - 1)));
                if(tmpScore > currentScore && tmpScore != 0){
                    currentScore = tmpScore;
                    minimumIndex = j;
                }
            }
            else{
                history[j]--;
            }
        }
        
        //config = config ^ (long)pow((float)2, (float)(NUM_CLUSTERS - minimumIndex - 1));
        config = config ^ (1 << (NUM_CLUSTERS - minimumIndex - 1));
        history[minimumIndex] = TABU_TENURE;
        
        if(currentScore > minimumScore){
            minimumScore = currentScore;
            minimumConfig = config;
        }

    }
    return minimumConfig;

}*/
