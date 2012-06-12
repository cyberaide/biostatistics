#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <iostream>
#include <fstream>

#include "cmeansMPICPU.h"
#include "MDL.h"

using namespace std;


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
    int history[NUM_CLUSTERS];
    //memset(history,0,sizeof(int)*NUM_CLUSTERS);
    for (int i = 0; i < NUM_CLUSTERS; i++){
	history[i] = 0;
    }
    float minimumScore = EvaluateSolution(matrix, config);
    
    
    int minimumIndex =0;

    ofstream myfile;
    char logFileName [512];
    sprintf(logFileName, "%s_tabu_search_results_table_%d", inputFile, NUM_CLUSTERS);
    cout << "Tabu Search Results Table filename = " << logFileName << endl;
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

