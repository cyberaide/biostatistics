#include <stdlib.h>
#include <float.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cutil.h>

#include "cmeansMultiGPU.h"

using namespace std;

void ReportSummary(float* clusters, int count, char* inFileName){
    ofstream myfile;
    
    char logFileName [512];
    sprintf(logFileName, "%s.summary", inFileName);
    cout << "Log file name = " << logFileName << endl;
    myfile.open(logFileName);
    for(int i = 0; i < count; i ++){
        myfile << "Cluster " << i << ": ";
        for(int j = 0; j < NUM_DIMENSIONS; j++){
            myfile << clusters[i*NUM_DIMENSIONS + j] << "\t";
        }
        myfile << endl;
    }
    myfile.close();

}

void ReportResults(float* events, float* clusters, int count, char* inFileName){
    ofstream myfile;
    char logFileName [512];
    sprintf(logFileName, "%s.results", inFileName);
    cout << "Results Log file name = " << logFileName << endl;
    myfile.open(logFileName);
    

    for(int i = 0; i < NUM_EVENTS; i++){
        for(int j = 0; j < NUM_DIMENSIONS-1; j++){
            myfile << events[i*NUM_DIMENSIONS + j] << ",";
        }
        myfile << events[i*NUM_DIMENSIONS + NUM_DIMENSIONS - 1];
        myfile << "\t";
        for(int j = 0; j < count-1; j++){
            myfile << MembershipValueReduced(clusters, events, j, i, count) << ","; 
        }
        myfile << MembershipValueReduced(clusters, events, count-1, i, count);
        myfile << endl;
        
    }
    for(int i = 0; i < count; i++){
        for(int j = 0; j < NUM_DIMENSIONS-1; j++){
            myfile << clusters[i*NUM_DIMENSIONS + j] << ",";
        }
        myfile << clusters[i*NUM_DIMENSIONS+NUM_DIMENSIONS-1];
        myfile << "\t";
        for(int j = 0; j < count; j++){
            if(j == i)
                myfile << 1; 
            else
                myfile << 0;

            if(j < (count-1)) {
                myfile << ",";
            }
        }
        myfile << endl;
    }
    myfile.close();

}
__host__ float CalculateDistanceCPU(const float* clusters, const float* events, int clusterIndex, int eventIndex){

    float sum = 0;
    #if DISTANCE_MEASURE == 0
        for(int i = 0; i < NUM_DIMENSIONS; i++){
            float tmp = events[eventIndex*NUM_DIMENSIONS + i] - clusters[clusterIndex*NUM_DIMENSIONS + i];
            sum += tmp*tmp;
        }
        sum = sqrt(sum);
    #elif DISTANCE_MEASURE == 1
        for(int i = 0; i < NUM_DIMENSIONS; i++){
            float tmp = events[eventIndex*NUM_DIMENSIONS + i] - clusters[clusterIndex*NUM_DIMENSIONS + i];
            sum += abs(tmp);
        }
    #elif DISTANCE_MEASURE == 2
        for(int i = 0; i < NUM_DIMENSIONS; i++){
            float tmp = abs(events[eventIndex*NUM_DIMENSIONS + i] - clusters[clusterIndex*NUM_DIMENSIONS + i]);
            if(tmp > sum)
                sum = tmp;
        }
    #endif
    return sum;
}

float MembershipValueReduced(const float* clusters, const float* events, int clusterIndex, int eventIndex, int validClusters){
    float myClustDist = CalculateDistanceCPU(clusters, events, clusterIndex, eventIndex);
    float sum =0;
    float otherClustDist;
    for(int j = 0; j< validClusters; j++){
        otherClustDist = CalculateDistanceCPU(clusters, events, j, eventIndex); 
        if(otherClustDist < .000001)
            return 0.0;
        sum += pow((float)(myClustDist/otherClustDist),float(2/(FUZZINESS-1)));
    }
    return 1/sum;
}
