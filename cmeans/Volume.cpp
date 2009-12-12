#include <stdlib.h>

#ifdef MULTI_GPU
    #include "cmeansMultiGPU.h"
#else
    #include "cmeans.h"
#endif

#include <float.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cutil.h>
//#include <cmeans_kernel.cu>

using namespace std;

void FindCharacteristics(float* events, float* clusters, int finalClusterCount, float averageTime, float mdlTime, int numIterations, char* inFileName, clock_t total_start){

    float* clusterVolumes = (float*)malloc(sizeof(float) * finalClusterCount);
    float* clusterDensities = (float*)malloc(sizeof(float) * finalClusterCount);
    float* clusterOccupancies = (float*)malloc(sizeof(float) * finalClusterCount);
    ofstream charFile;  
    char charFileName [512];
    sprintf(charFileName, "%s_volume_log_%d", inFileName, NUM_CLUSTERS);
    cout << "Characteristics Log file name = " << charFileName << endl;
    charFile.open(charFileName);
    fflush(stdout); 
    
    for(unsigned i = 0; i < sizeof(VOLUME_INC_PARAMS)/sizeof(float); i++){
#if !VOLUME_TYPE
        FindBoxCharacteristics(events, clusters, finalClusterCount, clusterVolumes, clusterDensities, clusterOccupancies, i);
#else
        FindSphereCharacteristics(events, clusters, finalClusterCount, clusterVolumes, clusterDensities, clusterOccupancies, i);
#endif
        
        for( int j = 0; j < finalClusterCount; j++){
            charFile << j << ", " << VOLUME_INC_PARAMS[i] << ", " 
                 << clusterVolumes[j] << ", " << clusterDensities[j] 
                 << ", " << clusterOccupancies[j]/NUM_EVENTS << endl;
        }
    }
    charFile.close();

    ReportSummary(clusters, finalClusterCount, inFileName, averageTime, mdlTime, numIterations, total_start);

    ReportResults(events, clusters, finalClusterCount, inFileName);

    free(clusterVolumes);
    free(clusterDensities);
    free(clusterOccupancies);
}

void FindBoxCharacteristics(float* events, float* clusters, int finalClusterCount, float* volume, float* density, float* occupancy, int includeIndex){
    
    float* minArray = (float*)malloc(sizeof(float)*finalClusterCount*NUM_DIMENSIONS);
    float* maxArray = (float*)malloc(sizeof(float)*finalClusterCount*NUM_DIMENSIONS);
    for(int k = 0; k < finalClusterCount*NUM_DIMENSIONS; k++){
        minArray[k] = FLT_MAX;
        maxArray[k] = FLT_MIN;
    }

    for(int k = 0; k < finalClusterCount; k++){
        volume[k] = 0;
        occupancy[k] = 0;
        density[k] = 0; 
        
        for(int i = 0; i < NUM_EVENTS; i++){
            if(MembershipValueReduced(clusters, events, k, i, finalClusterCount) >  VOLUME_INC_PARAMS[includeIndex]){
                // include in volume calculation
                occupancy[k]++;
                for(int j = 0; j < NUM_DIMENSIONS; j++){
                    if(maxArray[k*NUM_DIMENSIONS + j] < events[i*NUM_DIMENSIONS + j]){
                        maxArray[k*NUM_DIMENSIONS + j] = events[i*NUM_DIMENSIONS + j];  
                    }
                    if(minArray[k*NUM_DIMENSIONS + j] > events[i*NUM_DIMENSIONS + j]){
                        minArray[k*NUM_DIMENSIONS + j] = events[i*NUM_DIMENSIONS + j];  
                    }
                }
            }
        }
        volume[k] = maxArray[k*NUM_DIMENSIONS] - minArray[k*NUM_DIMENSIONS];
        for(int i = 1; i < NUM_DIMENSIONS; i++){
            volume[k] *=  (maxArray[k*NUM_DIMENSIONS + i] - minArray[k*NUM_DIMENSIONS + i]);
        }
        density[k] = occupancy[k] / volume[k];
    }
}

void FindSphereCharacteristics(float* events, float* clusters, int finalClusterCount, float* volume, float* density, float* occupancy, int includeIndex){

    float maxDist = 0;
    
    for(int k = 0; k < finalClusterCount; k++){
        volume[k] = 0;
        occupancy[k] = 0;
        density[k] = 0; 
        
        for(int i = 0; i < NUM_EVENTS; i++){
            float distance = CalculateDistanceCPU(clusters, events, k, i);
            if(MembershipValueReduced(clusters, events, k, i, finalClusterCount) > VOLUME_INC_PARAMS[includeIndex]){
                // include in volume calculation
                occupancy[k]++;
                if(distance > maxDist){
                    maxDist = distance;
                }
            }
            
        }
        if(NUM_DIMENSIONS & 1) { // odd d => even sphere
            
            volume[k] = pow(2*PI, ((NUM_DIMENSIONS - 1) >> 1))*pow((maxDist/2), (NUM_DIMENSIONS-1)); 
            int denom = 2;
            for(int i = 4; i < NUM_DIMENSIONS; i+=2){
                denom *= i;
            }
            volume[k] /= denom;
        } else{
            volume[k] = 2*pow(2*PI, ((NUM_DIMENSIONS - 2) >> 1))*pow((maxDist/2), (NUM_DIMENSIONS-1)); 
            int denom = 1;
            for(int i = 3; i < NUM_DIMENSIONS; i+=2){
                denom *= i;
            }
            volume[k] /= denom;
        }
        density[k] = occupancy[k] / volume[k];
    }
}

void ReportSummary(float* clusters, int count, char* inFileName, float averageTime, float mdlTime, int iterations, clock_t total_start){
    ofstream myfile;
    
    char logFileName [512];
    sprintf(logFileName, "%s_summary_log_%d_%d_%d", inFileName, NUM_CLUSTERS,  CPU_ONLY, MDL_on_GPU);
    cout << "Log file name = " << logFileName << endl;
    myfile.open(logFileName);
    float totalTime = (float)(clock() - total_start)/(float)(CLOCKS_PER_SEC);
    myfile << "Average c-means iteration time = " << averageTime 
        << " (ms) for " << iterations << " iterations."
        << "\nMDL matrix generation time = " << mdlTime 
        << " (s)\nTotal Time = " << totalTime << " (s)\n\n";
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
    sprintf(logFileName, "%s_%d_out.txt", inFileName, NUM_CLUSTERS);
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
