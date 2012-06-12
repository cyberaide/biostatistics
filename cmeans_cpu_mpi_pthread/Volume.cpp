#include <stdlib.h>
#include <float.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include "cmeansPthreadMPICPU.h"

using namespace std;
extern int NUM_EVENTS;

void ReportSummary(float* clusters, int count, char* inFileName){
    ofstream myfile;
    
    char logFileName [512];
    sprintf(logFileName, "%s.summary", inFileName);
    cout << "Log file name = " << logFileName << endl;
    myfile.open(logFileName);
    myfile << "Cluster Centers:" << endl;
    for(int i = 0; i < count; i ++){
        for(int j = 0; j < NUM_DIMENSIONS; j++){
            myfile << clusters[i*NUM_DIMENSIONS + j] << "\t";
        }
        myfile << endl;
    }
    myfile.close();

}

void ReportBinaryResults(float* events, float* memberships, int count, char* inFileName){
    FILE* myfile;
    char logFileName [512];
    sprintf(logFileName, "%s.results", inFileName);
    cout << "Results Log file name = " << logFileName << endl;
    myfile = fopen(logFileName,"wb");

    for(int i = 0; i < NUM_EVENTS; i++){
        for(int j = 0; j < NUM_DIMENSIONS; j++){
            fwrite(&events[i*NUM_DIMENSIONS + j],4,1,myfile);
        }
        for(int j = 0; j < count; j++){
            fwrite(&memberships[j*NUM_EVENTS+i],4,1,myfile); 
        }
    }
    //fwrite(events,4,NUM_EVENTS*NUM_DIMENSIONS,myfile);
    //fwrite(memberships,4,NUM_EVENTS*NUM_CLUSTERS,myfile);
    fclose(myfile);
}

void ReportResults(float* events, float* memberships, int count, char* inFileName){
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
            myfile << memberships[j*NUM_EVENTS+i] << ","; 
        }
        myfile << memberships[(count-1)*NUM_EVENTS+i]; 
        myfile << endl;
    }
    myfile.close();
}

