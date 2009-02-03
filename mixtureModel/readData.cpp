/*
 * readData.cpp
 *
 *  Created on: Nov 4, 2008
 *      Author: Doug Roberts
 *      Modified by: Andrew Pangborn
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

extern "C"
float* readData(char* f, int* ndims, int* nevents);

float* readData(char* f, int* ndims, int* nevents) {
    string line1;
    ifstream file(f);
    vector<string> lines;
    int num_dims = 0;
    char* temp;
    float* data;

    if (file.is_open()) {
        while(!file.eof()) {
            getline(file, line1);

            if (!line1.empty()) {
                lines.push_back(line1);
            }
        }

        file.close();
    }
    else {
        cout << "Unable to read the file " << f << endl;
        return NULL;
    }
    
    if(lines.size() > 0) {
        line1 = lines[0];
        string line2 (line1.begin(), line1.end());

        temp = strtok((char*)line1.c_str(), " ");

        while(temp != NULL) {
            num_dims++;
            temp = strtok(NULL, " ");
        }

        int num_events = (int)lines.size();

        // Allocate space for all the FCS data
        data = (float*)malloc(sizeof(float) * num_dims * num_events);
        if(!data){
            printf("Cannot allocate enough memory for FCS data.\n");
            return NULL;
        }

        for (int i = 0; i < num_events; i++) {
            temp = strtok((char*)lines[i].c_str(), " ");

            for (int j = 0; j < num_dims; j++) {
                if(temp == NULL) {
                    free(data);
                    return NULL;
                }
                data[i * num_dims + j] = atof(temp);
                temp = strtok(NULL, " ");
            }
        }

        *ndims = num_dims;
        *nevents = num_events;

        return data;    
    } else {
        return NULL;
    }
    
    
}
