/*
 * test.h
 *
 *  Created on: Mar 16, 2009
 *      Author: doug 
 *
 *  Modified on: July 27, 2009
 *      Author: sid pendelberry
 *  Change: added determinant code
 * 
 */

#ifndef CLUSTERINGUTILS_H_
#define CLUSTERINGUTILS_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/types.h>

#ifdef WINDOWS
#include <conio.h>   //Windows only
#include <process.h> //Windows only
#endif

// Input file delimiter, usually "," or "\t" or " " (comma tab or space)
#define DELIMITER "," 

using namespace std;

struct Params {
    int         numEvents;          // Number of rows
    int         numDimensions;      // p in the paper.
    int         numClusters;        // Set by user.
    int         option;             // Specify which algorithm
	int			setup;				// How the matrix is initialized
	int			innerloop;			// How many passes through the inner loop for Ti -- stopping condition
    double      threshold;          // Epsilon == max acceptable change 
    double*     data;               // NxP matrix of point coordinates
    double*     centers;            // Nx<numClusters> coords of centers.
    double*     membership;         // Px<numclusters> values that tell
                                    // how connected each point is to each
                                    // cluster center.
    double*     membership2;        // Copy of membership.
    double*     scatters;           // Scatter matrices. One N*N matrix per cluster, dynamically allocated
    double*     scatter_inverses;   // Inver of all scatter matrices
    int*        Ti;                 // N data points that are set if a point has any
                                    // negative membership.
    double*     n;                  // effective size of each cluster (sum of fuzzy memberships for that cluster)
    double*     A_t;                // Used for eqn (31)
    
    double*      determinants;      // determinants of scatter matrices 
    
    double      beta;
    double      tau;
};

void    initRand();
double  randdouble();
double  randdoubleRange(double min, double max);
int     randInt(int max);
int     randIntRange(int min, int max);
int     getRandIndex(int w, int h);

bool    contains(Params* p, double points[]);
void    setCenters(Params* p);
void    getPoints(Params* p, double points[], int i);
void    printCenters(Params* p);
void    printScatters(Params* p);

void    allocateParamArrays(Params* p);
int    readData(char* f, Params* p);
void    writeData(Params* p, const char* f);

int     clusterColor(double i, int nc);
string  generateOutputFileName(int nc);
void invert_cpu(double* data, int actualsize, double* determinant);


#endif /* CLUSTERINGUTILS_H_ */
