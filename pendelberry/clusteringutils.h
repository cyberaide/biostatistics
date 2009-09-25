/*
 * test.h
 *
 *  Created on: Mar 16, 2009
 *      Author: doug 
 *
 *  Modified on: July 27, 2009
 *      Author: sid pendelberry
 *  Change: added determinant code
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
//#include <unistd.h>  // unix only
//#include <conio.h>   /* added by sid */
//#include <process.h> /* added by sid */
//#include <algorithm>

using namespace std;

struct Params {
    int         numEvents;          // What is this?  Number of points???
    int         numDimensions;      // p in the paper.
    int         numClusters;        // Set by user.
    //int       fuzziness;          // User params also.
    int         option;            // Specify which algorithm
    double      threshold;          // Epsilon == max acceptable change 
    double*     data;               // NxP matrix of point coordinates
    double*     centers;            // Nx<numClusters> coords of centers.
    double*     membership;         // Px<numclusters> values that tell
                                    // how connected each point is to each
                                    // cluster center.
    double*     membership2;        // Copy of membership.
    double*     scatters;           // Scatter matrices. One N*N matrix per cluster, dynamically allocated
    double*     scatter_inverses;    // Inver of all scatter matrices
    int*        Ti;                 // N data points that are set if a point has any
                                    // negative membership.
    double       newNorm;
    double       oldNorm;
    
    double*     n;                   // effective size of each cluster (sum of fuzzy memberships for that cluster)
    double*     A_t;                // Used for eqn (31)
    
    double*      determinants;       // determinants of scatter matrices 
    //double*      means              // multidimensional mean for every cluster
    
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

void    allocateParamArrays(Params* p);
int    readData(char* f, Params* p);
void    writeData(Params* p, const char* f);

int     clusterColor(double i, int nc);
string  generateOutputFileName(int nc);
void invert_cpu(double* data, int actualsize, double* determinant);

void printCenters(Params* p) {
    for(int c=0;c<p->numClusters;c++) {
        cout << "Cluster Center #" << c << ": ";
        for(int d=0; d<p->numDimensions;d++){
            cout << p->centers[c*p->numDimensions+d] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void initRand() {
    //int seed = (int)time(0) * (int)getpid();
    srand((int)getpid());   // REMOVED for testing
    //srand((unsigned) 42);  // 42 ADDED For Repeatability.
}

double randdouble() {
    return rand() / (double(RAND_MAX) + 1);
}

double randdoubleRange(double min, double max) {
    if (min > max) {
        return randdouble() * (min - max) + max;
    }
    else {
        return randdouble() * (max - min) + min;
    }
}

int randInt(int max) {
    return int(rand() % max) + 1;
}

int randIntRange(int min, int max) {
    if (min > max) {
        return max + int(rand() % (min - max));
    }
    else {
        return min + int(rand() % (max - min));
    }
}

int getRandIndex(int w, int h) {
    return (randIntRange(0, w) - 1) + (randIntRange(0, h) - 1) * w;
}

// What does this do?
// Is it convex hull or the cluster
bool contains(Params* p, double points[]) {
    int count = 1;

    for (int i = 0; i < p->numClusters; i++) {
        for (int j = 0; j < p->numDimensions; j++) {
            if (p->centers[j + i * p->numDimensions] == points[j]) {
                count++;
            }
        }
    }

    if (count == p->numDimensions) {
        return true;
    }
    else {
        return false;
    }
}


// Get the coordinates of a point
void getPoints(Params* p, double points[], int i) {
    for (int j = 0; j < p->numDimensions; j++) {
        points[j] = p->data[j + i * p->numDimensions];
    }
}

// Initializes and allocates memory for all arrays of the Params structure
// Requires numDimensions, numClusters, numEvents to be defined (by readData)
void allocateParamArrays(Params* p) {
    p->centers = new double[p->numClusters*p->numDimensions];
    p->membership = new double[p->numClusters*p->numEvents];
    p->membership2 = new double[p->numClusters*p->numEvents];
    p->scatters = new double[p->numClusters*p->numDimensions*p->numDimensions];
    p->scatter_inverses = new double[p->numClusters*p->numDimensions*p->numDimensions];
    p->determinants = new double[p->numClusters];
    p->Ti = new int[p->numEvents*p->numClusters];
    p->n = new double[p->numClusters];
    p->A_t = new double[p->numClusters];
    //p->means = new double[p->numClusters*p->numDimensions];
}

// Read in the file named "f"
int readData(char* f, Params* p) {
    string line1;
    ifstream file(f);           // input file stream
    vector<string> lines;
    int dim = 0;
    char* temp;

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
        return -1;
    }

    line1 = lines[0];
    string line2 (line1.begin(), line1.end());

    temp = strtok((char*)line1.c_str(), " ");

    while(temp != NULL) {
        dim++;
        temp = strtok(NULL, " ");
    }

    p->numDimensions = dim;
    p->numEvents = (int)lines.size();

    p->data = (double*)malloc(sizeof(double) * p->numDimensions * p->numEvents);
    temp = strtok((char*)line2.c_str(), " ");

    for (int i = 0; i < p->numEvents; i++) {
        if (i != 0) {
            temp = strtok((char*)lines[i].c_str(), " ");
        }

        for (int j = 0; j < p->numDimensions && temp != NULL; j++) {
            p->data[j + i * p->numDimensions] = atof(temp);
            temp = strtok(NULL, " ");
        }
    }
    return 0;
}

void writeData(Params* p, const char* f) {
    ofstream file;
    ofstream summary;
    int precision = 5;

    file.open(f);

    for (int i = 0; i < p->numEvents; i++) {
        for (int j = 0; j < p->numDimensions; j++) {
            file << fixed << setprecision(precision) << p->data[j + i * p->numDimensions]; 
            if(j < p->numDimensions-1) {
                file << ",";
            }
        }
        file << "\t";
        for (int j = 0; j < p->numClusters; j++) {
            file << fixed << setprecision(precision) << p->membership[j + i * p->numClusters];
            if(j < p->numClusters-1) {
                file << ",";
            }
        }
        file << endl;
    }

    file.close();
    
    summary.open("output.summary");
    for (int t = 0; t < p->numClusters; t++) {
        summary << "Cluster #" << t << endl;
        summary << "Probability: " << p->n[t]/p->numEvents << endl;
        summary << "N: " << p->n[t] << endl;
        summary << "Means: ";
        for(int d=0; d < p->numDimensions; d++) {
            summary << p->centers[t*p->numDimensions+d] << " ";
        }
        summary << endl << endl;
        summary << "R Matrix:" << endl;
        for(int i=0; i< p->numDimensions; i++) {
            for(int j=0; j< p->numDimensions; j++) {
                summary << p->scatters[t*p->numDimensions*p->numDimensions+ i*p->numDimensions + j] << " ";
            }
            summary << endl;
        } 
        summary << endl << endl;
    }
}

int clusterColor(double i, int nc) {
    return (int)((i / nc) * 256);
}

string generateOutputFileName(int nc) {
    string output;
    time_t rawtime;
    struct tm *timeinfo;
    int i;
    char ch[50];

    output = "../../../output/output";
    //output = "../output/output";

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    i = 1 + timeinfo->tm_mon;
    sprintf(ch, "-%d", i);
    output.append(ch);

    i = timeinfo->tm_mday;
    sprintf(ch, "-%d", i);
    output.append(ch);

    i = 1900 + timeinfo->tm_year;
    sprintf(ch, "-%d", i);
    output.append(ch);

    i = timeinfo->tm_hour;
    sprintf(ch, "-%d", i);
    output.append(ch);

    i = timeinfo->tm_min;
    sprintf(ch, "-%d", i);
    output.append(ch);

    sprintf(ch, "_%d.dat", nc);
    output.append(ch);

    return output;
}

/*
 *  * Inverts a square matrix (stored as a 1D double array)
 *   * 
 *    * actualsize - the dimension of the matrix
 *     *
 *      * written by Mike Dinolfo 12/98
 *       * version 1.0
 *        */
void invert_cpu(double* data, int actualsize, double* determinant)  {
    int maxsize = actualsize;
    int n = actualsize;
    *determinant = 1.0;

    /*printf("\n\nR matrix before inversion:\n");
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            printf("%.4f ",data[i*n+j]);
        }
        printf("\n");
    }*/
    
  if (actualsize <= 0) return;  // sanity check
  if (actualsize == 1) return;  // must be of dimension >= 2
  for (int i=1; i < actualsize; i++) data[i] /= data[0]; // normalize row 0
  for (int i=1; i < actualsize; i++)  { 
    for (int j=i; j < actualsize; j++)  { // do a column of L
      double sum = 0.0;
      for (int k = 0; k < i; k++)  
          sum += data[j*maxsize+k] * data[k*maxsize+i];
      data[j*maxsize+i] -= sum;
      }
    if (i == actualsize-1) continue;
    for (int j=i+1; j < actualsize; j++)  {  // do a row of U
      double sum = 0.0;
      for (int k = 0; k < i; k++)
          sum += data[i*maxsize+k]*data[k*maxsize+j];
      data[i*maxsize+j] = 
         (data[i*maxsize+j]-sum) / data[i*maxsize+i];
      }
    }
    
    for(int i=0; i<actualsize; i++) {
        *determinant *= data[i*n+i];
    }
    
  for ( int i = 0; i < actualsize; i++ )  // invert L
    for ( int j = i; j < actualsize; j++ )  {
      double x = 1.0;
      if ( i != j ) {
        x = 0.0;
        for ( int k = i; k < j; k++ ) 
            x -= data[j*maxsize+k]*data[k*maxsize+i];
        }
      data[j*maxsize+i] = x / data[j*maxsize+j];
      }
  for ( int i = 0; i < actualsize; i++ )   // invert U
    for ( int j = i; j < actualsize; j++ )  {
      if ( i == j ) continue;
      double sum = 0.0;
      for ( int k = i; k < j; k++ )
          sum += data[k*maxsize+j]*( (i==k) ? 1.0 : data[i*maxsize+k] );
      data[i*maxsize+j] = -sum;
      }
  for ( int i = 0; i < actualsize; i++ )   // final inversion
    for ( int j = 0; j < actualsize; j++ )  {
      double sum = 0.0;
      for ( int k = ((i>j)?i:j); k < actualsize; k++ )  
          sum += ((j==k)?1.0:data[j*maxsize+k])*data[k*maxsize+i];
      data[j*maxsize+i] = sum;
      }
      
    /*
      printf("\n\nR matrix after inversion:\n");
      for(int i=0; i<n; i++) {
          for(int j=0; j<n; j++) {
              printf("%.4f ",data[i*n+j]);
          }
          printf("\n");
      }
    */
 }

#endif /* CLUSTERINGUTILS_H_ */
