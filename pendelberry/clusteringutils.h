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
#include <unistd.h>
#include <conio.h>   /* added by sid */
#include <process.h> /* added by sid */
//#include <algorithm>

using namespace std;

struct Params {
	int 		numEvents;			// What is this?  Number of points???
	int 		numDimensions;		// p in the paper.
	int 		numClusters;		// Set by user.
	int 		fuzziness;			// User params also.
	int 		maxLikelihood;		// True or false.
	float 		threshold;			// Epsilon == max acceptable change 
	float 		newNorm;			// ???
	float 		oldNorm;			// ???
	float* 		data;				// NxP matrix of point coordinates
	float* 		centers;			// Nx<numClusters> coords of centers.
	float* 		membership;			// Px<numclusters> values that tell
									// how connected each point is to each
									// cluster center.
	float*		membership2;		// Copy of membership.
	
	int*		Ti;					// N data points that are set if a point has any
									// negative membership.
};


extern "C" void 	initRand();
extern "C" float 	randFloat();
extern "C" float 	randFloatRange(float min, float max);
extern "C" int 		randInt(int max);
extern "C" int 		randIntRange(int min, int max);
extern "C" int 		getRandIndex(int w, int h);

extern "C" bool 	contains(Params* p, float points[]);
extern "C" void 	setCenters(Params* p);
extern "C" void 	setCentersNew(Params* p);
extern "C" void 	getPoints(Params* p, float points[], int i);

extern "C" void 	readData(char* f, Params* p);
extern "C" void 	writeData(Params* p, const char* f);

extern "C" int 		clusterColor(float i, int nc);
extern "C" string 	generateOutputFileName(int nc);

void initRand() {
	int seed = (int)time(0) * (int)getpid();
	// srand((unsigned)seed);	// REMOVED for testing
	srand((unsigned) 42);  // 42 ADDED For Repeatability.
}

float randFloat() {
	return rand() / (float(RAND_MAX) + 1);
}

float randFloatRange(float min, float max) {
	if (min > max) {
		return randFloat() * (min - max) + max;
	}
	else {
		return randFloat() * (max - min) + min;
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
bool contains(Params* p, float points[]) {
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

// Given the clusters, find the center?
void setCenters(Params* p) {
	float *temp=new float[p->numDimensions];

	for (int i = 0; i < p->numClusters; i++) {
		getPoints(p, temp, randIntRange(0, p->numEvents));

		if (i != 0) {
			while (contains(p, temp)) {
				getPoints(p, temp, randIntRange(0, p->numEvents));
			}
		}

		for (int j = 0; j < p->numDimensions; j++) {
			p->centers[j + i * p->numDimensions] = temp[j];
		}
	}
}

// Get the coordinates of a point?
void getPoints(Params* p, float points[], int i) {
	for (int j = 0; j < p->numDimensions; j++) {
		points[j] = p->data[j + i * p->numDimensions];
	}
}

// Read in the file named "f"
void readData(char* f, Params* p) {
	string line1;
	ifstream file(f);			// input file stream
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
		return;
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

	p->data = (float*)malloc(sizeof(float) * p->numDimensions * p->numEvents);
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
}

void writeData(Params* p, const char* f) {
	ofstream file;
	int precision = 5;

	file.open(f);

	file << "Data: last " << p->numClusters << " columns indicate cluster membership." << endl << endl;

	for (int i = 0; i < p->numEvents; i++) {
		for (int j = 0; j < p->numDimensions; j++) {
			file << fixed << setprecision(precision) << p->data[j + i * p->numDimensions] << " ";
		}

		for (int j = 0; j < p->numClusters; j++) {
			file << fixed << setprecision(precision) << p->membership[j + i * p->numClusters] << " ";
		}

		file << endl;
	}

	//int identity[p->numClusters][p->numClusters];
	int **identity = new int*[p->numClusters];
	for(int i = 0; i < p->numClusters; i++)
	{
       identity[i] = new int[p->numClusters];
	}

	for (int i = 0; i < p->numClusters; i++) {
		for (int j = 0; j < p->numClusters; j++) {
			if (i == j) {
				identity[i][j] = 1;
			}
			else {
				identity[i][j] = 0;
			}
		}
	}

	file << endl << "Cluster Centers: last " << p->numClusters << " columns is the identity matrix." << endl << endl;

	for (int i = 0; i < p->numClusters; i++) {
		for (int j = 0; j < p->numDimensions; j++) {
			file << fixed << setprecision(precision) << p->centers[j + i * p->numDimensions] << " ";
		}

		for (int j = 0; j < p->numClusters; j++) {
			file << identity[i][j] << " ";
		}

		file << endl;
	}

	file.close();
}

int clusterColor(float i, int nc) {
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


double determinant(double **a, int n) {
/* added by sid and borrowed from http://local.wasp.uwa.edu.au/~pbourke/miscellaneous/determinant/determinant.c */ 
//==============================================================================
// Recursive definition of determinate using expansion by minors.
//
// Notes: 1) arguments:
//             a (double **) pointer to a pointer of an arbitrary square matrix
//             n (int) dimension of the square matrix
//
//        2) Determinant is a recursive function, calling itself repeatedly
//           each time with a sub-matrix of the original till a terminal
//           2X2 matrix is achieved and a simple determinat can be computed.
//           As the recursion works backwards, cumulative determinants are
//           found till untimately, the final determinate is returned to the
//           initial function caller.
//
//        3) m is a matrix (4X4 in example)  and m13 is a minor of it.
//           A minor of m is a 3X3 in which a row and column of values
//           had been excluded.   Another minor of the submartix is also
//           possible etc.
//             m  a b c d   m13 . . . .
//                e f g h       e f . h     row 1 column 3 is elminated
//                i j k l       i j . l     creating a 3 X 3 sub martix
//                m n o p       m n . p
//
//        4) the following function finds the determinant of a matrix
//           by recursively minor-ing a row and column, each time reducing
//           the sub-matrix by one row/column.  When a 2X2 matrix is
//           obtained, the determinat is a simple calculation and the
//           process of unstacking previous recursive calls begins.
//
//                m n
//                o p  determinant = m*p - n*o
//
//        5) this function uses dynamic memory allocation on each call to
//           build a m X m matrix  this requires **  and * pointer variables
//           First memory allocation is ** and gets space for a list of other
//           pointers filled in by the second call to malloc.
//
//        6) C++ implements two dimensional arrays as an array of arrays
//           thus two dynamic malloc's are needed and have corresponsing
//           free() calles.
//
//        7) the final determinant value is the sum of sub determinants
//
//==============================================================================


    int i,j,j1,j2 ;                    // general loop and matrix subscripts
    double det = 0 ;                   // init determinant
    double **m = NULL ;                // pointer to pointers to implement 2d
                                       // square array

    if (n < 1)    {   }                // error condition, should never get here

    else if (n == 1) {                 // should not get here
        det = a[0][0] ;
        }

    else if (n == 2)  {                // basic 2X2 sub-matrix determinate
                                       // definition. When n==2, this ends the
        det = a[0][0] * a[1][1] - a[1][0] * a[0][1] ;// the recursion series
        }


                                       // recursion continues, solve next sub-matrix
    else {                             // solve the next minor by building a
                                       // sub matrix
        det = 0 ;                      // initialize determinant of sub-matrix

                                           // for each column in sub-matrix
        for (j1 = 0 ; j1 < n ; j1++) {
                                           // get space for the pointer list
            m = (double **) malloc((n-1)* sizeof(double *)) ;

            for (i = 0 ; i < n-1 ; i++)
                m[i] = (double *) malloc((n-1)* sizeof(double)) ;
                       //     i[0][1][2][3]  first malloc
                       //  m -> +  +  +  +   space for 4 pointers
                       //       |  |  |  |          j  second malloc
                       //       |  |  |  +-> _ _ _ [0] pointers to
                       //       |  |  +----> _ _ _ [1] and memory for
                       //       |  +-------> _ a _ [2] 4 doubles
                       //       +----------> _ _ _ [3]
                       //
                       //                   a[1][2]
                      // build sub-matrix with minor elements excluded
            for (i = 1 ; i < n ; i++) {
                j2 = 0 ;               // start at first sum-matrix column position
                                       // loop to copy source matrix less one column
                for (j = 0 ; j < n ; j++) {
                    if (j == j1) continue ; // don't copy the minor column element

                    m[i-1][j2] = a[i][j] ;  // copy source element into new sub-matrix
                                            // i-1 because new sub-matrix is one row
                                            // (and column) smaller with excluded minors
                    j2++ ;                  // move to next sub-matrix column position
                    }
                }

            det += pow(-1.0,1.0 + j1 + 1.0) * a[0][j1] * determinant(m,n-1) ;
                                            // sum x raised to y power
                                            // recursively get determinant of next
                                            // sub-matrix which is now one
                                            // row & column smaller

            for (i = 0 ; i < n-1 ; i++) free(m[i]) ;// free the storage allocated to
                                            // to this minor's set of pointers
            free(m) ;                       // free the storage for the original
                                            // pointer to pointer
        }
    }
    return(det) ;
}



#endif /* CLUSTERINGUTILS_H_ */
