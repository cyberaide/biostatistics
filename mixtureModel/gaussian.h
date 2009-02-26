/*
 * Parameters file for gaussian mixture model based clustering application
 */

#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#define MAX_CLUSTERS	64
#define PI  3.141593
#define	NUM_BLOCKS 24
#define NUM_THREADS 64
#define NUM_DIMENSIONS 21

#define VERBOSE 1
#define EMU 0

typedef struct 
{
    float N;        // expected # of pixels in cluster
    float pi;       // probability of cluster in GMM
    float *means;   // Spectral mean for the cluster
    float *R;      // Covariance matrix
    float *Rinv;   // Inverse of covariance matrix
    float avgvar;    // average variance
    float constant; // Normalizing constant
    float *p;       // Probability that each pixel belongs to this cluster
} cluster;

int validateArguments(int argc, char** argv, int* num_clusters, FILE** infile, FILE** outfile);
void printUsage(char** argv);
#endif

