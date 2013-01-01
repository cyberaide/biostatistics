#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */

#include "kmeans.h"

int main(int argc, char **argv) {

   int     numClusters, numCoords, numObjs;
   int    *membership;    /* [numObjs] */
   char   *filename;
   float **objects;       /* [numObjs][numCoords] data objects */
   float **clusters;      /* [numClusters][numCoords] cluster center */
   float   threshold;
   int     loop_iterations;

    threshold        = 0.001;
    numClusters      = 0;
    isBinaryFile     = 0;
    is_output_timing = 0;
    filename         = NULL;

    if (filename == 0 || numClusters <= 1) usage(argv[0], threshold);

    objects = file_read(0, filename, &numObjs, &numCoords);
    if (objects == NULL) exit(1);

    membership = (int*) malloc(numObjs * sizeof(int));
    assert(membership != NULL);

    clusters = cuda_kmeans(objects, numCoords, numObjs, numClusters, threshold,
                          membership, &loop_iterations);
    free(objects[0]);
    free(objects);

    file_write(filename, numClusters, numObjs, numCoords, clusters,
               membership);

    free(membership);
    free(clusters[0]);
    free(clusters);

    return(0);
}
