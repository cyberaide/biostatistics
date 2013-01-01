#include <stdio.h>
#include <stdlib.h>
#include <string.h>     
#include <sys/types.h> 
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* read(), close() */

#include "kmeans.h"
#define MAX_CHAR_PER_LINE 128

float** file_read(int   isBinaryFile,  
                  char *filename,     
                  int  *numObjs,      
                  int  *numCoords)     
{
    float **objects;
    int     i, j, len;
    ssize_t numBytesRead;

        FILE *infile;
        char *line, *ret;
        int   lineLen;

        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file %s\n", filename);
            return NULL;
        }

        /* first find the number of objects */
        lineLen = MAX_CHAR_PER_LINE;
        line = (char*) malloc(lineLen);
        assert(line != NULL);

        (*numObjs) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            while (strlen(line) == lineLen-1) {
                len = strlen(line);
                fseek(infile, -len, SEEK_CUR);

                lineLen += MAX_CHAR_PER_LINE;
                line = (char*) realloc(line, lineLen);
                assert(line != NULL);

                ret = fgets(line, lineLen, infile);
                assert(ret != NULL);
            }

            if (strtok(line, " \t\n") != 0)
                (*numObjs)++;
        }
        rewind(infile);

        (*numCoords) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                /* ignore the id (first coordiinate): numCoords = 1; */
                while (strtok(NULL, " ,\t\n") != NULL) (*numCoords)++;
                break; /* this makes read from 1st object */
            }
        }
        rewind(infile);
        printf("File %s numObjs   = %d\n",filename,*numObjs);
        printf("File %s numCoords = %d\n",filename,*numCoords);

        len = (*numObjs) * (*numCoords);
        objects    = (float**)malloc((*numObjs) * sizeof(float*));
        assert(objects != NULL);
        objects[0] = (float*) malloc(len * sizeof(float));
        assert(objects[0] != NULL);
        for (i=1; i<(*numObjs); i++)
            objects[i] = objects[i-1] + (*numCoords);

        i = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue;
            for (j=0; j<(*numCoords); j++)
                objects[i][j] = atof(strtok(NULL, " ,\t\n"));
            i++;
        }
        fclose(infile);
        free(line);
    return objects;
}

int file_write(char      *filename,    
               int        numClusters, 
               int        numObjs,     
               int        numCoords,    
               float    **clusters,     
               int       *membership)  
{
    FILE *fptr;
    int   i, j;
    char  outFileName[1024];

    sprintf(outFileName, "%s.cluster_centres", filename);
    fptr = fopen(outFileName, "w");
    for (i=0; i<numClusters; i++) {
        fprintf(fptr, "%d ", i);
        for (j=0; j<numCoords; j++)
            fprintf(fptr, "%f ", clusters[i][j]);
        fprintf(fptr, "\n");
    }
    fclose(fptr);

    sprintf(outFileName, "%s.membership", filename);
    fptr = fopen(outFileName, "w");
    for (i=0; i<numObjs; i++)
        fprintf(fptr, "%d %d\n", i, membership[i]);
    fclose(fptr);

    return 1;
}
