/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* Template project which demonstrates the basics on how to setup a project 
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>
#include "gaussian.h"

// includes, kernels
#include <theta_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest( int argc, char** argv);

extern "C"
void computeGold( float* reference, float* idata, const unsigned int len);

extern "C"
float* readData(char* f, int* ndims, int*nevents);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) {
    runTest( argc, argv);

    //CUT_EXIT(argc, argv);
}

///////////////////////////////////////////////////////////////////////////////
// Validate command line arguments
///////////////////////////////////////////////////////////////////////////////
int validateArguments(int argc, char** argv, int* num_clusters) {
    if(argc <= 4 && argc >= 3) {
        // parse num_clusters
        if(!sscanf(argv[1],"%d",num_clusters)) {
            printf("Invalid number of starting clusters\n\n");
            printUsage(argv);
            return 1;
        } 
        
        // Check bounds for num_clusters
        if(*num_clusters < 1 || *num_clusters > MAX_CLUSTERS) {
            printf("Invalid number of starting clusters\n\n");
            printUsage(argv);
            return 1;
        }
        
        // parse infile
        FILE* infile = fopen(argv[2],"r");
        if(!infile) {
            printf("Invalid infile.\n\n");
            printUsage(argv);
            return 2;
        } 
        
        // parse outfile
        FILE* outfile = fopen(argv[3],"w");
        if(!outfile) {
            printf("Unable to create output file.\n\n");
            printUsage(argv);
            return 3;
        }
        
        // Clean up so the EPA is happy
        fclose(infile);
        fclose(outfile);
        return 0;
    } else {
        printUsage(argv);
        return 1;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Print usage statement
///////////////////////////////////////////////////////////////////////////////
void printUsage(char** argv)
{
   printf("Usage: %s num_clusters infile [outfile]\n",argv[0]);
   printf("\t num_clusters: The number of starting clusters\n");
   printf("\t infile: ASCII space-delimited FCS data file\n");
   printf("\t outfile: Clustering results output file\n");
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
int
runTest( int argc, char** argv) 
{
    
    int num_clusters;
    
    int error = validateArguments(argc,argv,&num_clusters);
    
    // Don't continue if we had a problem with the program arguments
    if(error) {
        return 1;
    }
    
    int num_dimensions;
    int num_events;
        
    float* fcs_data = readData(argv[2],&num_dimensions,&num_events);
    
    if(!fcs_data) {
        printf("Error parsing input file. This could be due to an empty file ");
        printf("or an inconsistent number of dimensions. Aborting.\n");
        return 1;
    }
    
    printf("Number of events: %d\n",num_events);
    printf("Number of dimensions: %d\n\n",num_dimensions);
    
    CUT_DEVICE_INIT(argc, argv);

    unsigned int timer = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    
    // print the input
    for( unsigned int i = 0; i < num_events*num_dimensions; i += num_dimensions ) 
    {
        for(unsigned int j = 0; j < num_dimensions; j++) {
            printf("%f ",fcs_data[i+j]);
        }
        printf("\n");
    }

    unsigned int mem_size = num_dimensions*num_events*sizeof(float);
    unsigned int num_threads = num_dimensions;
    
    // allocate device memory
    float* d_idata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, mem_size));
    // copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( d_idata, fcs_data, mem_size,
                                cudaMemcpyHostToDevice) );

    // allocate device memory for result
    float* d_odata;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, num_dimensions*sizeof(float)));

    // setup execution parameters
    dim3  grid( 1, 1, 1);
    dim3  threads( num_threads, 1, 1);

    // execute the kernel
    testKernel<<< 1, num_threads >>>( d_idata, d_odata, num_dimensions, num_events);

    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc(num_dimensions*sizeof(float));
    // copy result from device to host
    CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata, sizeof(float) * num_dimensions,
                                cudaMemcpyDeviceToHost) );

    CUT_SAFE_CALL(cutStopTimer(timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    CUT_SAFE_CALL(cutDeleteTimer(timer));
    
    printf("Spectral Mean: ");
    for(int i=0; i<num_dimensions; i++){
        printf("%f ",h_odata[i]);
    }
    printf("\n");
    
    // cleanup memory
    free(fcs_data);
    free(h_odata);
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_odata));
    
    return 0;
}
