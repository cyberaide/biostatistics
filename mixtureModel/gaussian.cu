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
#include "invert_matrix.h"

// includes, kernels
#include <theta_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest( int argc, char** argv);

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
        if(argc == 4) {
            FILE* outfile = fopen(argv[3],"w");
            if(!outfile) {
                printf("Unable to create output file.\n\n");
                printUsage(argv);
                return 3;
            }
            fclose(outfile);
        }
        
        // Clean up so the EPA is happy
        fclose(infile);
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

void printCluster(cluster c, int num_dimensions) {
    printf("Probability: %f\n", c.pi);
    printf("N: %f\n",c.N);
    printf("Spectral Mean: ");
    for(int i=0; i<num_dimensions; i++){
        printf("%.3f ",c.means[i]);
    }
    printf("\n");

    printf("\nR Matrix:\n");
    for(int i=0; i<num_dimensions; i++) {
        for(int j=0; j<num_dimensions; j++) {
            printf("%.3f ", c.R[i*num_dimensions+j]);
        }
        printf("\n");
    }   
    
    printf("\nR-inverse Matrix:\n");
    for(int i=0; i<num_dimensions; i++) {
        for(int j=0; j<num_dimensions; j++) {
            printf("%.3f ", c.Rinv[i*num_dimensions+j]);
        }
        printf("\n");
    } 
    
    printf("\nConstant: %f\n",c.constant);
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
    
    // Read FCS data    
    float* fcs_data = readData(argv[2],&num_dimensions,&num_events);
    
    if(!fcs_data) {
        printf("Error parsing input file. This could be due to an empty file ");
        printf("or an inconsistent number of dimensions. Aborting.\n");
        return 1;
    }
    
    printf("Number of events: %d\n",num_events);
    printf("Number of dimensions: %d\n\n",num_dimensions);
    
    CUT_DEVICE_INIT(argc, argv);

    
    // print the input
    for( unsigned int i = 0; i < num_events*num_dimensions; i += num_dimensions ) 
    {
        for(unsigned int j = 0; j < num_dimensions; j++) {
            //printf("%f ",fcs_data[i+j]);
        }
        //printf("\n");
    }
    
    unsigned int num_threads = num_dimensions*num_dimensions;
    if(num_threads > NUM_THREADS) {
        num_threads = NUM_THREADS;
    }

    // Setup the cluster data structures on host
    cluster* clusters = (cluster*)malloc(sizeof(cluster)*num_clusters);
    for(int i=0; i<num_clusters;i++) {
        clusters[i].N = 0.0;
        clusters[i].pi = 0.0;
        clusters[i].means = (float*) malloc(sizeof(float)*num_dimensions);
        clusters[i].R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
        clusters[i].Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
        clusters[i].constant = 0.0;
        clusters[i].p = (float*) malloc(sizeof(float)*num_events);
        clusters[i].w = (float*) malloc(sizeof(float)*num_events);
    }
    
    unsigned int timer = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    
    // Setup the cluster data structures on device
    // First allocate structures on the host, CUDA malloc the arrays
    // Then CUDA malloc structures on the device and copy them over
    cluster* temp_clusters = (cluster*) malloc(sizeof(cluster)*num_clusters);
    for(int i=0; i<num_clusters;i++) {
        temp_clusters[i].N = 0.0;
        temp_clusters[i].pi = 0.0;
        temp_clusters[i].constant = 0.0;
        CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters[i].means),sizeof(float)*num_dimensions));
        if(!temp_clusters[i].means) printf("ERROR: Could not allocate memory.\n");
        CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters[i].R),sizeof(float)*num_dimensions*num_dimensions));
        if(!temp_clusters[i].R) printf("ERROR: Could not allocate memory.\n");
        CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters[i].Rinv),sizeof(float)*num_dimensions*num_dimensions));
        if(!temp_clusters[i].Rinv) printf("ERROR: Could not allocate memory.\n");
        CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters[i].p),sizeof(float)*num_events));
        if(!temp_clusters[i].p) printf("ERROR: Could not allocate memory.\n");
        CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters[i].w),sizeof(float)*num_events));
        if(!temp_clusters[i].w) printf("ERROR: Could not allocate memory.\n");
    }
    cluster* d_clusters;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_clusters, sizeof(cluster)*num_clusters));
    
    unsigned int mem_size = num_dimensions*num_events*sizeof(float);
    
    // allocate device memory for FCS data
    float* d_fcs_data;
    CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data, mem_size));
    // copy FCS to device
    CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data, fcs_data, mem_size,cudaMemcpyHostToDevice) );

    // Copy Cluster data to device
    CUDA_SAFE_CALL(cudaMemcpy(d_clusters,temp_clusters,sizeof(cluster)*num_clusters,cudaMemcpyHostToDevice));
    
    printf("Invoking seed_clusters kernel\n");
    // execute the kernel
    seed_clusters<<< 1, num_threads >>>( d_fcs_data, d_clusters, num_dimensions, num_clusters, num_events);
    
    double determinant;
        
    // Compute new constants and invert matrix
    // copy clusters from the device
    CUDA_SAFE_CALL(cudaMemcpy(temp_clusters, d_clusters, sizeof(cluster)*num_clusters,cudaMemcpyDeviceToHost));
    for(int i=0; i<num_clusters; i++) {
        // copy the R matrix from the device
        CUDA_SAFE_CALL(cudaMemcpy(temp_clusters[i].R, d_clusters[i].R, sizeof(float)*num_dimensions*num_dimensions,cudaMemcpyDeviceToHost));

        // copy R into Rinv
        memcpy(temp_clusters[i].Rinv,temp_clusters[i].R,sizeof(float)*num_dimensions*num_dimensions);
        
        // invert the matrix
        //invert_matrix(temp_clusters[i].Rinv,num_dimensions,&determinant);
        invert(temp_clusters[i].Rinv,num_dimensions,&determinant);
        
        // compute the new constant
        temp_clusters[i].constant = (-num_dimensions)*0.5*log(2*3.14159)-0.5*log(fabs(determinant));
        printf("Determinant: %E, new constant: %f\n",fabs(determinant),temp_clusters[i].constant);
        
        // copy the R matrix back to the device
        CUDA_SAFE_CALL(cudaMemcpy(d_clusters[i].Rinv, temp_clusters[i].Rinv, sizeof(float)*num_dimensions*num_dimensions,cudaMemcpyHostToDevice));
        // copy the new constant to the device
        CUDA_SAFE_CALL(cudaMemcpy(&(d_clusters[i].constant), &(temp_clusters[i].constant), sizeof(float),cudaMemcpyHostToDevice));            
    }
    
    // Calculate an epsilon value
    int ndata_points = num_events*num_dimensions;
    float epsilon = (1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)*log((float)ndata_points)*0.01;
    float likelihood, old_likelihood;
    
    epsilon = epsilon*1;
    printf("Gaussian.cu: epsilon = %f\n",epsilon);
    
    float* d_likelihood;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_likelihood, sizeof(float)));

    // do initial regrouping
    printf("Invoking regroup kernel\n");
    regroup<<<1, num_threads>>>(d_fcs_data,d_clusters,num_dimensions,num_clusters,num_events,d_likelihood);
    CUDA_SAFE_CALL(cudaMemcpy(&likelihood,d_likelihood,sizeof(float),cudaMemcpyDeviceToHost));
    printf("Gaussian.cu: likelihood = %f\n",likelihood);

    float change = epsilon*2;
    
    while(change > epsilon) {
        old_likelihood = likelihood;
        printf("Invoking reestimate_parameters kernel\n");
        reestimate_parameters<<<1, num_threads>>>(d_fcs_data,d_clusters,num_dimensions,num_clusters,num_events);
        
        // Compute new constants and invert matrix
        // copy clusters from the device
        CUDA_SAFE_CALL(cudaMemcpy(temp_clusters, d_clusters, sizeof(cluster)*num_clusters,cudaMemcpyDeviceToHost));
        for(int i=0; i<num_clusters; i++) {
            // copy the R matrix from the device
            CUDA_SAFE_CALL(cudaMemcpy(temp_clusters[i].R, d_clusters[i].R, sizeof(float)*num_dimensions*num_dimensions,cudaMemcpyDeviceToHost));

            // copy R into Rinv
            memcpy(temp_clusters[i].Rinv,temp_clusters[i].R,sizeof(float)*num_dimensions*num_dimensions);
            
            // invert the matrix
            //invert_matrix(temp_clusters[i].Rinv,num_dimensions,&determinant);
            invert(temp_clusters[i].Rinv,num_dimensions,&determinant);
            
            // compute the new constant
            temp_clusters[i].constant = (-num_dimensions)*0.5*log(2*3.14159)-0.5*log(fabs(determinant));
            printf("Determinant: %E, new constant: %f\n",fabs(determinant),temp_clusters[i].constant);
            
            // copy the R matrix back to the device
            CUDA_SAFE_CALL(cudaMemcpy(d_clusters[i].Rinv, temp_clusters[i].Rinv, sizeof(float)*num_dimensions*num_dimensions,cudaMemcpyHostToDevice));
            // copy the new constant to the device
            CUDA_SAFE_CALL(cudaMemcpy(&(d_clusters[i].constant), &(temp_clusters[i].constant), sizeof(float),cudaMemcpyHostToDevice));            
        }
        
        printf("Invoking regroup kernel\n");
        regroup<<<1, num_threads>>>(d_fcs_data,d_clusters,num_dimensions,num_clusters,num_events,d_likelihood);
        CUDA_SAFE_CALL(cudaMemcpy(&likelihood,d_likelihood,sizeof(float),cudaMemcpyDeviceToHost));
        printf("likelihood = %f\n",likelihood);
        change = likelihood - old_likelihood;
        printf("Change in likelihood: %f\n",change);
    }

    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");
    
    // copy clusters from the device
    CUDA_SAFE_CALL(cudaMemcpy(temp_clusters, d_clusters, sizeof(cluster)*num_clusters,cudaMemcpyDeviceToHost));
    // copy all of the arrays from the structs
    for(int i=0; i<num_clusters; i++) {
        CUDA_SAFE_CALL(cudaMemcpy(clusters[i].means, temp_clusters[i].means, sizeof(float)*num_dimensions,cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(clusters[i].R, temp_clusters[i].R, sizeof(float)*num_dimensions*num_dimensions,cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(clusters[i].Rinv, temp_clusters[i].Rinv, sizeof(float)*num_dimensions*num_dimensions,cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(clusters[i].p, temp_clusters[i].p, sizeof(float)*num_events,cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(clusters[i].w, temp_clusters[i].w, sizeof(float)*num_events,cudaMemcpyDeviceToHost));
        clusters[i].N = temp_clusters[i].N;
        clusters[i].pi = temp_clusters[i].pi;
        clusters[i].constant = temp_clusters[i].constant;
    }
    
    CUT_SAFE_CALL(cutStopTimer(timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    CUT_SAFE_CALL(cutDeleteTimer(timer));
    
    for(int c=0; c<num_clusters; c++) {
        printCluster(clusters[c],num_dimensions);
    }
    // cleanup memory
    free(fcs_data);
    for(int i=0; i<num_clusters; i++) {
        free(clusters[i].means);
        free(clusters[i].R);
        free(clusters[i].Rinv);
        free(clusters[i].p);
        free(clusters[i].w);
    }    
    free(clusters);
    CUDA_SAFE_CALL(cudaFree(d_fcs_data));
    for(int i=0; i<num_clusters; i++) {
        CUDA_SAFE_CALL(cudaFree(temp_clusters[i].means));
        CUDA_SAFE_CALL(cudaFree(temp_clusters[i].R));
        CUDA_SAFE_CALL(cudaFree(temp_clusters[i].Rinv));
        CUDA_SAFE_CALL(cudaFree(temp_clusters[i].p));
        CUDA_SAFE_CALL(cudaFree(temp_clusters[i].w));
    }
    free(temp_clusters);
    CUDA_SAFE_CALL(cudaFree(d_clusters));

    return 0;
}
