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

float cluster_distance(cluster* cluster1, cluster* cluster2, cluster* temp_cluster, int num_dimensions);
void copy_cluster(cluster* dest, cluster* src, int num_dimensions);
void add_clusters(cluster* cluster1, cluster* cluster2, cluster* temp_cluster, int num_dimensions);

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
int validateArguments(int argc, char** argv, int* num_clusters, int* target_num_clusters) {
    if(argc <= 5 && argc >= 4) {
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
        // parse target_num_clusters
        if(argc == 5) {
            if(!sscanf(argv[4],"%d",target_num_clusters)) {
                printf("Invalid number of desired clusters.\n\n");
                printUsage(argv);
                return 4;
            }
            if(target_num_clusters > num_clusters) {
                printf("target_num_clusters must be less than equal to num_clusters\n\n");
                printUsage(argv);
                return 4;
            }
        } else {
            *target_num_clusters = 0;
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
   printf("Usage: %s num_clusters infile outfile [target_num_clusters]\n",argv[0]);
   printf("\t num_clusters: The number of starting clusters\n");
   printf("\t infile: ASCII space-delimited FCS data file\n");
   printf("\t outfile: Clustering results output file\n");
   printf("\t target_num_clusters: A desired number of clusters. Must be less than or equal to num_clusters\n");
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
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
int
runTest( int argc, char** argv) 
{
    int original_num_clusters, desired_num_clusters, stop_number;
    
    int error = validateArguments(argc,argv,&original_num_clusters,&desired_num_clusters);
    
    // Don't continue if we had a problem with the program arguments
    if(error) {
        return 1;
    }
    
    if(desired_num_clusters == 0) {
        stop_number = 1;
    } else {
        stop_number = desired_num_clusters;
    }
    
    printf("Starting with %d cluster(s), will stop at %d cluster(s).\n",original_num_clusters,stop_number);
   
    int GPUCount;
    int device = 0;

    CUDA_SAFE_CALL(cudaGetDeviceCount(&GPUCount));

    if (GPUCount > 1) {
        device = 0;
        CUDA_SAFE_CALL(cudaSetDevice(device));
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("\nUsing device - %s\n\n", prop.name);

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
    
    //CUT_DEVICE_INIT(argc, argv);

    
    // print the input
    for( unsigned int i = 0; i < num_events*num_dimensions; i += num_dimensions ) 
    {
        for(unsigned int j = 0; j < num_dimensions; j++) {
            //printf("%f ",fcs_data[i+j]);
        }
        //printf("\n");
    }
    
    unsigned int num_threads = max(num_dimensions*num_dimensions,original_num_clusters);
    if(num_threads > NUM_THREADS) {
        num_threads = NUM_THREADS;
    }

    // Setup the cluster data structures on host
    cluster* clusters = (cluster*)malloc(sizeof(cluster)*original_num_clusters);
    for(int i=0; i<original_num_clusters;i++) {
        clusters[i].N = 0.0;
        clusters[i].pi = 0.0;
        clusters[i].means = (float*) malloc(sizeof(float)*num_dimensions);
        clusters[i].R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
        clusters[i].Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
        clusters[i].constant = 0.0;
        clusters[i].p = (float*) malloc(sizeof(float)*num_events);
    }
    // Used as a temporary space for combining clusters in reduce_order
    cluster* scratch_cluster = (cluster*)malloc(sizeof(cluster));
    scratch_cluster->N = 0.0;
    scratch_cluster->pi = 0.0;
    scratch_cluster->N = 0.0;
    scratch_cluster->pi = 0.0;
    scratch_cluster->means = (float*) malloc(sizeof(float)*num_dimensions);
    scratch_cluster->R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
    scratch_cluster->Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
    scratch_cluster->constant = 0.0;
    scratch_cluster->p = (float*) malloc(sizeof(float)*num_events);
    // Declare another set of clusters for saving the results
    // Here we're only concerned with the statistics of the cluster, so we don't need to malloc all the arrays
    cluster* saved_clusters = (cluster*)malloc(sizeof(cluster)*original_num_clusters);
    for(int i=0; i<original_num_clusters;i++) {
        saved_clusters[i].N = 0.0;
        saved_clusters[i].pi = 0.0;
        saved_clusters[i].means = (float*) malloc(sizeof(float)*num_dimensions);
        saved_clusters[i].R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
        saved_clusters[i].Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
    }
    
    unsigned int timer = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    
    // Setup the cluster data structures on device
    // First allocate structures on the host, CUDA malloc the arrays
    // Then CUDA malloc structures on the device and copy them over
    cluster* temp_clusters = (cluster*) malloc(sizeof(cluster)*original_num_clusters);
    for(int i=0; i<original_num_clusters;i++) {
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
    }
    cluster* d_clusters;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_clusters, sizeof(cluster)*original_num_clusters));
    
    unsigned int mem_size = num_dimensions*num_events*sizeof(float);
    
    double min_rissanen, rissanen;
    
    // allocate device memory for FCS data
    float* d_fcs_data;
    CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data, mem_size));
    // copy FCS to device
    CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data, fcs_data, mem_size,cudaMemcpyHostToDevice) );

    // Copy Cluster data to device
    CUDA_SAFE_CALL(cudaMemcpy(d_clusters,temp_clusters,sizeof(cluster)*original_num_clusters,cudaMemcpyHostToDevice));
    
    if(VERBOSE) {
        printf("Invoking seed_clusters kernel...");
    }
    // execute the kernel
    seed_clusters<<< 1, num_threads >>>( d_fcs_data, d_clusters, num_dimensions, original_num_clusters, num_events);
    cudaThreadSynchronize();
    if(VERBOSE) {
        printf("done.\n"); 
    }
    if(VERBOSE) {
        printf("Invoking constants kernel...",num_threads);
        fflush(stdout);
    }
    constants_kernel<<<NUM_BLOCKS, num_threads>>>(d_clusters,original_num_clusters,num_dimensions);
    cudaThreadSynchronize();
    if(VERBOSE) {
        printf("done.\n");
    }
    
    double determinant = 1.0;
        
    // copy clusters from the device
    CUDA_SAFE_CALL(cudaMemcpy(temp_clusters, d_clusters, sizeof(cluster)*original_num_clusters,cudaMemcpyDeviceToHost));
    
    // Calculate an epsilon value
    int ndata_points = num_events*num_dimensions;
    float epsilon = (1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)*log((float)ndata_points)*0.01;
    float likelihood, old_likelihood;
    
    epsilon = epsilon*1;

    if(VERBOSE) {
        printf("Gaussian.cu: epsilon = %f\n",epsilon);
    }

    // Used to hold the result from regroup kernel
    float* likelihoods = (float*) malloc(sizeof(float)*NUM_BLOCKS);
    float* d_likelihoods;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_likelihoods, sizeof(float)*NUM_BLOCKS));
    
    // Variables for GMM reduce order
    float distance, min_distance = 0.0;
    int min_c1, min_c2;
    int ideal_num_clusters;
    float* d_c;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_c, sizeof(float)));
     
    for(int num_clusters=original_num_clusters; num_clusters >= stop_number; num_clusters--) {
        /*************** EM ALGORITHM *****************************/
        // do initial regrouping
        if(VERBOSE) {
            printf("Invoking regroup kernel...");
        }
        regroup<<<NUM_BLOCKS, num_threads>>>(d_fcs_data,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
        cudaThreadSynchronize();
        printf("done.\n");
        // check if kernel execution generated and error
        CUT_CHECK_ERROR("Kernel execution failed");
        CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*NUM_BLOCKS,cudaMemcpyDeviceToHost));
        likelihood = 0.0;
        for(int i=0;i<NUM_BLOCKS;i++) {
            likelihood += likelihoods[i]; 
        }
        //printf("Gaussian.cu: likelihood = %f\n",likelihood);

        float change = epsilon*2;
        
        printf("Performing EM algorthm on %d clusters.\n",num_clusters);
        
        while(change > epsilon) {
            old_likelihood = likelihood;
            if(VERBOSE) {
                printf("Invoking reestimate_parameters kernel...",num_threads);
                fflush(stdout);
            }
            reestimate_parameters<<<NUM_BLOCKS, num_threads>>>(d_fcs_data,d_clusters,num_dimensions,num_clusters,num_events);
            cudaThreadSynchronize();
            if(VERBOSE) {
                printf("done.\n");
            }
            
            if(VERBOSE) {
                printf("Invoking constants kernel...",num_threads);
                fflush(stdout);
            }
            constants_kernel<<<NUM_BLOCKS, num_threads>>>(d_clusters,num_clusters,num_dimensions);
            cudaThreadSynchronize();
            if(VERBOSE) {
                printf("done.\n");
            }

            // check if kernel execution generated and error
            CUT_CHECK_ERROR("Kernel execution failed");
        
            if(VERBOSE) {
                printf("Invoking regroup kernel...");
                fflush(stdout);
            }
            regroup<<<NUM_BLOCKS, num_threads>>>(d_fcs_data,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
            cudaThreadSynchronize();
            if(VERBOSE) {
                printf("done.\n");
            }
        
            // check if kernel execution generated and error
            CUT_CHECK_ERROR("Kernel execution failed");
        
            CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*NUM_BLOCKS,cudaMemcpyDeviceToHost));
            likelihood = 0.0;
            for(int i=0;i<NUM_BLOCKS;i++) {
                //printf("Block #%d likelihood: ",likelihoods[i]);
                likelihood += likelihoods[i]; 
            }
            
            change = likelihood - old_likelihood;
            if(VERBOSE) {
                printf("likelihood = %f\n",likelihood);
                printf("Change in likelihood: %f\n",change);
            }
        }
        
        // copy clusters from the device
        CUDA_SAFE_CALL(cudaMemcpy(temp_clusters, d_clusters, sizeof(cluster)*num_clusters,cudaMemcpyDeviceToHost));
        // copy all of the arrays from the structs
        for(int i=0; i<num_clusters; i++) {
            CUDA_SAFE_CALL(cudaMemcpy(clusters[i].means, temp_clusters[i].means, sizeof(float)*num_dimensions,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(clusters[i].R, temp_clusters[i].R, sizeof(float)*num_dimensions*num_dimensions,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(clusters[i].Rinv, temp_clusters[i].Rinv, sizeof(float)*num_dimensions*num_dimensions,cudaMemcpyDeviceToHost));
            clusters[i].N = temp_clusters[i].N;
            clusters[i].pi = temp_clusters[i].pi;
            clusters[i].constant = temp_clusters[i].constant;
        }
        // Calculate Rissanen Score
        rissanen = -likelihood + 0.5*(num_clusters*(1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)-1)*log((double)num_events*num_dimensions);
        printf("\nRissanen Score: %f\n",rissanen);
        
        // Save the cluster data the first time through, so we have a base rissanen score and result
        // Save te cluster data if the solution is better and the user didn't specify a desired number
        // If the num_clusters equals the desired number, stop
        if(num_clusters == original_num_clusters || (rissanen < min_rissanen && desired_num_clusters == 0) || (num_clusters == desired_num_clusters)) {
            min_rissanen = rissanen;
            ideal_num_clusters = num_clusters;
            // Save the cluster configuration somewhere
            for(int i=0; i<num_clusters;i++) {
                saved_clusters[i].pi = clusters[i].pi;
                saved_clusters[i].N = clusters[i].N;
                memcpy(saved_clusters[i].means,clusters[i].means,sizeof(float)*num_dimensions);
                memcpy(saved_clusters[i].R,clusters[i].R,sizeof(float)*num_dimensions*num_dimensions);
                memcpy(saved_clusters[i].Rinv,clusters[i].Rinv,sizeof(float)*num_dimensions*num_dimensions);
            }
        }

        
        /**************** Reduce GMM Order ********************/
        
        // Don't want to reduce order on the last iteration
        if(num_clusters > stop_number) {
            
            
            // For all combinations of subclasses...
            for(int c1=0; c1<num_clusters;c1++) {
                for(int c2=c1+1; c2<num_clusters;c2++) {
                    // compute distance between the 2 clustesr
                    distance = cluster_distance(&(clusters[c1]),&(clusters[c2]),scratch_cluster,num_dimensions);
                    
                    // Keep track of minimum distance
                    if((c1 ==0 && c2 == c1+1) || distance < min_distance) {
                        min_distance = distance;
                        min_c1 = c1;
                        min_c2 = c2;
                    }
                }
            }

            printf("\nMinimum distance between (%d,%d). Combining clusters\n",min_c1,min_c2);
            // Add the two clusters with min distance together
            add_clusters(&(clusters[min_c1]),&(clusters[min_c2]),scratch_cluster,num_dimensions);
            // Copy new combined cluster into the main group of clusters, compact them
            copy_cluster(&(clusters[min_c1]),scratch_cluster,num_dimensions);
            for(int i=min_c2; i < num_clusters-1; i++) {
                copy_cluster(&(clusters[i]),&(clusters[i+1]),num_dimensions);
            }

            // Copy the clusters back to the device
            for(int i=0; i<num_clusters-1; i++) {
                CUDA_SAFE_CALL(cudaMemcpy(temp_clusters[i].means, clusters[i].means, sizeof(float)*num_dimensions,cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(temp_clusters[i].R, clusters[i].R, sizeof(float)*num_dimensions*num_dimensions,cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(temp_clusters[i].Rinv, clusters[i].Rinv, sizeof(float)*num_dimensions*num_dimensions,cudaMemcpyHostToDevice));
                temp_clusters[i].N = clusters[i].N;
                temp_clusters[i].pi = clusters[i].pi;
                temp_clusters[i].constant = clusters[i].constant;
            }
            CUDA_SAFE_CALL(cudaMemcpy(d_clusters,temp_clusters,sizeof(cluster)*(num_clusters-1),cudaMemcpyHostToDevice));
        }
        
    }
    printf("\n\nSolution coverged or began to diverge. Printing solution.\n");
    printf("\nFinal rissanen Score was: %f, with %d clusters.\n",min_rissanen,ideal_num_clusters);
 
    
    // copy clusters from the device
    CUDA_SAFE_CALL(cudaMemcpy(temp_clusters, d_clusters, sizeof(cluster)*original_num_clusters,cudaMemcpyDeviceToHost));
    
    /*
    // copy all of the arrays from the structs
    for(int i=0; i<num_clusters; i++) {
        CUDA_SAFE_CALL(cudaMemcpy(clusters[i].means, temp_clusters[i].means, sizeof(float)*num_dimensions,cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(clusters[i].R, temp_clusters[i].R, sizeof(float)*num_dimensions*num_dimensions,cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(clusters[i].Rinv, temp_clusters[i].Rinv, sizeof(float)*num_dimensions*num_dimensions,cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(clusters[i].p, temp_clusters[i].p, sizeof(float)*num_events,cudaMemcpyDeviceToHost));
        clusters[i].N = temp_clusters[i].N;
        clusters[i].pi = temp_clusters[i].pi;
        clusters[i].constant = temp_clusters[i].constant;
    }
    */
    
    CUT_SAFE_CALL(cutStopTimer(timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    CUT_SAFE_CALL(cutDeleteTimer(timer));
    
    for(int c=0; c<ideal_num_clusters; c++) {
        printf("-----------------------    Cluster #%d  ------------------------------\n",c);
        printCluster(saved_clusters[c],num_dimensions);
        printf("\n\n");
    }
    // cleanup memory
    free(fcs_data);
    for(int i=0; i<original_num_clusters; i++) {
        free(clusters[i].means);
        free(clusters[i].R);
        free(clusters[i].Rinv);
        free(clusters[i].p);
    }    
    free(clusters);
    for(int i=0; i<original_num_clusters; i++) {
        free(saved_clusters[i].means);
        free(saved_clusters[i].R);
        free(saved_clusters[i].Rinv);
    }    
    free(saved_clusters);
    
    free(scratch_cluster->means);
    free(scratch_cluster->R);
    free(scratch_cluster->Rinv);
    free(scratch_cluster);
   
    free(likelihoods);
    CUDA_SAFE_CALL(cudaFree(d_likelihoods));
 
    CUDA_SAFE_CALL(cudaFree(d_fcs_data));

    for(int i=0; i<original_num_clusters; i++) {
        CUDA_SAFE_CALL(cudaFree(temp_clusters[i].means));
        CUDA_SAFE_CALL(cudaFree(temp_clusters[i].R));
        CUDA_SAFE_CALL(cudaFree(temp_clusters[i].Rinv));
        CUDA_SAFE_CALL(cudaFree(temp_clusters[i].p));
    }
    free(temp_clusters);
    CUDA_SAFE_CALL(cudaFree(d_clusters));

    return 0;
}

float cluster_distance(cluster* cluster1, cluster* cluster2, cluster* temp_cluster, int num_dimensions) {
    double determinant;
    // Add the clusters together, this updates pi,means,R,N and stores in temp_cluster
    add_clusters(cluster1,cluster2,temp_cluster,num_dimensions);
    // Copy R to Rinv matrix
    memcpy(temp_cluster->Rinv,temp_cluster->R,sizeof(float)*num_dimensions*num_dimensions);
    // Invert the matrix
    invert(temp_cluster->Rinv,num_dimensions,&determinant);
    // Compute the constant
    temp_cluster->constant = (-num_dimensions)*0.5*log(2*PI)-0.5*log(fabs(determinant));
    
    return cluster1->N*cluster1->constant + cluster2->N*cluster2->constant - temp_cluster->N*temp_cluster->constant;
}

void add_clusters(cluster* cluster1, cluster* cluster2, cluster* temp_cluster, int num_dimensions) {
    double wt1,wt2;
    
    wt1 = ((double) cluster1->N) / ((double)(cluster1->N + cluster2->N));
    wt2 = 1 - wt1;
    
    // Compute new weighted means
    for(int i=0; i<num_dimensions;i++) {
        temp_cluster->means[i] = wt1*cluster1->means[i] + wt2*cluster2->means[i];
    }
    
    // Compute new weighted covariance
    for(int i=0; i<num_dimensions; i++) {
        for(int j=i; j<num_dimensions; j++) {
            // Compute R contribution from cluster1
            temp_cluster->R[i*num_dimensions+j] = ((temp_cluster->means[i]-cluster1->means[i])
                                                *(temp_cluster->means[j]-cluster1->means[j])
                                                +cluster1->R[i*num_dimensions+j])*wt1;
            // Add R contribution from cluster2
            temp_cluster->R[i*num_dimensions+j] += ((temp_cluster->means[i]-cluster2->means[i])
                                                    *(temp_cluster->means[j]-cluster2->means[j])
                                                    +cluster2->R[i*num_dimensions+j])*wt2;
            // Because its symmetric...
            temp_cluster->R[j*num_dimensions+i] = temp_cluster->R[i*num_dimensions+j];
        }
    }
    
    // Compute pi
    temp_cluster->pi = cluster1->pi + cluster2->pi;
    
    // compute N
    temp_cluster->N = cluster1->N + cluster2->N;
}

void copy_cluster(cluster* dest, cluster* src, int num_dimensions) {
    dest->N = src->N;
    dest->pi = src->pi;
    dest->constant = src->constant;
    memcpy(dest->means,src->means,sizeof(float)*num_dimensions);
    memcpy(dest->R,src->R,sizeof(float)*num_dimensions*num_dimensions);
    memcpy(dest->Rinv,src->Rinv,sizeof(float)*num_dimensions*num_dimensions);
}
