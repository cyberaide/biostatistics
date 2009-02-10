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
 * Device code.
 */

#define COVARIANCE_DYNAMIC_RANGE 1E5

#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>
#include "gaussian.h"

#define sdata(index)      CUT_BANK_CHECKER(sdata, index)

/*
 * Compute the spectral means of the FCS data
 */ 
__device__ void spectralMean(float* fcs_data, int num_dimensions, int num_events, float* means) {
    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;
    // Cluster Id, what cluster this thread block is working on
    //const unsigned int cid = blockIdx.x;

    if(tid < num_dimensions) {
        means[tid] = 0.0;
    }

    __syncthreads();

    // Sum up all the values for the dimension
    for(unsigned int i=tid; i < num_events*num_dimensions; i+= num_dimensions) {
        if(tid < num_dimensions) {
            means[tid] += fcs_data[i];
        }  
    }
    
    __syncthreads();

    // Divide by the # of elements to get the average
    if(tid < num_dimensions) {
        means[tid] /= (float) num_events;
    }
}

__device__ void averageVariance(float* fcs_data, float* means, int num_dimensions, int num_events, float* avgvar) {
    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads
    const unsigned int num_threads = blockDim.x;
    
    __shared__ float variances[21];
    __shared__ float total_variance;
    
    // Compute average variance for each dimension
    for(int i=0; i < num_dimensions; i += num_threads) {
        if(tid+i < num_dimensions) {
            variances[tid] = 0.0;
            // Sum up all the variance
            for(int j=0; j < num_events; j++) {
                // variance = (data - mean)^2
                //variances[tid+i] += (fcs_data[j*num_dimensions + tid + i]-means[tid+i])*(fcs_data[j*num_dimensions + tid + i]-means[tid+i]);
                variances[tid+i] += (fcs_data[j*num_dimensions + tid + i])*(fcs_data[j*num_dimensions + tid + i]);
            }
            variances[tid+i] /= (float) num_events;
            variances[tid+i] -= means[tid+i]*means[tid+i];
        }
    }
    
    __syncthreads();
    
    if(tid == 0) {
        total_variance = 0.0;
        for(int i=0; i<num_dimensions;i++) {
            //printf("%f ",variances[tid]);
            total_variance += variances[i];
        }
        //printf("\nTotal variance: %f\n",total_variance);
        *avgvar = total_variance / (float) num_dimensions;
        //printf("Average Variance: %f\n",*avgvar);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata          FCS data: [num_events]
//! @param clusters         Clusters: [num_clusters]
//! @param num_dimensions   number of dimensions in an FCS event
//! @param num_events       number of FCS events
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel( float* g_idata, cluster* clusters, int num_dimensions, int num_clusters, int num_events) 
{
    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;
    // Cluster Id, what cluster this thread block is working on
    //const unsigned int cid = blockIdx.x;

    // shared memory
    __shared__ float means[21]; // TODO: setup #define for the number of dimensions
    spectralMean(g_idata, num_dimensions, num_events, means);

    __syncthreads();
    
    float avgvar;
    
    averageVariance(g_idata, means, num_dimensions, num_events, &avgvar);
    
    //printf("Average Variance: %f\n",avgvar);
    
    // Initialize covariances
    __shared__ float covs[21*21]; // TODO: setup #define for the number of dimensions
    __shared__ int num_elements, row, col;
    // Number of elements in the covariance matrix
    num_elements = num_dimensions*num_dimensions; 

    __syncthreads();

    // Compute the initial covariance matrix of the data
    for(int i=0; i < num_elements; i+= num_threads) {
        if(i+tid < num_elements) { // make sure we don't proces too many elements
            // zero the value, find what row and col this thread is computing
            covs[i+tid] = 0.0;
            row = (i+tid) / num_dimensions;
            col = (i+tid) % num_dimensions;

            for(int j=0; j < num_events; j++) {
                covs[i+tid] += (g_idata[j*num_dimensions+row])*(g_idata[j*num_dimensions+col]); 
            }
            covs[i+tid] = covs[i+tid] / (float) num_events;
            covs[i+tid] = covs[i+tid] - means[row]*means[col];
        }
    }
    
    __syncthreads();
    
    // Copy the covariance matrix into every cluster
    for(int c=0; c < num_clusters; c++) {
        if(tid < num_dimensions) {
            clusters[c].means[tid] = means[tid];
        }
        for(int i=0; i < num_elements; i+= num_threads) {
            if(i+tid < num_elements) { // make sure we don't process too many elements
                row = (i+tid) / num_dimensions;
                col = (i+tid) % num_dimensions;
                // Add the average variance divided by a constant, this keeps the cov matrix from becoming singular
                clusters[c].R[i+tid] = covs[i+tid] + avgvar/COVARIANCE_DYNAMIC_RANGE;
            }
        }
    }
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
