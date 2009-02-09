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

#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>
#include "gaussian.h"

#define sdata(index)      CUT_BANK_CHECKER(sdata, index)

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel( float* g_idata, cluster* clusters, int num_dimensions, int num_clusters, int num_events) 
{
    // shared memory
    __shared__ float means[21];

    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    if(tid < num_dimensions) {
        means[tid] = 0.0;
    }

    __syncthreads();

    // Compute means
    for(unsigned int i=tid; i < num_events*num_dimensions; i+= num_dimensions) {
        if(tid < num_dimensions) {
            means[tid] += g_idata[i];
        }  
    }
    
    __syncthreads();

    // write data to global memory
    if(tid < num_dimensions) {
        means[tid] /= (float) num_events;
        clusters[0].means[tid] = means[tid];
    }

    __syncthreads();
    
    // Initialize covariances
    __shared__ float covs[21*21];
    __shared__ int num_elements;
    __shared__ int row;
    __shared__ int col;
    num_elements = num_dimensions*num_dimensions; 
    row = 0;
    col = 0;

    __syncthreads();

    for(int i=0; i < num_elements; i+= num_threads) {
        if(i+tid < num_elements) { // make sure we don't proces too many elements

            // zero the value, find what row and col this thread is computing
            covs[i+tid] = 0.0;
            row = (i+tid) / num_dimensions;
            col = (i+tid) % num_dimensions;

            for(int j=0; j < num_events; j++) {
                //printf("data[%d][%d]: %f, data[%d][%d]: %f\n",j,row,g_idata[j*num_dimensions+row],j,col,g_idata[j*num_dimensions+col]);
                covs[i+tid] += (g_idata[j*num_dimensions+row])*(g_idata[j*num_dimensions+col]); 
            }
            //printf("covs[%d][%d]: %f\n",row,tid,covs[i+tid]);
            clusters[0].R[i+tid] = covs[i+tid] / (float) num_events;
            clusters[0].R[i+tid] -= means[row]*means[col];
        }
    }
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
