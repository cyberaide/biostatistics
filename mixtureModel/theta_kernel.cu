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

__device__ void invert_RMatrix(float* matrix, int num_dimensions, float* determinant) {
    // Do an LU decomposition on the matrix...
    
    // Backsubstitute...
    
    
    *determinant = 1.0;
}

__device__ void normalize_pi(cluster* clusters, int num_clusters) {
    __shared__ float sum;
    // TODO: could maybe use a parallel reduction..but the # of elements is really small
    if(threadIdx.x == 0) {
        sum = 0.0;
        for(int i=0; i<num_clusters; i++) {
            sum += clusters[i].pi;
            //printf("original pi value: %f\n",clusters[i].pi);
        }
    }
    
    // Other threads need to wait for thread 0 to do the sum
    __syncthreads();
    
    if(threadIdx.x < num_clusters) {
        //printf("Sum of probabilities: %f\n",sum);
        if(sum > 0.0) {
            clusters[threadIdx.x].pi /= sum;
        } else {
            clusters[threadIdx.x].pi = 0.0;
        }
        //printf("Normalized pi value: %f\n",clusters[threadIdx.x].pi);
    }
}

__device__ void compute_constants(cluster* clusters, int num_clusters, int num_dimensions) {
    float determinant;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int num_elements = num_dimensions*num_dimensions;
    
    __shared__ float matrix[21*21]; // TODO: Make num_dimensions a #define constant
    
    // Invert the matrix for every cluster
    for(int c=0; c < num_clusters; c++) {
        // Copy the R matrix into shared memory for doing the matrix inversion
        for(int i=0; i<num_elements; i+= num_threads ) {
            if(i+tid < num_elements) {
                matrix[i+tid] = clusters[c].R[i+tid];
            }
        }
        
        __syncthreads(); // Not sure if this is neccesary..

        invert_RMatrix(matrix,num_dimensions,&determinant);

        __syncthreads(); // Not sure if this is neccesary..
        
        // Copy the matrx from shared memory back into the cluster memory
        for(int i=0; i<num_elements; i+= num_threads) {
            if(i+tid < num_elements) {
                clusters[c].Rinv[i+tid] = matrix[i+tid];
            }
        }
        
        __syncthreads();
    }
    
    // Compute the constant
    if(threadIdx.x < num_clusters) {
        clusters[tid].constant = -num_dimensions*0.5*log(2*PI) - 0.5*log(determinant);
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
seed_clusters( float* g_idata, cluster* clusters, int num_dimensions, int num_clusters, int num_events) 
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
    __shared__ int num_elements;
    __shared__ int row[192];
    __shared__ int col[192];
    
    // Number of elements in the covariance matrix
    num_elements = num_dimensions*num_dimensions; 

    __syncthreads();

    // Compute the initial covariance matrix of the data
    for(int i=0; i < num_elements; i+= num_threads) {
        if( (i+tid) < num_elements) { // make sure we don't proces too many elements
            // zero the value, find what row and col this thread is computing
            covs[i+tid] = 0.0;
            row[tid] = (i+tid) / num_dimensions;
            col[tid] = (i+tid) % num_dimensions;

            for(int j=0; j < num_events; j++) {
                covs[i+tid] += (g_idata[j*num_dimensions+row[tid]])*(g_idata[j*num_dimensions+col[tid]]); 
            }
            covs[i+tid] = covs[i+tid] / (float) num_events;
            covs[i+tid] = covs[i+tid] - means[row[tid]]*means[col[tid]];
        }
    }
    
    __syncthreads();    
    
    // Calculate a seed value for the means
    float seed;
    if(num_clusters > 1) {
        seed = (num_events-1.0)/(num_clusters-1.0);
    } else {
        seed = 0.0;
    }
    
    __syncthreads();
    
    // Seed the pi, means, and covariances for every cluster
    for(int c=0; c < num_clusters; c++) {
        clusters[c].pi = 1.0/num_clusters;
        if(tid < num_dimensions) {
            //clusters[c].means[tid] = fcs_data[((int)(c*seed))*num_dimensions+tid];
            clusters[c].means[tid] = means[tid];
        }
          
        for(int i=0; i < num_elements; i+= num_threads) {
            if( (i+tid) < num_elements) { // make sure we don't process too many elements
                // Add the average variance divided by a constant, this keeps the cov matrix from becoming singular
                clusters[c].R[i+tid] = covs[i+tid] + avgvar/COVARIANCE_DYNAMIC_RANGE;
            }
        }
    }
    __syncthreads();
    
    // Compute matrix inverses and constants for each cluster
    compute_constants(clusters,num_clusters,num_dimensions);
    
    __syncthreads();
    
    normalize_pi(clusters,num_clusters);
    
    __syncthreads();
}

__device__ float
regroup(float* fcs_data, cluster* clusters, int num_dimensions, int num_clusters, int num_events) {
    __shared__ float temp[192];
    __shared__ float max_likelihoods[192];
    __shared__ float denominator_sums[192];
    __shared__ float total_likelihoods[192];
    
    const int num_threads = blockDim.x;
    
    total_likelihoods[threadIdx.x] = 0.0;
    
    // Compute likelihood for every event, for every cluster
    for(int pixel=threadIdx.x; pixel<num_events; pixel += num_threads) {
        max_likelihoods[threadIdx.x] = -1000000000000.0;
        
        // compute likelihood of pixel in cluster 'c'
        for(int c=0; c<num_clusters; c++) {
            temp[threadIdx.x] = 0.0;
            // this does the loglike() function
            for(int i=0; i<num_dimensions; i++) {
                for(int j=0; j<num_dimensions; j++) {
                    //printf("fcs_data[%d]: %f, clusters[%d].means[%d]: %f\n",pixel*num_dimensions+j,fcs_data[pixel*num_dimensions+j],c,j,clusters[c].means[j]);
                    //printf("diff1: %f, diff2: %f, Rinv: %f\n",(fcs_data[pixel*num_dimensions+i]-clusters[c].means[i]),(fcs_data[pixel*num_dimensions+j]-clusters[c].means[j]),clusters[c].Rinv[i*num_dimensions+j]);
                    temp[threadIdx.x] += (fcs_data[pixel*num_dimensions+i]-clusters[c].means[i])*(fcs_data[pixel*num_dimensions+j]-clusters[c].means[j])*clusters[c].Rinv[i*num_dimensions+j];
                }
            }            
            clusters[c].p[pixel] = -0.5*temp[threadIdx.x]+clusters[c].constant;
            // Keep track of the maximum likelihood
            max_likelihoods[threadIdx.x] = fmaxf(max_likelihoods[threadIdx.x],clusters[c].p[pixel]);
        }
        
        //printf("maximum_likelihood for pixel %d is %f\n",pixel,max_likelihoods[threadIdx.x]);
        denominator_sums[threadIdx.x] = 0.0;
        for(int c=0; c<num_clusters; c++) {
            //printf("Clusters[%d].pi: %f\n",c,clusters[c].pi);
            //printf("Clusters[%d].p[%d] before exp(): %f\n",c,pixel,clusters[c].p[pixel]);
            // ????: for some reason if I don't use a temporary variable here I get NaN results in clusters[c].pixel[pixel]
            // possible compiler optimization bug? or is just something goofy with the printing and the actual value would be fine on a real card?
            //temp[threadIdx.x] = exp(clusters[c].p[pixel]-max_likelihoods[threadIdx.x])*clusters[c].pi;
            //clusters[c].p[pixel] = temp[threadIdx.x];
            clusters[c].p[pixel] = exp(clusters[c].p[pixel]-max_likelihoods[threadIdx.x])*clusters[c].pi;
            //printf("Thread %d: Clusters[%d].p[%d]: %f\n",threadIdx.x,c,pixel,clusters[c].p[pixel]);
            denominator_sums[threadIdx.x] += clusters[c].p[pixel];
        }
        
        //printf("Denominator_sum: %f\n",denominator_sums[threadIdx.x]);
        
        total_likelihoods[threadIdx.x] += log(denominator_sums[threadIdx.x]) + max_likelihoods[threadIdx.x];
        
        // Normalizes probabilities
        for(int c=0; c<num_clusters; c++) {
            //temp[threadIdx.x] = clusters[c].p[pixel];
            //clusters[c].p[pixel] = temp[threadIdx.x] / denominator_sums[threadIdx.x];
            clusters[c].p[pixel] /= denominator_sums[threadIdx.x];
            //printf("Probability that pixel #%d is in cluster #%d: %f\n",pixel,c,clusters[c].p[pixel]);
        }
    }
    
    float likelihood = 0.0;
    
    __syncthreads();
    
    if(threadIdx.x == 0) {
        for(int i=0; i<num_threads; i++) {
            likelihood += total_likelihoods[i];
        }
    }
    
    __syncthreads();
    return likelihood;
}

__device__ void
reestimate_parameters(float* fcs_data, cluster* clusters, int num_dimensions, int num_clusters, int num_events) {
    // Figure out # of elements each thread should add up
    int num_elements_per_thread = num_events / blockDim.x;
    int start_index = threadIdx.x * num_elements_per_thread;
    int end_index;
    // handle the end block so that we add left-over elements too
    if(threadIdx.x == blockDim.x -1) {
        end_index = num_events;
    } else {
        end_index = start_index + num_elements_per_thread;
    }

    __shared__ int temp_sums[192];
    
    // Compute new N 
    for(int i=0; i<num_clusters; i++) {
        temp_sums[threadIdx.x] = 0.0;
        for(int s=start_index; s<end_index; s++) {
            temp_sums[threadIdx.x] += clusters[i].p[s];
        }
        __syncthreads();
        // Let the first thread add up all the intermediate sums
        if(threadIdx.x == 0) {
            clusters[i].N = 0;
            for(int j=0; j<blockDim.x; j++) {
                clusters[i].N += temp_sums[j];
            }
        }
    }
}

/*
 * EM Algorthm kernel. Iterates until the change in likelihood of the data points is less than some epsilon
 *
 * Each iteration calculates likelihoods for all data points to every cluster, 
 *   regroups the data pointers, and re-estimates model parameters
 */
__global__ void 
refine_clusters(float* fcs_data, cluster* clusters, int num_dimensions, int num_clusters, int num_events) {
    int nparams_clust = 1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions;
    int ndata_points = num_events*num_dimensions;
    float epsilon = nparams_clust*log((float)ndata_points)*0.01;
    
    float old_likelihood;
    
    // do initial regrouping
    float likelihood = regroup(fcs_data,clusters,num_dimensions,num_clusters,num_events);
    
    __syncthreads();
    
    float change = epsilon*2;
    while(change > epsilon) {
        old_likelihood = likelihood;
        reestimate_parameters(fcs_data,clusters,num_dimensions,num_clusters,num_events);
        likelihood = regroup(fcs_data,clusters,num_dimensions,num_clusters,num_events);
        change = likelihood - old_likelihood;
    }
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
