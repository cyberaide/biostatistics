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
    int tid = threadIdx.x;
    // access number of threads in this block
    int num_threads = blockDim.x;

    if(tid < num_dimensions) {
        means[tid] = 0.0;
    }

    __syncthreads();

    int num_data_points = num_events*num_dimensions;

    // Sum up all the values for the dimension
    for(int i=tid; i < num_data_points; i+= num_dimensions) {
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
    int tid = threadIdx.x;
    // access number of threads
    int num_threads = blockDim.x;
    
    __shared__ float variances[NUM_DIMENSIONS];
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
            ////printf("%f ",variances[tid]);
            total_variance += variances[i];
        }
        ////printf("\nTotal variance: %f\n",total_variance);
        *avgvar = total_variance / (float) num_dimensions;
        ////printf("Average Variance: %f\n",*avgvar);
    }
}

__device__ void invert(float* data, int actualsize, float* determinant)  {
    int maxsize = actualsize;
    int n = actualsize;
    
    if(threadIdx.x == 0) {
    *determinant = 1.0;

#if EMU
    if(VERBOSE) {
        printf("\n\nR matrix before inversion:\n");
        for(int i=0; i<n; i++) {
            for(int j=0; j<n; j++) {
                printf("%.2f ",data[i*n+j]);
            }
            printf("\n");
        }
    }
#endif
    
  if (actualsize <= 0) return;  // sanity check
  if (actualsize == 1) return;  // must be of dimension >= 2
  for (int i=1; i < actualsize; i++) data[i] /= data[0]; // normalize row 0
  for (int i=1; i < actualsize; i++)  { 
    for (int j=i; j < actualsize; j++)  { // do a column of L
      float sum = 0.0;
      for (int k = 0; k < i; k++)  
          sum += data[j*maxsize+k] * data[k*maxsize+i];
      data[j*maxsize+i] -= sum;
      }
    if (i == actualsize-1) continue;
    for (int j=i+1; j < actualsize; j++)  {  // do a row of U
      float sum = 0.0;
      for (int k = 0; k < i; k++)
          sum += data[i*maxsize+k]*data[k*maxsize+j];
      data[i*maxsize+j] = 
         (data[i*maxsize+j]-sum) / data[i*maxsize+i];
      }
    }
    
    for(int i=0; i<actualsize; i++) {
        *determinant *= data[i*n+i];
    }
#if EMU
    if(VERBOSE) {
        printf("Determinant: %E\n",*determinant);
    }
#endif
    
  for ( int i = 0; i < actualsize; i++ )  // invert L
    for ( int j = i; j < actualsize; j++ )  {
      float x = 1.0;
      if ( i != j ) {
        x = 0.0;
        for ( int k = i; k < j; k++ ) 
            x -= data[j*maxsize+k]*data[k*maxsize+i];
        }
      data[j*maxsize+i] = x / data[j*maxsize+j];
      }
  for ( int i = 0; i < actualsize; i++ )   // invert U
    for ( int j = i; j < actualsize; j++ )  {
      if ( i == j ) continue;
      float sum = 0.0;
      for ( int k = i; k < j; k++ )
          sum += data[k*maxsize+j]*( (i==k) ? 1.0 : data[i*maxsize+k] );
      data[i*maxsize+j] = -sum;
      }
  for ( int i = 0; i < actualsize; i++ )   // final inversion
    for ( int j = 0; j < actualsize; j++ )  {
      float sum = 0.0;
      for ( int k = ((i>j)?i:j); k < actualsize; k++ )  
          sum += ((j==k)?1.0:data[j*maxsize+k])*data[k*maxsize+i];
      data[j*maxsize+i] = sum;
      }
      
#if EMU
      if(VERBOSE) {
          printf("\n\nR matrix after inversion:\n");
          for(int i=0; i<n; i++) {
              for(int j=0; j<n; j++) {
                  printf("%.2f ",data[i*n+j]);
              }
              printf("\n");
          }
      }
#endif
    }
 }


__device__ void normalize_pi(cluster* clusters, int num_clusters) {
    __shared__ float sum;
    
    // TODO: could maybe use a parallel reduction..but the # of elements is really small
    // What is better: having thread 0 compute a shared sum and sync, or just have each one compute the sum?
    if(threadIdx.x == 0) {
        sum = 0.0;
        for(int i=0; i<num_clusters; i++) {
            sum += clusters[i].pi;
        }
    }
    
    __syncthreads();
    
    if(threadIdx.x < num_clusters) {
        if(sum > 0.0) {
            clusters[threadIdx.x].pi /= sum;
        } else {
            clusters[threadIdx.x].pi = 0.0;
        }
    }
    
    __syncthreads();
}


__device__ void compute_constants(cluster* clusters, int num_clusters, int num_dimensions) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int num_elements = num_dimensions*num_dimensions;
    
    __shared__ float determinant_arg;
    
    float determinant;
    
    __shared__ float matrix[NUM_DIMENSIONS*NUM_DIMENSIONS];
    
    // Invert the matrix for every cluster
    for(int c=blockIdx.x; c < num_clusters; c+=NUM_BLOCKS) {
        // Copy the R matrix into shared memory for doing the matrix inversion
        for(int i=tid; i<num_elements; i+= num_threads ) {
            matrix[i] = clusters[c].R[i];
        }
        
        __syncthreads(); 

        invert(matrix,num_dimensions,&determinant_arg);

        __syncthreads(); 
        
        determinant = determinant_arg;
        
        // Copy the matrx from shared memory back into the cluster memory
        for(int i=tid; i<num_elements; i+= num_threads) {
            clusters[c].Rinv[i] = matrix[i];
        }
        
        __syncthreads();
        
        // Compute the constant
        if(tid == 0) {
            determinant = fabs(determinant);
            clusters[c].constant = -num_dimensions*0.5*logf(2*PI) - 0.5*logf(determinant);
        }
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
    int tid = threadIdx.x;
    // access number of threads in this block
    int num_threads = blockDim.x;

    // shared memory
    __shared__ float means[NUM_DIMENSIONS];
    
    // Compute the means
    spectralMean(g_idata, num_dimensions, num_events, means);

    __syncthreads();
    
    float avgvar;
    
    // Compute the average variance
    averageVariance(g_idata, means, num_dimensions, num_events, &avgvar);
        
    // Initialize covariances
    __shared__ float covs[NUM_DIMENSIONS*NUM_DIMENSIONS]; 
    
    int num_elements;
    int row, col;
        
    // Number of elements in the covariance matrix
    num_elements = num_dimensions*num_dimensions; 

    __syncthreads();

    // Compute the initial covariance matrix of the data
    for(int i=tid; i < num_elements; i+= num_threads) {
            // zero the value, find what row and col this thread is computing
            covs[i] = 0.0;
            row = (i) / num_dimensions;
            col = (i) % num_dimensions;

            for(int j=0; j < num_events; j++) {
                covs[i] += (g_idata[j*num_dimensions+row])*(g_idata[j*num_dimensions+col]); 
            }
            covs[i] = covs[i] / (float) num_events;
            covs[i] = covs[i] - means[row]*means[col];
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
            clusters[c].means[tid] = g_idata[((int)(c*seed))*num_dimensions+tid];
            //clusters[c].means[tid] = means[tid];
        }
          
        for(int i=tid; i < num_elements; i+= num_threads) {
            // Add the average variance divided by a constant, this keeps the cov matrix from becoming singular
            clusters[c].R[i] = covs[i] + avgvar/COVARIANCE_DYNAMIC_RANGE;
        }
        
        clusters[c].avgvar = avgvar / COVARIANCE_DYNAMIC_RANGE;
    }
}

__global__ void
regroup(float* fcs_data, cluster* clusters, int num_dimensions, int num_clusters, int num_events, float* likelihood) {
    float like;
    float max_likelihood;
    float denominator_sum;
    float temp;
    float thread_likelihood = 0.0;
    __shared__ float total_likelihoods[NUM_THREADS];
    
    const int num_threads = blockDim.x;
    int num_pixels_per_block = num_events / NUM_BLOCKS;  
    const int tid = threadIdx.x;
    
    int start_index;
    int end_index;
    start_index = blockIdx.x * num_pixels_per_block + tid;
    
    if(blockIdx.x == NUM_BLOCKS-1) {
        end_index = num_events;
    } else {
        end_index = (blockIdx.x+1) * num_pixels_per_block;
    }
    
    //printf("Block Index: %d, Thread Index: %d, start_index: %d, end_index: %d\n",blockIdx.x,tid,start_index,end_index);

    int data_index;
    
    total_likelihoods[tid] = 0.0;
  
#if EMU
    if(VERBOSE) { 
        for(int c=0;c<num_clusters;c++) {
            if(tid==0) {
                printf("cluster[%d].Rinv matrix:\n",c);
                for(int i=0;i<num_dimensions;i++) {
                    for(int j=0; j<num_dimensions;j++) {
                        printf("%.3f ",clusters->Rinv[i*num_dimensions+j]);
                    }
                    printf("\n");
                }
            }
        }
    }
#endif
    
    
    // Compute likelihood for every event, for every cluster
    for(int pixel=start_index; pixel<end_index; pixel += num_threads) {
       
        data_index = pixel*num_dimensions;
        // compute likelihood of pixel in cluster 'c'
        for(int c=0; c<num_clusters; c++) {
            like = 0.0;
            // this does the loglike() function
            for(int i=0; i<num_dimensions; i++) {
                for(int j=0; j<num_dimensions; j++) {
                    like += (fcs_data[data_index+i]-clusters[c].means[i])*(fcs_data[data_index+j]-clusters[c].means[j])*clusters[c].Rinv[i*num_dimensions+j];
                }
            }
            temp = -0.5*like+clusters[c].constant;
            clusters[c].p[pixel] = temp;
 
            // Keep track of the maximum likelihood
            if(c == 0) {
                max_likelihood = temp;
            } 
            if( temp > max_likelihood) {
                max_likelihood = temp;
            }
        }
        
        denominator_sum = 0.0;
        for(int c=0; c<num_clusters; c++) {
            temp = exp(clusters[c].p[pixel]-max_likelihood)*clusters[c].pi;
            denominator_sum += temp;
            clusters[c].p[pixel] = temp;
        }
        thread_likelihood += log(denominator_sum) + max_likelihood;
        
        // Normalizes probabilities
        for(int c=0; c<num_clusters; c++) {
            clusters[c].p[pixel] /= denominator_sum;
            //printf("Probability that pixel #%d is in cluster #%d: %f\n",pixel,c,clusters[c].p[pixel]);
        }
    }
    
    total_likelihoods[tid] = thread_likelihood;

    
    float retval = 0.0;
    
    __syncthreads();
    
    // Reduce all the total_likelihoods to a single total
    if(tid == 0) {
        for(int i=0; i<num_threads; i++) {
            retval += total_likelihoods[i];
        }
        likelihood[blockIdx.x] = retval;
    }
}

/*
 * This kernel re-computes the means, N (number of data points per cluster),
 * and R (covariance matrix). The computations for each cluster are independent
 * therefore each cluster can be computed by a different block
 */
__global__ void
reestimate_parameters(float* fcs_data, cluster* clusters, int num_dimensions, int num_clusters, int num_events) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Number of elements in the covariance matrix
    int num_elements = num_dimensions*num_dimensions;
    
    // Figure out # of elements each thread should add up
    int num_elements_per_thread = num_events / num_threads;
    int start_index = tid * num_elements_per_thread;
    int end_index;
    // handle the end block so that we add left-over elements too
    if(threadIdx.x == (num_threads-1)) {
        end_index = num_events;
    } else {
        end_index = start_index + num_elements_per_thread;
    }

    // Need to store the sum computed by each thread so in the end
    // a single thread can reduce to get the final sum
    __shared__ float temp_sums[NUM_THREADS];

    // Store the means in shared memory to speed up the covariance computations
    __shared__ float means[NUM_DIMENSIONS];
 
    // Compute new N
    for(int c=blockIdx.x; c<num_clusters; c += NUM_BLOCKS) {
        temp_sums[tid] = 0.0;
        // Break all the events accross the threads, add up probabilities
        for(int s=start_index; s<end_index; s++) {
            temp_sums[tid] += clusters[c].p[s];
        }
        
        __syncthreads();
        
        // Let the first thread add up all the intermediate sums
        if(tid == 0) {
            clusters[c].N = 0.0;
            for(int j=0; j<num_threads; j++) {
                clusters[c].N += temp_sums[j];
            }
            //printf("clusters[%d].N = %f\n",c,clusters[c].N);
            
            // Set PI to the # of expected items, and then normalize it later
            clusters[c].pi = clusters[c].N;
        }
    }

    // Synchronize because threads need to use clusters[c].N for means calculation    
    __syncthreads();

    float mean_sum;   
    float cov_sum = 0.0;
    int row,col,data_index;
   
    cluster* clust;
    // Compute means and covariances for each subcluster
    for(int c=blockIdx.x; c<num_clusters; c += NUM_BLOCKS) {
        clust = &(clusters[c]);
        
        // Compute means
        //  Let one thread handle each dimension
        //  There are only 8 cores per multiprocessor so I don't think we're really wasting
        //  resources badly by doing it this way. It's got alot fewer loops and potential branching
        //  than doing it like the N computation above
        if(tid < num_dimensions) {    
            mean_sum = 0.0;
            for(int s=0; s<num_events; s++) {
                mean_sum += fcs_data[s*num_dimensions+tid]*clust->p[s];
            }
            // Divide by # of elements in the cluster
            means[tid] = mean_sum / clust->N;
            clust->means[tid] = means[tid];
        }

        __syncthreads();

        // Compute the covariance matrix of the data
        for(int i=tid; i < num_elements; i+= num_threads) {
            // zero the value, find what row and col this thread is computing
            cov_sum = 0.0;
            row = (i) / num_dimensions;
            col = (i) % num_dimensions;
            data_index = 0;

            for(int j=0; j < num_events; j++) {
                cov_sum += (fcs_data[data_index+row]-means[row])*(fcs_data[data_index+col]-means[col])*clust->p[j]; 
                data_index += num_dimensions;
            }
            clust->R[i] = cov_sum / clust->N;
        }
        
        __syncthreads();

        // Regularize matrix
        if(tid < num_dimensions) {
            clust->R[tid*num_dimensions+tid] += clust->avgvar;
        }
    }
}

/*
 * Computes the constant for each cluster and normalizes pi for every cluster
 * In the process it inverts R and finds the determinant
 */
__global__ void
constants_kernel(cluster* clusters, int num_clusters, int num_dimensions) {
    compute_constants(clusters,num_clusters,num_dimensions);
    
    __syncthreads();
    
    if(blockIdx.x == 0) {
        normalize_pi(clusters,num_clusters);
    }
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
