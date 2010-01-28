/*
 * Gaussian Mixture Model Clustering wtih CUDA
 *
 * Author: Andrew Pangborn
 *
 * Department of Computer Engineering
 * Rochester Institute of Technology
 * 
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h> // for clock(), clock_t, CLOCKS_PER_SEC

// includes, project
#include <cutil.h>
#include "gaussian.h"
#include "invert_matrix.h"

// includes, kernels
#include <theta_kernel.cu>

#include <cublas.h>

// Function prototypes
extern "C" float* readData(char* f, int* ndims, int*nevents);
int validateArguments(int argc, char** argv, int* num_clusters, int* target_num_clusters);
void writeCluster(FILE* f, clusters_t clusters, int c,  int num_dimensions);
void printCluster(clusters_t clusters, int c, int num_dimensions);
float cluster_distance(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions);
void copy_cluster(clusters_t dest, int c_dest, clusters_t src, int c_src, int num_dimensions);
void add_clusters(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) {
    int original_num_clusters, desired_num_clusters, stop_number;
    
    // For profiling the seed kernel
    clock_t seed_start, seed_end, seed_total;

    // For profiling the regroup kernel
    clock_t regroup_start, regroup_end, regroup_total;
    int regroup_iterations = 0;
    
    // for profiling the reestimate_parameters kernel
    clock_t params_start, params_end, params_total;
    int params_iterations = 0;
    
    // for profiling the constants kernel
    clock_t constants_start, constants_end, constants_total;
    int constants_iterations = 0;
    
    // for profiling the GMM order reduction
    clock_t reduce_start, reduce_end, reduce_total;
    int reduce_iterations = 0;
    
    regroup_total = regroup_iterations = 0;
    params_total = params_iterations = 0;
    constants_total = constants_iterations = 0;
    reduce_total = reduce_iterations = 0;
    
    // Keep track of total time
    unsigned int total_timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&total_timer));
    CUT_SAFE_CALL(cutStartTimer(total_timer));
    
    // For profiling input parsing
    unsigned int io_timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&io_timer));
    
    // For CPU processing
    unsigned int cpu_timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&cpu_timer));

    // Keep track of gpu memcpying
    unsigned int memcpy_timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&memcpy_timer));
   
    CUT_SAFE_CALL(cutStartTimer(io_timer));
    // Validate the command-line arguments, parse # of clusters, etc 
    int error = validateArguments(argc,argv,&original_num_clusters,&desired_num_clusters);
    
    // Don't continue if we had a problem with the program arguments
    if(error) {
        return 1;
    }
    
    // Number of clusters to stop iterating at.
    if(desired_num_clusters == 0) {
        stop_number = 1;
    } else {
        stop_number = desired_num_clusters;
    }
    
    int num_dimensions;
    int num_events;
    
    // Read FCS data   
    PRINT("Parsing input file...");
    // This stores the data in a 1-D array with consecutive values being the dimensions from a single event
    // (num_events by num_dimensions matrix)
    float* fcs_data_by_event = readData(argv[2],&num_dimensions,&num_events);    

    if(!fcs_data_by_event) {
        printf("Error parsing input file. This could be due to an empty file ");
        printf("or an inconsistent number of dimensions. Aborting.\n");
        return 1;
    }
    
    // Transpose the event data (allows coalesced access pattern in E-step kernel)
    // This has consecutive values being from the same dimension of the data 
    // (num_dimensions by num_events matrix)
    float* fcs_data_by_dimension  = (float*) malloc(sizeof(float)*num_events*num_dimensions);
    
    for(int e=0; e<num_events; e++) {
        for(int d=0; d<num_dimensions; d++) {
            fcs_data_by_dimension[d*num_events+e] = fcs_data_by_event[e*num_dimensions+d];
        }
    }    

    CUT_SAFE_CALL(cutStopTimer(io_timer));
   
    PRINT("Number of events: %d\n",num_events);
    PRINT("Number of dimensions: %d\n\n",num_dimensions);
    
    PRINT("Starting with %d cluster(s), will stop at %d cluster(s).\n",original_num_clusters,stop_number);
   
    CUT_SAFE_CALL(cutStartTimer(cpu_timer));
    // Set the device to run on... 0 for GTX 260, 1 for Tesla C870 on oak
    int GPUCount;
    int device;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&GPUCount));
    if(GPUCount == 0) {
        PRINT("Only 1 CUDA device found, defaulting to it.\n");
        device = 0;
    } else if (GPUCount > 1 && DEVICE <= GPUCount) {
        PRINT("Multiple CUDA devices found, selecting device %d\n",DEVICE);
        device = DEVICE;
    } else {
        printf("Fatal Error: Unable to set device to %d, not enough GPUs.\n",DEVICE);
        exit(2);
    }
    CUDA_SAFE_CALL(cudaSetDevice(DEVICE));
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    //PRINT("\nUsing device - %s\n\n", prop.name);
    printf("\nUsing device - %s\n\n", prop.name);
    
    cublasStatus status;
    status = cublasInit();
    if(status != CUBLAS_STATUS_SUCCESS) {
        printf("!!! CUBLAS initialization error\n");
    }
    
    int num_threads = NUM_THREADS;

    // Setup the cluster data structures on host
    clusters_t clusters;
    clusters.N = (float*) malloc(sizeof(float)*original_num_clusters);
    clusters.pi = (float*) malloc(sizeof(float)*original_num_clusters);
    clusters.constant = (float*) malloc(sizeof(float)*original_num_clusters);
    clusters.avgvar = (float*) malloc(sizeof(float)*original_num_clusters);
    clusters.means = (float*) malloc(sizeof(float)*num_dimensions*original_num_clusters);
    clusters.R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
    clusters.Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
    clusters.memberships = (float*) malloc(sizeof(float)*num_events*original_num_clusters);
    if(!clusters.means || !clusters.R || !clusters.Rinv || !clusters.memberships) { 
        printf("ERROR: Could not allocate memory for clusters.\n"); 
        return 1; 
    }
    
    // Declare another set of clusters for saving the results of the best configuration
    clusters_t saved_clusters;
    saved_clusters.N = (float*) malloc(sizeof(float)*original_num_clusters);
    saved_clusters.pi = (float*) malloc(sizeof(float)*original_num_clusters);
    saved_clusters.constant = (float*) malloc(sizeof(float)*original_num_clusters);
    saved_clusters.avgvar = (float*) malloc(sizeof(float)*original_num_clusters);
    saved_clusters.means = (float*) malloc(sizeof(float)*num_dimensions*original_num_clusters);
    saved_clusters.R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
    saved_clusters.Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
    saved_clusters.memberships = (float*) malloc(sizeof(float)*num_events*original_num_clusters);
    if(!saved_clusters.means || !saved_clusters.R || !saved_clusters.Rinv || !saved_clusters.memberships) { 
        printf("ERROR: Could not allocate memory for clusters.\n"); 
        return 1; 
    }

    // Used as a temporary cluster for combining clusters in "distance" computations
    clusters_t scratch_cluster;
    scratch_cluster.N = (float*) malloc(sizeof(float));
    scratch_cluster.pi = (float*) malloc(sizeof(float));
    scratch_cluster.constant = (float*) malloc(sizeof(float));
    scratch_cluster.avgvar = (float*) malloc(sizeof(float));
    scratch_cluster.means = (float*) malloc(sizeof(float)*num_dimensions);
    scratch_cluster.R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
    scratch_cluster.Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
    scratch_cluster.memberships = (float*) malloc(sizeof(float)*num_events);

    DEBUG("Finished allocating memory on host for clusters.\n");
    CUT_SAFE_CALL(cutStopTimer(cpu_timer));
    
    // Setup the cluster data structures on device
    // First allocate structures on the host, CUDA malloc the arrays
    // Then CUDA malloc structures on the device and copy them over
    CUT_SAFE_CALL( cutStartTimer(memcpy_timer));
    clusters_t temp_clusters;
    CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.N),sizeof(float)*original_num_clusters));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.pi),sizeof(float)*original_num_clusters));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.constant),sizeof(float)*original_num_clusters));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.avgvar),sizeof(float)*original_num_clusters));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.means),sizeof(float)*num_dimensions*original_num_clusters));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.R),sizeof(float)*num_dimensions*num_dimensions*original_num_clusters));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.Rinv),sizeof(float)*num_dimensions*num_dimensions*original_num_clusters));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.memberships),sizeof(float)*num_events*original_num_clusters));
   
    // Allocate a struct on the device 
    clusters_t* d_clusters;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_clusters, sizeof(clusters_t)));
    DEBUG("Finished allocating memory on device for clusters.\n");
    
    // Copy Cluster data to device
    CUDA_SAFE_CALL(cudaMemcpy(d_clusters,&temp_clusters,sizeof(clusters_t),cudaMemcpyHostToDevice));
    CUT_SAFE_CALL( cutStopTimer(memcpy_timer));
    DEBUG("Finished copying cluster data to device.\n");

    // for cublas result for means: M x D
    //float* h_C = (float*) calloc(num_dimensions*original_num_clusters,sizeof(float));
    //float* d_C;
    //CUDA_SAFE_CALL(cudaMalloc((void**) &(d_C), sizeof(float)*original_num_clusters*num_dimensions)); 
    //CUDA_SAFE_CALL(cudaMemcpy(d_C,h_C,sizeof(float)*num_dimensions*original_num_clusters,cudaMemcpyHostToDevice));
    
    int mem_size = num_dimensions*num_events*sizeof(float);
    
    float min_rissanen, rissanen;
    
    // allocate device memory for FCS data
    float* d_fcs_data_by_event;
    float* d_fcs_data_by_dimension;
    CUT_SAFE_CALL( cutStartTimer(memcpy_timer));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_event, mem_size));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_dimension, mem_size));
    DEBUG("Finished allocating memory on device for clusters.\n");
    // copy FCS to device
    CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_event, fcs_data_by_event, mem_size,cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_dimension, fcs_data_by_dimension, mem_size,cudaMemcpyHostToDevice) );
    CUT_SAFE_CALL( cutStopTimer(memcpy_timer));
    DEBUG("Finished copying FCS data to device.\n");
    
   
    //////////////// Initialization done, starting kernels //////////////// 
    DEBUG("Invoking seed_clusters kernel...");
    fflush(stdout);

    // seed_clusters sets initial pi values, 
    // finds the means / covariances and copies it to all the clusters
    // TODO: Does it make any sense to use multiple blocks for this?
    seed_start = clock();
    seed_clusters<<< 1, 256 >>>( d_fcs_data_by_event, d_clusters, num_dimensions, original_num_clusters, num_events);
    cudaThreadSynchronize();
    DEBUG("done.\n"); 
    CUT_CHECK_ERROR("Seed Kernel execution failed: ");
   
    DEBUG("Invoking constants kernel...",num_threads);
    // Computes the R matrix inverses, and the gaussian constant
    constants_kernel<<<original_num_clusters, num_threads>>>(d_clusters,original_num_clusters,num_dimensions);
    cudaThreadSynchronize();
    CUT_CHECK_ERROR("Constants Kernel execution failed: ");
    DEBUG("done.\n");
    seed_end = clock();
    seed_total = seed_end - seed_start;
    
    // Calculate an epsilon value
    //int ndata_points = num_events*num_dimensions;
    float epsilon = (1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)*log((float)num_events*num_dimensions)*0.01;
    float likelihood, old_likelihood;
    int iters;
    
    //epsilon = 1e-6;
    PRINT("Gaussian.cu: epsilon = %f\n",epsilon);

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
        // Regrouping means calculate a cluster membership probability
        // for each event and each cluster. Each event is independent,
        // so the events are distributed to different blocks 
        // (and hence different multiprocessors)
        DEBUG("Invoking regroup (E-step) kernel with %d blocks...",NUM_BLOCKS);
        regroup_start = clock();
        //regroup<<<NUM_BLOCKS, NUM_THREADS>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
        regroup1<<<dim3(NUM_BLOCKS,num_clusters), NUM_THREADS>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_events,d_likelihoods);
        regroup2<<<NUM_BLOCKS, NUM_THREADS>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
        cudaThreadSynchronize();
        regroup_end = clock();
        regroup_total += regroup_end - regroup_start;
        regroup_iterations++;
        DEBUG("done.\n");
        DEBUG("Regroup Kernel Iteration Time: %f\n\n",((double)(regroup_end-regroup_start))/CLOCKS_PER_SEC);
        // check if kernel execution generated an error
        CUT_CHECK_ERROR("Kernel execution failed");


        // Copy the likelihood totals from each block, sum them up to get a total
        CUT_SAFE_CALL( cutStartTimer(memcpy_timer));
        CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*NUM_BLOCKS,cudaMemcpyDeviceToHost));
        CUT_SAFE_CALL( cutStopTimer(memcpy_timer));
        CUT_SAFE_CALL(cutStartTimer(cpu_timer));
        likelihood = 0.0;
        for(int i=0;i<NUM_BLOCKS;i++) {
            likelihood += likelihoods[i]; 
        }
        DEBUG("Likelihood: %e\n",likelihood);
        CUT_SAFE_CALL(cutStopTimer(cpu_timer));

        float change = epsilon*2;
        
        PRINT("Performing EM algorithm on %d clusters.\n",num_clusters);
        iters = 0;
        // This is the iterative loop for the EM algorithm.
        // It re-estimates parameters, re-computes constants, and then regroups the events
        // These steps keep repeating until the change in likelihood is less than some epsilon        
        while(fabs(change) > epsilon && iters < MAX_ITERS) {
            old_likelihood = likelihood;
            
            DEBUG("Invoking reestimate_parameters (M-step) kernel...",num_threads);
            params_start = clock();
            // This kernel computes a new N, pi isn't updated until compute_constants though
            mstep_N<<<num_clusters, 256>>>(d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events);
            cudaThreadSynchronize();
            CUT_SAFE_CALL( cutStartTimer(memcpy_timer));
            CUDA_SAFE_CALL(cudaMemcpy(&temp_clusters,d_clusters,sizeof(clusters_t),cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(clusters.N,temp_clusters.N,sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
            CUT_SAFE_CALL( cutStopTimer(memcpy_timer));
            // This kernel computes new means
            //mstep_means1<<<num_clusters, NUM_THREADS>>>(d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events);

            dim3 gridDim1(num_clusters,num_dimensions);
            mstep_means2<<<gridDim1, NUM_THREADS>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events);
            cudaThreadSynchronize();
            CUT_SAFE_CALL( cutStartTimer(memcpy_timer));
            CUDA_SAFE_CALL(cudaMemcpy(clusters.means,temp_clusters.means,sizeof(float)*num_clusters*num_dimensions,cudaMemcpyDeviceToHost));
            CUT_SAFE_CALL( cutStopTimer(memcpy_timer));
            
            /*
            // Alternative way to calculate the means via matrix multiplication
            // Means = Data * P + 0
            //  Data is fcs events, a [N x D] (fcs_data_by_event) matrix, but [D x N] in column major
            //  P is the memberships, a [M x N], but [N x M] in column-major
            //  Thus we get an [D x M] result for the means in column-major format, so it's actually [M x D] in row-major
            /// This is slower than my mstep_means kernel when either D or M is relatively small (like 16 and 32).
            //cublasSgemm('t','t',num_clusters,num_dimensions,num_events,1.0,temp_clusters.memberships,num_events,d_fcs_data_by_event,num_dimensions,0.0,d_C,num_clusters);
            cublasSgemm('n','n',num_dimensions,num_clusters,num_events,1.0,d_fcs_data_by_event,num_dimensions,temp_clusters.memberships,num_events,0.0,d_C,num_dimensions);
            cudaThreadSynchronize();
            status = cublasGetError();
            if(status != CUBLAS_STATUS_SUCCESS) {  
                printf("Cublas kernel error!\n");
                return 1;
            }
            CUDA_SAFE_CALL(cudaMemcpy(h_C,d_C,sizeof(float)*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
            //printf("\n");
            for(int i=0; i < num_clusters; i++) {
                for(int j=0; j < num_dimensions; j++) {
                    //clusters.means[i*num_dimensions+j] = h_C[j*num_clusters+i] / clusters.N[i];
                    clusters.means[i*num_dimensions+j] = h_C[i*num_dimensions+j] / clusters.N[i];
                    //printf("%.2f ",clusters.means[i*num_dimensions+j]);
                }
                //printf("\n");
            }
            //CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.means,clusters.means,sizeof(float)*num_dimensions*num_clusters,cudaMemcpyHostToDevice));
            */

            // Covariance is symmetric, so we only need to compute N*(N+1)/2 matrix elements per cluster
            dim3 gridDim2(num_clusters,num_dimensions*(num_dimensions+1)/2);
            //dim3 gridDim2(num_clusters,num_dimensions*num_dimensions);
            mstep_covariance1<<<gridDim2, NUM_THREADS>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events);
            //return 1;
            
            // This method is fast on a 1.3 device with higher dimensionality (where num_dimensions^2 >= NUM_THREADS)
            //  since each thread handles one spot in the matrix, low d data means low threads and wasted resources
            //mstep_covariance2<<<num_clusters, NUM_THREADS>>>(d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events);
            
            cudaThreadSynchronize();
            params_end = clock();
            CUT_CHECK_ERROR("M-step Kernel execution failed: ");
            params_total += params_end - params_start;
            params_iterations++;
            DEBUG("done.\n");
            DEBUG("Model Parameters Kernel Iteration Time: %f\n\n",((double)(params_end-params_start))/CLOCKS_PER_SEC);
            //return 0; // RETURN FOR FASTER PROFILING
            
            DEBUG("Invoking constants kernel...",num_threads);
            // Inverts the R matrices, computes the constant, normalizes cluster probabilities
            constants_start = clock();
            constants_kernel<<<num_clusters, NUM_THREADS>>>(d_clusters,num_clusters,num_dimensions);
            cudaThreadSynchronize();
            constants_end = clock();
            CUT_CHECK_ERROR("Constants Kernel execution failed: ");
            constants_total += constants_end - constants_start;
            constants_iterations++;
            DEBUG("done.\n");
            DEBUG("Constants Kernel Iteration Time: %f\n\n",((double)(constants_end-constants_start))/CLOCKS_PER_SEC);

            DEBUG("Invoking regroup (E-step) kernel with %d blocks...",NUM_BLOCKS);
            regroup_start = clock();
            // Compute new cluster membership probabilities for all the events
            //regroup<<<NUM_BLOCKS, NUM_THREADS>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
            regroup1<<<dim3(NUM_BLOCKS,num_clusters), NUM_THREADS>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_events,d_likelihoods);
            regroup2<<<NUM_BLOCKS, NUM_THREADS>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
            cudaThreadSynchronize();
            CUT_CHECK_ERROR("E-step Kernel execution failed: ");
            regroup_end = clock();
            regroup_total += regroup_end - regroup_start;
            regroup_iterations++;
            DEBUG("done.\n");
            DEBUG("Regroup Kernel Iteration Time: %f\n\n",((double)(regroup_end-regroup_start))/CLOCKS_PER_SEC);
        
            // check if kernel execution generated an error
            CUT_CHECK_ERROR("Kernel execution failed");
        
            CUT_SAFE_CALL( cutStartTimer(memcpy_timer));
            CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*NUM_BLOCKS,cudaMemcpyDeviceToHost));
            CUT_SAFE_CALL( cutStopTimer(memcpy_timer));
            CUT_SAFE_CALL(cutStartTimer(cpu_timer));
            likelihood = 0.0;
            for(int i=0;i<NUM_BLOCKS;i++) {
                likelihood += likelihoods[i]; 
            }
            
            change = likelihood - old_likelihood;
            DEBUG("likelihood = %f\n",likelihood);
            DEBUG("Change in likelihood: %f\n",change);

            iters++;
            CUT_SAFE_CALL(cutStopTimer(cpu_timer));
            

            /*
            // copy clusters from the device
            CUDA_SAFE_CALL(cudaMemcpy(temp_clusters, d_clusters, sizeof(cluster)*num_clusters,cudaMemcpyDeviceToHost));
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
            for(int i=0; i<num_clusters; i++) {
                printf("N: %f, mean: %f, variance: %f, Rinv: %f\n",clusters[i].N,clusters[i].means[0],clusters[i].R[0],clusters[i].Rinv[0]);
            }
            printf("\n\n");
            */

        }
        
        // copy clusters from the device
        CUT_SAFE_CALL( cutStartTimer(memcpy_timer));
        CUDA_SAFE_CALL(cudaMemcpy(&temp_clusters, d_clusters, sizeof(clusters_t),cudaMemcpyDeviceToHost));
        // copy all of the arrays from the structs
        CUDA_SAFE_CALL(cudaMemcpy(clusters.N, temp_clusters.N, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(clusters.pi, temp_clusters.pi, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(clusters.constant, temp_clusters.constant, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(clusters.avgvar, temp_clusters.avgvar, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(clusters.means, temp_clusters.means, sizeof(float)*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(clusters.R, temp_clusters.R, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(clusters.Rinv, temp_clusters.Rinv, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(clusters.memberships, temp_clusters.memberships, sizeof(float)*num_events*num_clusters,cudaMemcpyDeviceToHost));
        CUT_SAFE_CALL( cutStopTimer(memcpy_timer));
        
        // Calculate Rissanen Score
        CUT_SAFE_CALL(cutStartTimer(cpu_timer));
        rissanen = -likelihood + 0.5*(num_clusters*(1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)-1)*logf((float)num_events*num_dimensions);
        PRINT("\nRissanen Score: %e\n",rissanen);
        
        
        // Save the cluster data the first time through, so we have a base rissanen score and result
        // Save the cluster data if the solution is better and the user didn't specify a desired number
        // If the num_clusters equals the desired number, stop
        if(num_clusters == original_num_clusters || (rissanen < min_rissanen && desired_num_clusters == 0) || (num_clusters == desired_num_clusters)) {
            min_rissanen = rissanen;
            ideal_num_clusters = num_clusters;
            // Save the cluster configuration somewhere
            memcpy(saved_clusters.N,clusters.N,sizeof(float)*num_clusters);
            memcpy(saved_clusters.pi,clusters.pi,sizeof(float)*num_clusters);
            memcpy(saved_clusters.constant,clusters.constant,sizeof(float)*num_clusters);
            memcpy(saved_clusters.avgvar,clusters.avgvar,sizeof(float)*num_clusters);
            memcpy(saved_clusters.means,clusters.means,sizeof(float)*num_dimensions*num_clusters);
            memcpy(saved_clusters.R,clusters.R,sizeof(float)*num_dimensions*num_dimensions*num_clusters);
            memcpy(saved_clusters.Rinv,clusters.Rinv,sizeof(float)*num_dimensions*num_dimensions*num_clusters);
            memcpy(saved_clusters.memberships,clusters.memberships,sizeof(float)*num_events*num_clusters);
        }
        CUT_SAFE_CALL(cutStopTimer(cpu_timer));

        
        /**************** Reduce GMM Order ********************/
        reduce_start = clock();
        // Don't want to reduce order on the last iteration
        if(num_clusters > stop_number) {
            // First eliminate any "empty" clusters 
            for(int i=num_clusters-1; i >= 0; i--) {
                if(clusters.N[i] < 1.0) {
                    DEBUG("Cluster #%d has less than 1 data point in it.\n",i);
                    for(int j=i; j < num_clusters-1; j++) {
                        copy_cluster(clusters,j,clusters,j+1,num_dimensions);
                    }
                    num_clusters--;
                }
            }
            
            min_c1 = 0;
            min_c2 = 1;
            DEBUG("Number of non-empty clusters: %d\n",num_clusters); 
            // For all combinations of subclasses...
            // If the number of clusters got really big might need to do a non-exhaustive search
            // Even with 100*99/2 combinations this doesn't seem to take too long
            for(int c1=0; c1<num_clusters;c1++) {
                for(int c2=c1+1; c2<num_clusters;c2++) {
                    // compute distance function between the 2 clusters
                    distance = cluster_distance(clusters,c1,c2,scratch_cluster,num_dimensions);
                    
                    // Keep track of minimum distance
                    if((c1 ==0 && c2 == 1) || distance < min_distance) {
                        min_distance = distance;
                        min_c1 = c1;
                        min_c2 = c2;
                    }
                }
            }

            PRINT("\nMinimum distance between (%d,%d). Combining clusters\n",min_c1,min_c2);
            // Add the two clusters with min distance together
            //add_clusters(&(clusters[min_c1]),&(clusters[min_c2]),scratch_cluster,num_dimensions);
            add_clusters(clusters,min_c1,min_c2,scratch_cluster,num_dimensions);
            // Copy new combined cluster into the main group of clusters, compact them
            //copy_cluster(&(clusters[min_c1]),scratch_cluster,num_dimensions);
            copy_cluster(clusters,min_c1,scratch_cluster,0,num_dimensions);
            for(int i=min_c2; i < num_clusters-1; i++) {
                //printf("Copying cluster %d to cluster %d\n",i+1,i);
                //copy_cluster(&(clusters[i]),&(clusters[i+1]),num_dimensions);
                copy_cluster(clusters,i,clusters,i+1,num_dimensions);
            }

            // Copy the clusters back to the device
            CUT_SAFE_CALL( cutStartTimer(memcpy_timer));
            CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.N, clusters.N, sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.pi, clusters.pi, sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.constant, clusters.constant, sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.avgvar, clusters.avgvar, sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.means, clusters.means, sizeof(float)*num_dimensions*num_clusters,cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.R, clusters.R, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.Rinv, clusters.Rinv, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.memberships, clusters.memberships, sizeof(float)*num_events*num_clusters,cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(d_clusters,&temp_clusters,sizeof(clusters_t),cudaMemcpyHostToDevice));
            CUT_SAFE_CALL( cutStopTimer(memcpy_timer));
        } // GMM reduction block 
        reduce_end = clock();
        reduce_total += reduce_end - reduce_start;
        reduce_iterations++;
    } // outer loop from M to 1 clusters
    PRINT("\nFinal rissanen Score was: %f, with %d clusters.\n",min_rissanen,ideal_num_clusters);
 
    
    // copy clusters from the device
    CUT_SAFE_CALL( cutStartTimer(memcpy_timer));
    CUDA_SAFE_CALL(cudaMemcpy(&temp_clusters, d_clusters, sizeof(clusters_t),cudaMemcpyDeviceToHost));
    CUT_SAFE_CALL( cutStopTimer(memcpy_timer));
    
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
    
    
    
    CUT_SAFE_CALL(cutStartTimer(cpu_timer));
    char* result_suffix = ".results";
    char* summary_suffix = ".summary";
    int filenamesize1 = strlen(argv[3]) + strlen(result_suffix) + 1;
    int filenamesize2 = strlen(argv[3]) + strlen(summary_suffix) + 1;
    char* result_filename = (char*) malloc(filenamesize1);
    char* summary_filename = (char*) malloc(filenamesize2);
    strcpy(result_filename,argv[3]);
    strcpy(summary_filename,argv[3]);
    strcat(result_filename,result_suffix);
    strcat(summary_filename,summary_suffix);
    
    PRINT("Summary filename: %s\n",summary_filename);
    PRINT("Results filename: %s\n",result_filename);
    CUT_SAFE_CALL(cutStopTimer(cpu_timer));
    
    CUT_SAFE_CALL(cutStartTimer(io_timer));
    // Open up the output file for cluster summary
    FILE* outf = fopen(summary_filename,"w");
    if(!outf) {
        printf("ERROR: Unable to open file '%s' for writing.\n",argv[3]);
    }

    // Print the clusters with the lowest rissanen score to the console and output file
    for(int c=0; c<ideal_num_clusters; c++) {
        //if(saved_clusters.N[c] == 0.0) {
        //    continue;
        //}
        if(ENABLE_PRINT) {
            // Output the final cluster stats to the console
            PRINT("Cluster #%d\n",c);
            printCluster(saved_clusters,c,num_dimensions);
            PRINT("\n\n");
        }

        if(ENABLE_OUTPUT) {
            // Output the final cluster stats to the output file        
            fprintf(outf,"Cluster #%d\n",c);
            writeCluster(outf,saved_clusters,c,num_dimensions);
            fprintf(outf,"\n\n");
        }
    }
    
    // Print profiling information
    printf("Program Component\tTotal\tIters\tTime Per Iteration\n");
    printf("        Seed Kernel:\t%7.4f\t%d\t%7.4f\n",seed_total/(double)CLOCKS_PER_SEC,1, (double) seed_total / (double) CLOCKS_PER_SEC);
    printf("      E-step Kernel:\t%7.4f\t%d\t%7.4f\n",regroup_total/(double)CLOCKS_PER_SEC,regroup_iterations, (double) regroup_total / (double) CLOCKS_PER_SEC / (double) regroup_iterations);
    printf("      M-step Kernel:\t%7.4f\t%d\t%7.4f\n",params_total/(double)CLOCKS_PER_SEC,params_iterations, (double) params_total / (double) CLOCKS_PER_SEC / (double) params_iterations);
    printf("   Constants Kernel:\t%7.4f\t%d\t%7.4f\n",constants_total/(double)CLOCKS_PER_SEC,constants_iterations, (double) constants_total / (double) CLOCKS_PER_SEC / (double) constants_iterations);    
    printf("GMM Order Reduction:\t%7.4f\t%d\t%7.4f\n",reduce_total/(double)CLOCKS_PER_SEC,reduce_iterations, (double) reduce_total / (double) CLOCKS_PER_SEC / (double) reduce_iterations);
   
    // Write profiling info to summary file
    fprintf(outf,"Program Component\tTotal\tIters\tTime Per Iteration\n");
    fprintf(outf,"        Seed Kernel:\t%7.4f\t%d\t%7.4f\n",seed_total/(double)CLOCKS_PER_SEC,1, (double) seed_total / (double) CLOCKS_PER_SEC);
    fprintf(outf,"      E-step Kernel:\t%7.4f\t%d\t%7.4f\n",regroup_total/(double)CLOCKS_PER_SEC,regroup_iterations, (double) regroup_total / (double) CLOCKS_PER_SEC / (double) regroup_iterations);
    fprintf(outf,"      M-step Kernel:\t%7.4f\t%d\t%7.4f\n",params_total/(double)CLOCKS_PER_SEC,params_iterations, (double) params_total / (double) CLOCKS_PER_SEC / (double) params_iterations);
    fprintf(outf,"   Constants Kernel:\t%7.4f\t%d\t%7.4f\n",constants_total/(double)CLOCKS_PER_SEC,constants_iterations, (double) constants_total / (double) CLOCKS_PER_SEC / (double) constants_iterations);    
    fprintf(outf,"GMM Order Reduction:\t%7.4f\t%d\t%7.4f\n",reduce_total/(double)CLOCKS_PER_SEC,reduce_iterations, (double) reduce_total / (double) CLOCKS_PER_SEC / (double) reduce_iterations);
    fclose(outf);
    
    
    // Open another output file for the event level clustering results
    FILE* fresults = fopen(result_filename,"w");
   
    if(ENABLE_OUTPUT) { 
        for(int i=0; i<num_events; i++) {
            for(int d=0; d<num_dimensions-1; d++) {
                fprintf(fresults,"%f,",fcs_data_by_event[i*num_dimensions+d]);
            }
            fprintf(fresults,"%f",fcs_data_by_event[i*num_dimensions+num_dimensions-1]);
            fprintf(fresults,"\t");
            for(int c=0; c<ideal_num_clusters-1; c++) {
                fprintf(fresults,"%f,",saved_clusters.memberships[c*num_events+i]);
            }
            fprintf(fresults,"%f",saved_clusters.memberships[(ideal_num_clusters-1)*num_events+i]);
            fprintf(fresults,"\n");
        }
    }
    fclose(fresults); 
    CUT_SAFE_CALL(cutStopTimer(io_timer));
    printf("\n");
    printf( "I/O time: %f (ms)\n", cutGetTimerValue(io_timer));
    CUT_SAFE_CALL(cutDeleteTimer(io_timer));
    
    printf( "Memcpy time: %f (ms)\n", cutGetTimerValue(memcpy_timer));
    CUT_SAFE_CALL(cutDeleteTimer(memcpy_timer));
    
    printf( "CPU processing time: %f (ms)\n", cutGetTimerValue(cpu_timer));
    CUT_SAFE_CALL(cutDeleteTimer(cpu_timer));

    /// Print out the total program time
    CUT_SAFE_CALL(cutStopTimer(total_timer));
    printf( "Total time: %f (ms)\n", cutGetTimerValue(total_timer));
    CUT_SAFE_CALL(cutDeleteTimer(total_timer));
 
    // cleanup host memory
    free(fcs_data_by_event);
    free(fcs_data_by_dimension);
    free(clusters.N);
    free(clusters.pi);
    free(clusters.constant);
    free(clusters.avgvar);
    free(clusters.means);
    free(clusters.R);
    free(clusters.Rinv);
    free(clusters.memberships);

    free(saved_clusters.N);
    free(saved_clusters.pi);
    free(saved_clusters.constant);
    free(saved_clusters.avgvar);
    free(saved_clusters.means);
    free(saved_clusters.R);
    free(saved_clusters.Rinv);
    free(saved_clusters.memberships);
    
    free(scratch_cluster.N);
    free(scratch_cluster.pi);
    free(scratch_cluster.constant);
    free(scratch_cluster.avgvar);
    free(scratch_cluster.means);
    free(scratch_cluster.R);
    free(scratch_cluster.Rinv);
    free(scratch_cluster.memberships);
   
    free(likelihoods);

    // cleanup GPU memory
    CUDA_SAFE_CALL(cudaFree(d_likelihoods));
 
    CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_event));
    CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_dimension));

    CUDA_SAFE_CALL(cudaFree(temp_clusters.N));
    CUDA_SAFE_CALL(cudaFree(temp_clusters.pi));
    CUDA_SAFE_CALL(cudaFree(temp_clusters.constant));
    CUDA_SAFE_CALL(cudaFree(temp_clusters.avgvar));
    CUDA_SAFE_CALL(cudaFree(temp_clusters.means));
    CUDA_SAFE_CALL(cudaFree(temp_clusters.R));
    CUDA_SAFE_CALL(cudaFree(temp_clusters.Rinv));
    CUDA_SAFE_CALL(cudaFree(temp_clusters.memberships));
    CUDA_SAFE_CALL(cudaFree(d_clusters));

    return 0;
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
        //FILE* outfile = fopen(argv[3],"w");
        //if(!outfile) {
        //    printf("Unable to create output file.\n\n");
        //    printUsage(argv);
        //    return 3;
        //}        
        // parse target_num_clusters
        if(argc == 5) {
            if(!sscanf(argv[4],"%d",target_num_clusters)) {
                printf("Invalid number of desired clusters.\n\n");
                printUsage(argv);
                return 4;
            }
            if(*target_num_clusters > *num_clusters) {
                printf("target_num_clusters must be less than equal to num_clusters\n\n");
                printUsage(argv);
                return 4;
            }
        } else {
            *target_num_clusters = 0;
        }
        
        // Clean up so the EPA is happy
        fclose(infile);
        //fclose(outfile);
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

void writeCluster(FILE* f, clusters_t clusters, int c, int num_dimensions) {
    fprintf(f,"Probability: %f\n", clusters.pi[c]);
    fprintf(f,"N: %f\n",clusters.N[c]);
    fprintf(f,"Means: ");
    for(int i=0; i<num_dimensions; i++){
        fprintf(f,"%.3f ",clusters.means[c*num_dimensions+i]);
    }
    fprintf(f,"\n");

    fprintf(f,"\nR Matrix:\n");
    for(int i=0; i<num_dimensions; i++) {
        for(int j=0; j<num_dimensions; j++) {
            fprintf(f,"%.3f ", clusters.R[c*num_dimensions*num_dimensions+i*num_dimensions+j]);
        }
        fprintf(f,"\n");
    }
    fflush(f);   
    /*
    fprintf(f,"\nR-inverse Matrix:\n");
    for(int i=0; i<num_dimensions; i++) {
        for(int j=0; j<num_dimensions; j++) {
            fprintf(f,"%.3f ", c->Rinv[i*num_dimensions+j]);
        }
        fprintf(f,"\n");
    } 
    */
}

void printCluster(clusters_t clusters, int c, int num_dimensions) {
    writeCluster(stdout,clusters,c,num_dimensions);
}

float cluster_distance(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions) {
    // Add the clusters together, this updates pi,means,R,N and stores in temp_cluster
    add_clusters(clusters,c1,c2,temp_cluster,num_dimensions);
    
    return clusters.N[c1]*clusters.constant[c1] + clusters.N[c2]*clusters.constant[c2] - temp_cluster.N[0]*temp_cluster.constant[0];
}

void add_clusters(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions) {
    float wt1,wt2;
 
    wt1 = (clusters.N[c1]) / (clusters.N[c1] + clusters.N[c2]);
    wt2 = 1.0f - wt1;
    
    // Compute new weighted means
    for(int i=0; i<num_dimensions;i++) {
        temp_cluster.means[i] = wt1*clusters.means[c1*num_dimensions+i] + wt2*clusters.means[c2*num_dimensions+i];
    }
    
    // Compute new weighted covariance
    for(int i=0; i<num_dimensions; i++) {
        for(int j=i; j<num_dimensions; j++) {
            // Compute R contribution from cluster1
            temp_cluster.R[i*num_dimensions+j] = ((temp_cluster.means[i]-clusters.means[c1*num_dimensions+i])
                                                *(temp_cluster.means[j]-clusters.means[c1*num_dimensions+j])
                                                +clusters.R[c1*num_dimensions*num_dimensions+i*num_dimensions+j])*wt1;
            // Add R contribution from cluster2
            temp_cluster.R[i*num_dimensions+j] += ((temp_cluster.means[i]-clusters.means[c2*num_dimensions+i])
                                                    *(temp_cluster.means[j]-clusters.means[c2*num_dimensions+j])
                                                    +clusters.R[c2*num_dimensions*num_dimensions+i*num_dimensions+j])*wt2;
            // Because its symmetric...
            temp_cluster.R[j*num_dimensions+i] = temp_cluster.R[i*num_dimensions+j];
        }
    }
    
    // Compute pi
    temp_cluster.pi[0] = clusters.pi[c1] + clusters.pi[c2];
    
    // compute N
    temp_cluster.N[0] = clusters.N[c1] + clusters.N[c2];

    float log_determinant;
    // Copy R to Rinv matrix
    memcpy(temp_cluster.Rinv,temp_cluster.R,sizeof(float)*num_dimensions*num_dimensions);
    // Invert the matrix
    invert_cpu(temp_cluster.Rinv,num_dimensions,&log_determinant);
    // Compute the constant
    temp_cluster.constant[0] = (-num_dimensions)*0.5*logf(2*PI)-0.5*log_determinant;
    
    // avgvar same for all clusters
    temp_cluster.avgvar[0] = clusters.avgvar[0];
}

void copy_cluster(clusters_t dest, int c_dest, clusters_t src, int c_src, int num_dimensions) {
    dest.N[c_dest] = src.N[c_src];
    dest.pi[c_dest] = src.pi[c_src];
    dest.constant[c_dest] = src.constant[c_src];
    dest.avgvar[c_dest] = src.avgvar[c_src];
    memcpy(&(dest.means[c_dest*num_dimensions]),&(src.means[c_src*num_dimensions]),sizeof(float)*num_dimensions);
    memcpy(&(dest.R[c_dest*num_dimensions*num_dimensions]),&(src.R[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
    memcpy(&(dest.Rinv[c_dest*num_dimensions*num_dimensions]),&(src.Rinv[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
    // do we need to copy memberships?
}
