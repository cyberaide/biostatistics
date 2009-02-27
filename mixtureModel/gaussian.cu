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

// Function prototypes
extern "C" float* readData(char* f, int* ndims, int*nevents);
int validateArguments(int argc, char** argv, int* num_clusters, int* target_num_clusters);
void writeCluster(FILE* f, cluster*c, int num_dimensions);
void printCluster(cluster* c, int num_dimensions);
float cluster_distance(cluster* cluster1, cluster* cluster2, cluster* temp_cluster, int num_dimensions);
void copy_cluster(cluster* dest, cluster* src, int num_dimensions);
void add_clusters(cluster* cluster1, cluster* cluster2, cluster* temp_cluster, int num_dimensions);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) {
    int original_num_clusters, desired_num_clusters, stop_number;
   
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
    printf("Parsing input file...");
    fflush(stdout); 
    float* fcs_data = readData(argv[2],&num_dimensions,&num_events);
    printf("done\n");    

    if(!fcs_data) {
        printf("Error parsing input file. This could be due to an empty file ");
        printf("or an inconsistent number of dimensions. Aborting.\n");
        return 1;
    }
    
    printf("Number of events: %d\n",num_events);
    printf("Number of dimensions: %d\n\n",num_dimensions);
    
    printf("Starting with %d cluster(s), will stop at %d cluster(s).\n",original_num_clusters,stop_number);
   
    // Set the device to run on... 0 for GTX 260, 1 for Tesla C870
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
    
    int num_threads = NUM_THREADS;

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
        
        if(!clusters || !clusters[i].means || !clusters[i].R || !clusters[i].Rinv || !clusters[i].p) { 
            printf("ERROR: Could not allocate memory for clusters.\n"); 
            return 1; 
        }
    }
    // Used as a temporary cluster for combining clusters in "distance" computations
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
    // Here we're only concerned with the statistics of the cluster, so we don't need to malloc 'p' array
    cluster* saved_clusters = (cluster*)malloc(sizeof(cluster)*original_num_clusters);
    if(!saved_clusters) { printf("ERROR: Could not allocate memory for clusters.\n"); return 1; }
    for(int i=0; i<original_num_clusters;i++) {
        saved_clusters[i].N = 0.0;
        saved_clusters[i].pi = 0.0;
        saved_clusters[i].means = (float*) malloc(sizeof(float)*num_dimensions);
        saved_clusters[i].R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
        saved_clusters[i].Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
    }
   
    if(VERBOSE) {
        printf("Finished allocating memory on host for clusters.\n");
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
    
    if(VERBOSE) {
        printf("Finished allocating memory on device for clusters.\n");
    }

    unsigned int mem_size = num_dimensions*num_events*sizeof(float);
    
    double min_rissanen, rissanen;
    
    // allocate device memory for FCS data
    float* d_fcs_data;
    CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data, mem_size));
    if(VERBOSE) {
        printf("Finished allocating memory on device for FCS data.\n");
    }
    // copy FCS to device
    CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data, fcs_data, mem_size,cudaMemcpyHostToDevice) );

    if(VERBOSE) {
        printf("Finished copying FCS data to device.\n");
    }
    // Copy Cluster data to device
    CUDA_SAFE_CALL(cudaMemcpy(d_clusters,temp_clusters,sizeof(cluster)*original_num_clusters,cudaMemcpyHostToDevice));
    
    if(VERBOSE) {
        printf("Finished copying cluster data to device.\n");
    }
   
    //////////////// Initialization done, starting kernels //////////////// 
    if(VERBOSE) {
        printf("Invoking seed_clusters kernel...");
        fflush(stdout);
    }

    // seed_clusters sets initial pi values, 
    // finds the means / covariances and copies it to all the clusters
    seed_clusters<<< 1, num_threads >>>( d_fcs_data, d_clusters, num_dimensions, original_num_clusters, num_events);
    cudaThreadSynchronize();
    if(VERBOSE) {
        printf("done.\n"); 
    }
    if(VERBOSE) {
        printf("Invoking constants kernel...",num_threads);
        fflush(stdout);
    }
    // Computes the R matrix inverses, and the gaussian constant
    constants_kernel<<<NUM_BLOCKS, num_threads>>>(d_clusters,original_num_clusters,num_dimensions);
    cudaThreadSynchronize();
    if(VERBOSE) {
        printf("done.\n");
    }
    
    // copy clusters from the device
    CUDA_SAFE_CALL(cudaMemcpy(temp_clusters, d_clusters, sizeof(cluster)*original_num_clusters,cudaMemcpyDeviceToHost));
    
    // Calculate an epsilon value
    //int ndata_points = num_events*num_dimensions;
    float epsilon = (1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)*log((float)num_events*num_dimensions)*0.01;
    float likelihood, old_likelihood;
    
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
        // Regrouping means calculate a cluster membership probability
        // for each event and each cluster. Each event is independent,
        // so the events are distributed to different blocks 
        // (and hence different multiprocessors)
        if(VERBOSE) {
            printf("Invoking regroup kernel...");
            fflush(stdout);
        }
        regroup<<<NUM_BLOCKS, num_threads>>>(d_fcs_data,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
        cudaThreadSynchronize();
        if(VERBOSE) {
            printf("done.\n");
        }
        // check if kernel execution generated an error
        CUT_CHECK_ERROR("Kernel execution failed");

        // Copy the likelihood totals from each block, sum them up to get a total
        CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*NUM_BLOCKS,cudaMemcpyDeviceToHost));
        likelihood = 0.0;
        for(int i=0;i<NUM_BLOCKS;i++) {
            likelihood += likelihoods[i]; 
        }

        float change = epsilon*2;
        
        printf("Performing EM algorthm on %d clusters.\n",num_clusters);
        // This is the iterative loop for the EM algorithm.
        // It re-estimates parameters, re-computes constants, and then regroups the events
        // These steps keep repeating until the change in likelihood is less than some epsilon        
        while(fabs(change) > epsilon) {
            old_likelihood = likelihood;
            if(VERBOSE) {
                printf("Invoking reestimate_parameters kernel...",num_threads);
                fflush(stdout);
            }

            // This kernel computes a new N, means, and R based on the probabilities computed in regroup kernel
            reestimate_parameters<<<NUM_BLOCKS, num_threads>>>(d_fcs_data,d_clusters,num_dimensions,num_clusters,num_events);
            cudaThreadSynchronize();
            if(VERBOSE) {
                printf("done.\n");
            }
            
            if(VERBOSE) {
                printf("Invoking constants kernel...",num_threads);
                fflush(stdout);
            }
            // Inverts the R matrices, computes the constant, normalizes cluster probabilities
            constants_kernel<<<NUM_BLOCKS, num_threads>>>(d_clusters,num_clusters,num_dimensions);
            cudaThreadSynchronize();
            if(VERBOSE) {
                printf("done.\n");
            }

            // check if kernel execution generated an error
            CUT_CHECK_ERROR("Kernel execution failed");
        
            if(VERBOSE) {
                printf("Invoking regroup kernel...");
                fflush(stdout);
            }
            // Compute new cluster membership probabilities for all the events
            regroup<<<NUM_BLOCKS, num_threads>>>(d_fcs_data,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
            cudaThreadSynchronize();
            if(VERBOSE) {
                printf("done.\n");
            }
        
            // check if kernel execution generated an error
            CUT_CHECK_ERROR("Kernel execution failed");
        
            CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*NUM_BLOCKS,cudaMemcpyDeviceToHost));
            likelihood = 0.0;
            for(int i=0;i<NUM_BLOCKS;i++) {
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
        // Save the cluster data if the solution is better and the user didn't specify a desired number
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
                    // compute distance function between the 2 clusters
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
                //printf("Copying cluster %d to cluster %d\n",i+1,i);
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
    
    
    // Open up the output file
    FILE* outf = fopen(argv[3],"w");
    if(!outf) {
        printf("ERROR: Unable to open file '%s' for writing.\n",argv[3]);
    }

    // Print the clusters with the lowest rissanen score to the console and output file
    for(int c=0; c<ideal_num_clusters; c++) {
        // Output the final cluster stats to the console
        printf("-----------------------    Cluster #%d  ------------------------------\n",c);
        printCluster(&(saved_clusters[c]),num_dimensions);
        printf("\n\n");

        // Output the final cluster stats to the output file        
        fprintf(outf,"-----------------------    Cluster #%d  ------------------------------\n",c);
        writeCluster(outf,&(saved_clusters[c]),num_dimensions);
        fprintf(outf,"\n\n");
    }
    fclose(outf);
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

void writeCluster(FILE* f, cluster* c, int num_dimensions) {
    fprintf(f,"Probability: %f\n", c->pi);
    fprintf(f,"N: %f\n",c->N);
    fprintf(f,"Spectral Mean: ");
    for(int i=0; i<num_dimensions; i++){
        fprintf(f,"%.3f ",c->means[i]);
    }
    fprintf(f,"\n");

    fprintf(f,"\nR Matrix:\n");
    for(int i=0; i<num_dimensions; i++) {
        for(int j=0; j<num_dimensions; j++) {
            fprintf(f,"%.3f ", c->R[i*num_dimensions+j]);
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

void printCluster(cluster* c, int num_dimensions) {
    writeCluster(stdout,c,num_dimensions);
    /*
    printf("Probability: %f\n", c->pi);
    printf("N: %f\n",c->N);
    printf("Spectral Mean: ");
    for(int i=0; i<num_dimensions; i++){
        printf("%.3f ",c->means[i]);
    }
    printf("\n");

    printf("\nR Matrix:\n");
    for(int i=0; i<num_dimensions; i++) {
        for(int j=0; j<num_dimensions; j++) {
            printf("%.3f ", c->R[i*num_dimensions+j]);
        }
        printf("\n");
    }
    fflush(stdout);   
    fprintf(f,"\nR-inverse Matrix:\n");
    for(int i=0; i<num_dimensions; i++) {
        for(int j=0; j<num_dimensions; j++) {
            fprintf(f,"%.3f ", c->Rinv[i*num_dimensions+j]);
        }
        fprintf(f,"\n");
    } 
    */
}

float cluster_distance(cluster* cluster1, cluster* cluster2, cluster* temp_cluster, int num_dimensions) {
    float determinant;
    // Add the clusters together, this updates pi,means,R,N and stores in temp_cluster
    add_clusters(cluster1,cluster2,temp_cluster,num_dimensions);
    // Copy R to Rinv matrix
    memcpy(temp_cluster->Rinv,temp_cluster->R,sizeof(float)*num_dimensions*num_dimensions);
    // Invert the matrix
    invert_cpu(temp_cluster->Rinv,num_dimensions,&determinant);
    //invert_matrix(temp_cluster->Rinv,num_dimensions,&determinant);
    // Compute the constant
    temp_cluster->constant = (-num_dimensions)*0.5*logf(2*PI)-0.5*logf(fabs(determinant));
    
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
