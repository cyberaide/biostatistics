#define COVARIANCE_DYNAMIC_RANGE 1E6

void seed_clusters(float *data, clusters_t* clusters, int D, int M, int N) {
    float* variances = (float*) malloc(sizeof(float)*D);
    float* means = (float*) malloc(sizeof(float)*D);

    // Compute means
    for(int d=0; d < D; d++) {
        means[d] = 0.0;
        for(int n=0; n < N; n++) {
            means[d] += data[n*D+d];
        }
        means[d] /= (float) N;
    }

    // Compute variance of each dimension
    for(int d=0; d < D; d++) {
        variances[d] = 0.0;
        for(int n=0; n < N; n++) {
            variances[d] += data[n*D+d]*data[n*D+d];
        }
        variances[d] /= (float) N;
        variances[d] -= means[d]*means[d];
    }

    // Average variance
    float avgvar = 0.0;
    for(int d=0; d < D; d++) {
        avgvar += variances[d];
    }
    avgvar /= (float) D;
    
    float seed;
    if(M > 1) {
        seed = (N-1.0f)/(M-1.0f);
    } else {
        seed = 0.0f;
    }

    for(int m=0; m < M; m++) {
        clusters->N[m] = (float) N / (float) M;
        clusters->pi[m] = 1.0f / (float) M;
        clusters->avgvar[m] = avgvar / COVARIANCE_DYNAMIC_RANGE;

        DEBUG("N: %.2f\tPi: %.2f\tAvgvar: %e\n",clusters->N[m],clusters->pi[m],clusters->avgvar[m]);

        // Choose cluster centers
        DEBUG("Means: ");
        for(int d=0; d < D; d++) {
            clusters->means[m*D+d] = data[(int)(((float)m)*seed)*D+d];
            DEBUG("%.2f ",clusters->means[m*D+d]);
        }
        DEBUG("\n");

        // Set covariances to identity matrices
        for(int i=0; i < D; i++) {
            for(int j=0; j < D; j++) {
                if(i == j) {
                    clusters->R[m*D*D+i*D+j] = 1.0f;
                } else {
                    clusters->R[m*D*D+i*D+j] = 0.0f;
                }
            }
        }
        
        DEBUG("R:\n");
        for(int d=0; d < D; d++) {
            for(int e=0; e < D; e++) 
                DEBUG("%.2f ",clusters->R[m*D*D+d*D+e]);
            DEBUG("\n");
        }
        DEBUG("\n");

    }

    free(variances);
    free(means);
}

void constants(clusters_t* clusters, int M, int D) {
    float log_determinant;
    float* matrix = (float*) malloc(sizeof(float)*D*D);

    float sum = 0.0;
    for(int m=0; m < M; m++) {
        // Invert covariance matrix
        memcpy(matrix,&(clusters->R[m*D*D]),sizeof(float)*D*D);
        invert_cpu(matrix,D,&log_determinant);
        memcpy(&(clusters->Rinv[m*D*D]),matrix,sizeof(float)*D*D);
    
        // Compute constant
        clusters->constant[m] = -D*0.5*logf(2*PI) - 0.5*log_determinant;

        // Sum for calculating pi values
        sum += clusters->N[m];
    }

    // Compute pi values
    for(int m=0; m < M; m++) {
        clusters->pi[m] = clusters->N[m] / sum;
    }
    
    free(matrix);
}

void estep1(float* data, clusters_t* clusters, int D, int M, int N, float* likelihood) {
    // Compute likelihood for every data point in each cluster
    float like;
    float* means;
    float* Rinv;
    for(int m=0; m < M; m++) {
        means = (float*) &(clusters->means[m*D]);
        Rinv = (float*) &(clusters->Rinv[m*D*D]);
        
        for(int n=0; n < N; n++) {
            like = 0.0;
            #if DIAG_ONLY
                for(int i=0; i < D; i++) {
                    like += (data[i*N+n]-means[i])*(data[i*N+n]-means[i])*Rinv[i*D+i];
                }
            #else
                for(int i=0; i < D; i++) {
                    for(int j=0; j < D; j++) {
                        like += (data[i*N+n]-means[i])*(data[j*N+n]-means[j])*Rinv[i*D+j];
                    }
                }
            #endif  
            clusters->memberships[m*N+n] = -0.5f * like + clusters->constant[m] + logf(clusters->pi[m]); 
        }
    }
}

void estep2(float* data, clusters_t* clusters, int D, int M, int N, float* likelihood) {
    float max_likelihood, denominator_sum;
    *likelihood = 0.0f;

    for(int n=0; n < N; n++) {
        // initial condition, maximum is the membership in first cluster
        max_likelihood = clusters->memberships[n];
        // find maximum likelihood for this data point
        for(int m=1; m < M; m++) {
            max_likelihood = fmaxf(max_likelihood,clusters->memberships[m*N+n]);
        }

        // Computes sum of all likelihoods for this event
        denominator_sum = 0.0f;
        for(int m=0; m < M; m++) {
            denominator_sum += expf(clusters->memberships[m*N+n] - max_likelihood);
        }
        denominator_sum = max_likelihood + logf(denominator_sum);
        *likelihood = *likelihood + denominator_sum;

        // Divide by denominator to get each membership
        for(int m=0; m < M; m++) {
            clusters->memberships[m*N+n] = expf(clusters->memberships[m*N+n] - denominator_sum);
            //printf("Membership of event %d in cluster %d: %.3f\n",n,m,clusters->memberships[m*N+n]);
        }
    }
}

void mstep_n(float* data, clusters_t* clusters, int D, int M, int N) {
    DEBUG("mstep_n: D: %d, M: %d, N: %d\n",D,M,N);
    for(int m=0; m < M; m++) {
        clusters->N[m] = 0.0;
        // compute effective size of each cluster by adding up soft membership values
        for(int n=0; n < N; n++) {
            clusters->N[m] += clusters->memberships[m*N+n];
        }
    }
}

void mstep_mean(float* data, clusters_t* clusters, int D, int M, int N) {
    DEBUG("mstep_mean: D: %d, M: %d, N: %d\n",D,M,N);
    for(int m=0; m < M; m++) {
        for(int d=0; d < D; d++) {
            clusters->means[m*D+d] = 0.0;
            for(int n=0; n < N; n++) {
                clusters->means[m*D+d] += data[d*N+n]*clusters->memberships[m*N+n];
            }
            clusters->means[m*D+d] /= clusters->N[m];
        }
    }
}

void mstep_covar(float* data, clusters_t* clusters, int D, int M, int N) {
    DEBUG("mstep_covar: D: %d, M: %d, N: %d\n",D,M,N);
    float sum;
    float* means;
    for(int m=0; m < M; m++) {
        means = &(clusters->means[m*D]);
        for(int i=0; i < D; i++) {
            for(int j=0; j <= i; j++) {
                #if DIAG_ONLY
                    if(i != j) {
                        clusters->R[m*D*D+i*D+j] = 0.0f;
                        clusters->R[m*D*D+j*D+i] = 0.0f;
                        continue;
                    }
                #endif
                sum = 0.0;
                for(int n=0; n < N; n++) {
                    sum += (data[i*N+n]-means[i])*(data[j*N+n]-means[j])*clusters->memberships[m*N+n];
                }
                if(clusters->N[m] >= 1.0f) {
                    clusters->R[m*D*D+i*D+j] = sum / clusters->N[m];
                    clusters->R[m*D*D+j*D+i] = sum / clusters->N[m];
                } else {
                    clusters->R[m*D*D+i*D+j] = 0.0f;
                    clusters->R[m*D*D+j*D+i] = 0.0f;
                }
            }
        }
    }
}
