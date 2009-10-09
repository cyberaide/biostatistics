/**
 * Implement Fuzzy clustering using scatter matrices paper P.J. Rousseeuw, L. Kaufman, E. Trauwaert 1996...
 * 
 */

#include "clusteringutils.h"
#include "fuzzymatrix.h"
#include <math.h>

const int THETA = 1; // Maybe

void  usage();
void  fcSmatrix(Params* p);
void  computeCenters(Params* p);
double computeDistance(Params* p, int i, int c);
void  computeMaxLikelihood(Params* p); //not really a void -- sets Ti (i) to 0 or returns the Event
void  computeGeneralFormula(Params* p); //not really a void -- sets Ti (i) to 0 or returns the  Event
void compute_A_general(Params* p);
void compute_A_maxlikelihood(Params* p);
void printSquareMatrix(double* matrix, int n); 
void fuzzySmatrix(Params* p);

/******************************************************************************* 
 *
 *     MAIN
 *
 *******************************************************************************/


int main(int argc, char** argv)
{	
	
	if (argc != 7) {
		usage();
		return EXIT_FAILURE;
	}

	initRand();

	Params* params = (Params*) malloc(sizeof(Params));
	
	//Initialize Membership values of all objects with respect to each cluster
	params->numClusters = atoi(argv[1]);	// K in the paper.
	params->threshold = atof(argv[2]);		// When to stop... when epsilon < threshold.
	params->option = atoi(argv[3]);			// Which algorithm to use, 1 of 4.
	params->setup = atoi(argv[4]);			//how is the scatter matrix initialized, 1 to 3
	params->innerloop = atoi(argv[5]);		//how many time to loop through Ti

    cout << "Number of clusters: " << params->numClusters << endl;
    cout << "Threshold: " << params->threshold << endl;
    cout << "Option: " << params->option << endl << endl;
	cout << "Setup: " << params->setup <<endl << endl;
	cout << "Innerloop: " << params->innerloop <<endl <<endl;
    
    cout << "Reading in data...";
	int error = readData(argv[6], params);
	if(error) {
        cout << "error" << endl;
        exit(error);
	}
    cout << "done" << endl << endl;

    cout << "Number of events: " << params->numEvents << endl;
    cout << "Number of dimensions: " << params->numDimensions << endl;

	// Initializes and allocates arrays in the struct
    allocateParamArrays(params);
    
	switch (params->option) 
	{
		case 1:
			params->beta = 1.0/params->numDimensions;
			params->tau  = 0.0;
			break;
		case 2:
			params->beta = 0.5;
			params->tau = params->numDimensions / 2.0;
			break;
		case 3:
			params->beta = 1.0/params->numDimensions;
			params->tau = 1.0;
			break;
		case 4:
		    params->beta = 999.0;
			params->tau = 999.0;
            break;
		// catch all for bad option
		default: cout << "Invalid option selected" << endl;
        exit(1);
	}
	    
    cout << "Starting Scatter matrices" << endl;
	fuzzySmatrix( params ) ;
}

// REFERENCE COPY OF STRUCTURE:
//struct Params {
//    int         numEvents;			// Number of rows
//    int         numDimensions;		// p in the paper.
//    int         numClusters;			// Set by user.
//    int         option;				// Specify which algorithm
//	  int			setup;				// How the matrix is initialized
//	  int			innerloop;			// How many passes through the inner loop for Ti -- stopping condition
//    double      threshold;			// Epsilon == max acceptable change 
//    double*     data;					// NxP matrix of point coordinates
//    double*     centers;				// Nx<numClusters> coords of centers.
//    double*     membership;			// Px<numclusters> values that tell
										// how connected each point is to each
										// cluster center.
//    double*     membership2;			// Copy of membership.
//    double*     scatters;				// Scatter matrices. One N*N matrix per cluster, dynamically allocated
//    double*     scatter_inverses;		// Inver of all scatter matrices
//    int*        Ti;					// N data points that are set if a point has any
										// negative membership.
//    double*     n;					// effective size of each cluster (sum of fuzzy memberships for that cluster)
//    double*     A_t;					// Used for eqn (31)
    
//    double*      determinants;		// determinants of scatter matrices 
    
//    double      beta;
//    double      tau;
//};

// Given the clusters, find the centers, equation (30)
void setCenters(Params* p) {
    double numerator;
    double denominator;
    
    // Compute the center for all clusters
    for(int t=0; t < p->numClusters; t++) {
        // Each center has 'd' dimensions, loop over all of them
        for(int d=0; d < p->numDimensions; d++) {
            numerator = 0.0;
            denominator = 0.0;
            // Average over all events weighted by membership for this particular cluster
            for(int i=0; i < p->numEvents; i++) {
                double u_ti = p->membership[i*p->numClusters+t];
                double x_i = p->data[i*p->numDimensions+d];
                numerator += u_ti*u_ti*x_i;
                denominator += u_ti*u_ti;
            }
            p->centers[t*p->numDimensions+d] = numerator / denominator;            
        }
    }
}

void seedCenters(Params* p) {
        double *temp=new double[p->numDimensions];

        for (int i = 0; i < p->numClusters; i++) {
                getPoints(p, temp, randIntRange(0, p->numEvents));

                if (i != 0) {
                        while (contains(p, temp)) {
                                getPoints(p, temp, randIntRange(0, p->numEvents));
                        }
                }

                for (int j = 0; j < p->numDimensions; j++) {
                        p->centers[j + i * p->numDimensions] = temp[j];
                }
        }
        delete[]temp;
}

void printSquareMatrix(double* matrix, int n) {
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            printf("%0.2f ",matrix[i*n+j]);
        }
        printf("\n");
    }
}
// Computes scatter matrices (covariance matrices) for all clusters according to (17)
void setScatterMatrices(Params* p)
{
    double  denominator = 0;
    double *numerator=new double[p->numDimensions];
    int    membIndex;

    double* total = new double[p->numDimensions];
    double* avgs = new double[p->numDimensions];

    
	// Solve for each cluster at a time:
	for (int t = 0; t < p->numClusters; t++)
	{
		// For each Event:
		for (int event_id = 0; event_id < p->numEvents; event_id++)
		{
			for (int dim_id = 0; dim_id < p->numDimensions; dim_id++)
			{
				total[dim_id] = total[dim_id] + p->data[dim_id + event_id * p->numDimensions];
			}
		}

		// Normalize the totals:
		for (int dim_id = 0; dim_id < p->numDimensions; dim_id++)
		{
			avgs[dim_id] = total[dim_id] / p->numEvents;
		}

        double numerator;
        double denominator = 0.0;
        for (int event_id = 0; event_id < p->numEvents; event_id++)
        {
            //denominator += p->membership[t*p->numEvents+event_id];
            denominator += p->membership[event_id*p->numClusters+t];
        }
        
        // denominator is n_t
        p->n[t] = denominator;
		
        // Covariance matrix
		for (int i = 0; i < p->numDimensions; i++)
		{
			for (int j = 0; j < p->numDimensions; j++)
			{
				numerator = 0.0;
				for (int event_id = 0; event_id < p->numEvents; event_id++)
				{
					numerator = numerator + (p->membership[event_id*p->numClusters+t]*p->membership[event_id*p->numClusters+t]) *
							  //(p->data[i + event_id * p->numDimensions] - avgs[i]) *
							  (p->data[i + event_id * p->numDimensions] - p->centers[t*p->numDimensions+i]) *
							  //(p->data[j + event_id * p->numDimensions] - avgs[j]);
							  (p->data[j + event_id * p->numDimensions] - p->centers[t*p->numDimensions+j]);
				}

                if(fabs(numerator) < 1e-20) {
                       numerator = 1e-20;
                }
				p->scatters[t*p->numDimensions*p->numDimensions+i*p->numDimensions+j] = numerator / denominator;
			}
		}
	}
	
	// compute scatter_inverses and determinants
	for(int i=0; i<p->numClusters; i++) {
        double* scatter = &(p->scatters[i*p->numDimensions*p->numDimensions]);
        double* scatter_inverse = &(p->scatter_inverses[i*p->numDimensions*p->numDimensions]);
        memcpy(scatter_inverse,scatter,p->numDimensions*p->numDimensions*sizeof(double));
        double det = 0.0;
        invert_cpu(scatter_inverse,p->numDimensions,&det);
        p->determinants[i] = det;

    }
		
	// compute A_t
	if(p->option == 4) {
        compute_A_maxlikelihood(p);
    } else {
        compute_A_general(p);
    }
        
    // clean memory
    delete[]numerator;
    delete[]total;
    delete[]avgs;
}

double dotProduct(double* a, double* b, int n) {
    double dp = 0.0;
    for(int i=0; i<n; i++) {
        dp += a[i]*b[i];
    }
    return dp;
}

double compute_B_general(Params* p, int i, int t) {
    double n_t = p->n[t];
    double det_S_t = p->determinants[t];
    
    // All the scalar stuff on left size of matrix multiplication
    double constant = pow(n_t,p->numDimensions*p->beta - p->tau-1.0)*pow(THETA*det_S_t,p->beta);
    //cout << "constant: " << constant << endl;

    // Temp vector for (x_i - u_t)
    double* difference = new double[p->numDimensions];
    for(int j=0; j< p->numDimensions; j++) {
        difference[j] = p->data[i*p->numDimensions+j] - p->centers[t*p->numDimensions+j];
    }
    
    // Temp matrix for matrix mult of (x_i - u_t)*(S_t_inv)
    double* temp = new double[p->numDimensions];
    
    for(int j=0; j < p->numDimensions; j++) {
        // NOTE: Not sure about the apersand..might need to just do point arithmetic to get the double* instead of double
        temp[j] = dotProduct(difference,&(p->scatter_inverses[t*p->numDimensions*p->numDimensions+j*p->numDimensions]),p->numDimensions);
    }
    
    double mult_result = dotProduct(temp,difference,p->numDimensions);

    // cleanup   
    delete[]temp;
    delete[]difference;
 
    return constant*mult_result;
}

double compute_B_maxlikelihood(Params* p, int i, int t) {
	
    
    // Temp vector for (x_i - u_t)
    double* difference = new double[p->numDimensions];
    for(int j=0; j< p->numDimensions; j++) {
        difference[j] = p->data[i*p->numDimensions+j] - p->centers[t*p->numDimensions+j];
    }
    
    // Temp matrix for matrix mult of (x_i - u_t)*(S_t_inv)
    double* temp = new double[p->numDimensions];
    
    for(int j=0; j < p->numDimensions; j++) {
        temp[j] = dotProduct(difference,&(p->scatter_inverses[t*p->numDimensions*p->numDimensions+j*p->numDimensions]),p->numDimensions);
    }
    
    double b_it = dotProduct(temp,difference,p->numDimensions);

    // cleanup   
    delete[]temp;
    delete[]difference;
 
    return b_it;
}

void compute_A_general(Params* p) {
    for(int t=0; t < p->numClusters; t++) {
        double constant = pow(p->n[t],p->numDimensions*p->beta - p->tau-1.0)*pow(THETA*p->determinants[t],p->beta);
        p->A_t[t] = 0.5*p->tau/p->beta*constant;
    }
}

void compute_A_maxlikelihood(Params* p) {
    for(int t=0; t < p->numClusters; t++) {
        p->A_t[t] = -0.5 * log(fabs(p->determinants[t]));
    }  
}


/*******************************************************************************
 *
 *    Equation 31:
 *    Compute the membership-ness of each cluster.
 *  	sets: 	params->membership;
 *
 *******************************************************************************/
void computeGeneralFormula_eq31(Params* p)
{
    double membership = 0.0;
	
    for(int i=0; i<p->numEvents; i++) {
        double sum_memberships = 0.0;
        for(int t=0; t< p->numClusters; t++) {
			double B_it = 0.0;
			if (p->option == 4) {
                B_it = compute_B_maxlikelihood(p,i,t) + 0.001;
            } else {
                B_it = compute_B_general(p,i,t) + 0.001;
            }

            double B_ir = 0.0;
            double sum_B_ir = 0.0;
            double sum_A_ir_and_B_ir = 0.0;
            for(int r=0; r< p->numClusters; r++) {
                // Only include clusters not in the set Ti
                if(p->Ti[i*p->numClusters+r] == 0) {
					if (p->option == 4) {
                        B_ir = compute_B_maxlikelihood(p,i,r) + 0.001;
                    } else {
                        B_ir = compute_B_general(p,i,r) + 0.001;
                    }
                    //cout << "B_" << i << "_" << r << " = " << B_ir << endl;
                    sum_B_ir += 1.0/B_ir;
                    sum_A_ir_and_B_ir += p->A_t[r] / B_ir;
                }
            }
            membership = (1.0/B_it)/sum_B_ir - (1.0/B_it)*(sum_A_ir_and_B_ir/sum_B_ir - p->A_t[t]);
            if(membership < 0.0) {
            	p->Ti[i*p->numClusters+t] = 1;
            	membership = 0.0;
            }
            if( isnan(membership) ) {
                cout << "Membership #" << i << " is NaN" << endl;
                return;
            }
            p->membership[i*p->numClusters+t] = membership;
            sum_memberships += membership;
        }
        // normalize memberships for this event
        for(int t=0; t < p->numClusters; t++) {
            p->membership[i*p->numClusters+t] /= sum_memberships;
        }
    }
}

// Computes covariance of the data set, sets every scatter matrix to initial covariance
// Or should it just set them to identity (initialize to round clusters...)?
void seedScatters(Params* p) {
    // Compute Covariance
    //  identity matrix....
    for(int i=0; i<p->numDimensions;i++) {
        for(int j=0; j<p->numDimensions;j++) {
            if(i==j) {
                p->scatters[i*p->numDimensions+j] = 1.0;
            } else {
                p->scatters[i*p->numDimensions+j] = 0.0;
            }
        }
    }

    // Set every scatter to covariance
    int size = sizeof(double)*p->numDimensions*p->numDimensions;
    for(int i=1; i < p->numClusters; i++) {
        // Compute starting location of
        double* scatter = &(p->scatters[i*p->numDimensions*p->numDimensions]);
        memcpy(scatter,p->scatters,size);
    }

	// compute scatter_inverses and determinants
	for(int i=0; i<p->numClusters; i++) {
        double* scatter = &(p->scatters[i*p->numDimensions*p->numDimensions]);
        double* scatter_inverse = &(p->scatter_inverses[i*p->numDimensions*p->numDimensions]);
        memcpy(scatter_inverse,scatter,size);
        double det = 0.0;
        invert_cpu(scatter_inverse,p->numDimensions,&det);
        p->determinants[i] = det;
    }
}

/*********************
 *
 *     fuzzySmatrix
 *
 *  The main loop to impliment the algorithm.
 *
 *********************/


void fuzzySmatrix(Params* p) 
{
    int det 	= 0;
    int inner_iter 	= 0;
    int outer_iter 	= 0;
    int need_to_continue = 0;
	
	

    // Step 1:
    	// Initialize the membership functions.
    	
	switch (p->setup){
		case 1:													// Random Centers
            // Initialize the centers to random points
            seedCenters(p);
            // Initialize the scatters to identity matrices
            seedScatters(p);
            // Need to define N, divide points evenly for start
            for(int i=0; i < p->numClusters; i++) {
                p->n[i] = p->numEvents / p->numClusters;
            }
            // Compute the A_t values (needed for membership calculations)
            if(p->option == 4) {
                compute_A_maxlikelihood(p);
            } else {
                compute_A_general(p);
            }
            
            printCenters(p); 
            printScatters(p);

            // compute initial memberships based on the centers/scatters
            memset(p->Ti,0,sizeof(int)*p->numEvents*p->numClusters);
    		computeGeneralFormula_eq31(p);
			break;
		case 2:					
            cout << "Setup option #2 not implemented." << endl;
            exit(1);
            // Not implemented yet
            // Would run K-means for X iterations to determine initial cluster membership
			int* kmeans_result;
			for(int i=0; i< p->numEvents; i++) {
				for(int t=0; t < p->numClusters; t++) {
					if((t+1) == kmeans_result[i]) {
						p->membership[i*p->numClusters+t] = 1.0;
					} else {
						p->membership[i*p->numClusters+t] = 0.0;
					}
				}
			}
			break;
		case 3:													// Assign each point fully to a cluster
			for(int i=0; i< p->numEvents; i++) {
				int member = rand() % (p->numClusters);
				for(int t=0; t < p->numClusters; t++) {
					if(t == member) {
						p->membership[i*p->numClusters+t] = 1.0;   
					} else {
						p->membership[i*p->numClusters+t] = 0.0;   
					}
				}
			}
			setCenters( p );
			break;
		default: cout << "Invalid option selected" << endl;
        exit(1);
    }
    // TOP OF THE LOOP
    // AS LONG AS IT TAKES...

	double max_epsilon_change = p->threshold; //Set a stopping criteria

    int* Ti_copy = new int[p->numEvents*p->numClusters];

	// MAIN STEPS:
	// 1:
	need_to_continue = 1;
	outer_iter	 = 1;
	while ( need_to_continue == 1 )
	{
	    need_to_continue = 0;

	    // Make a copy of all the membership matrices:
	    memcpy( p->membership2, p->membership, 
		    sizeof(double) * p->numClusters * p->numEvents );

	    // Step 2a:
	    // Estimate the cluster centers:
	    // Changes the structure...
	    setCenters( p );
	    
        printCenters( p ); // for debugging

	    // Step 2b:
	    // Compute Scatter matrix according to equation 17 (covariance matrix):
	    setScatterMatrices( p ); 
        printScatters( p );

        for(int i=0; i< p->numClusters; i++) {
            cout << "N: " << p->n[i] << endl;
        }

	    // Step 3:
	    //  "Initialise the set Ti == 0,
	    //  and evalute the membership functions according to (31).
	    //  If some of the memberships are <0, clip them to zero,
	    //  add their index to the set T(i) (bad boys) and
	    //  recalculate the other memberships according to (31).
	    //  Iterate as long as there are any negative memberships."

	    // Initialize the set Ti = none.
	    // This is the set of points to ignore because they had a membership
	    // value that was negative.
	    // Compute the Ti's, and iterate as long as any Ti != 0.
	    memset( p->Ti, (int) 0, p->numEvents );

	    int any_Ti_set;
	    // Clear all the Ti values
        memset(p->Ti,0,sizeof(int)*p->numEvents*p->numClusters);
	    
	    inner_iter = 0;
	    do
	    {
    		any_Ti_set = 0;
            memcpy(Ti_copy,p->Ti,sizeof(int)*p->numClusters*p->numEvents);

    		// Evaluate the membership functions according to EQ 31.
    		// If any membership is negative, range clip it to zero.
    		// 		AND set p->Ti( that point ) to 1.
    		//		AND re-calculate the memberships according to EQ 31.
    		// THETA is a constant of 1 -- it is really only used in adaptive distance and is ignored in the other methods
    		// THIS IS EQUATION (31)....
		
    		computeGeneralFormula_eq31(p);

    		// Total up all values of "any_Ti_set".
    		// If the sum is non-zero, then one of them is set,
    		// and you need to keep looping.
    		for (int ii=0; ii < p->numEvents*p->numClusters; ii++ ){
    		    if(p->Ti[ii] != 0) {
    		        any_Ti_set = 1;
    		        break;
    		    }
    		}
    		inner_iter++;
            //if(memcmp(Ti_copy,p->Ti,sizeof(int)*p->numClusters*p->numEvents) == 0) {
            //    cout << "No change in Ti after " << inner_iter << " iterations. Terminating inner loop" << endl;
            //    break;
            //}
	    } while (any_Ti_set != 0 && inner_iter < p->innerloop);

        cout << "Inner Iter: " << inner_iter << endl;

        if (inner_iter == p->innerloop){
		    cout << "Did not resolve all Ti values in " << p->innerloop << " iterations" << endl;
    		//break; // Do we need to break? Or just issue a warning?
	    }

	    // Step 4:
	    // COMPARE THE MEMBERSHIP FUNCTIONS TO PREVIOUS METHODS.
        double difference;
        need_to_continue = 0; // Assume we don't need to continue, check differences until we find otherwise
	    for (int idx=0;idx < p->numClusters * p->numEvents; idx++){
	        difference = fabs(p->membership[idx] - p->membership2[idx]);
	        
	        if (difference > p->threshold) {
	            need_to_continue = 1;
	        }
	    }

	    outer_iter	 = outer_iter + 1;
    } // end of outer while loop

    cout << "Outer Iterations: " << outer_iter << endl << endl;

    cout << "Cluster centers:" << endl << endl;

    for (int i = 0; i < p->numClusters; i++) {
	    for (int j = 0; j < p->numDimensions; j++) {
		    cout << p->centers[j + i * p->numDimensions] << " ";
		}
	    cout << endl;
	}			    

	writeData(p, "output.dat");

	free(p->centers);
	free(p->data);
	free(p->membership);
	free(p);

}

void computeCenters(Params* p) {
	double denominator = 0;
	double *numerator=new double[p->numDimensions];
	int membIndex;

	for (int i = 0; i < p->numClusters; i++) {
		for (int x = 0; x < p->numDimensions; x++) {
			numerator[x] = 0;
		}

		for (int j = 0; j < p->numEvents; j++) {
			membIndex = i + j * p->numClusters;

			for (int x = 0; x < p->numDimensions; x++) {
				numerator[x] += p->data[x + j * p->numDimensions] * p->membership[membIndex];
			}

			denominator += p->membership[membIndex];
		}

		for (int x = 0; x < p->numDimensions; x++) {
			p->centers[x + i * p->numDimensions] = numerator[x] / denominator;
		}

		denominator = 0;
	}

    delete[]numerator;
}

double computeDistance(Params* p, int i, int c) {
	double sum = 0;
	double temp;

	for (int j = 0; j < p->numDimensions; j++) {
		temp = p->data[j + i * p->numDimensions] - p->centers[j + c * p->numDimensions];
		sum += pow(temp, 2);
	}

	return sqrt(sum);
}

void computeMaxLikelihood(Params* p) {
}

void usage() {
	cout << "Usage: ./scattermatrix <num clusters> <threshold> <option> <setup> <innerloop> <file name>" << endl;
    cout << "\t" << "option:" << endl;
    cout << "\t\t" << "1: Adaptive distances" << endl;
    cout << "\t\t" << "2: Minimum total volume" << endl;
    cout << "\t\t" << "3: SAND" << endl;
    cout << "\t\t" << "4: Maximum likelihood" << endl;
	cout << "\t" << "setup:" << endl;
    cout << "\t\t" << "1: Random Centers" << endl;
    cout << "\t\t" << "2: K-Means" << endl;
    cout << "\t\t" << "3: Random Memberships" << endl;
	cout << "\t" << "innerloop:" << endl;
    cout << "\t\t" << "0 for unlimited otherwise enter a number" << endl;
}
