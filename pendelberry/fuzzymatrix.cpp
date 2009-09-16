/*
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
void  computeGeneralFormula(Params* p, double beta, double tau, const int THETA); //not really a void -- sets Ti (i) to 0 or returns the  Event


// REFERENCE COPY OF STRUCTURE:
//	struct Params {
//		int 		numEvents;			// Number of cells
//		int 		numDimensions;		// p in the paper.
//		int 		numClusters;		// Set by user.
//		int 		fuzziness;			// User params also.
//		int 		maxLikelihood;		// True or false.
//		double 		threshold;			// Epsilon == max acceptable change 
//		double 		newNorm;			// ???
//		double 		oldNorm;			// ???
//
//		double* 		data;				// NxP matrix of point coordinates.
//
//		double* 		centers;			// Nx<numClusters> coords of centers.
//
//		double 		membership[3][64];		// Px<numclusters> values that tell
//								// how connected each point is to each
//								// cluster center.
//
//		double		membership2[3][64];		// Copy of membership.
//		
//		int*		Ti;				// N data points that are set if a point has any
//								// negative membership.
//	};

// Given the clusters, find the centers, equation (30)
void setCenters(Params* p) {
    double numerator;
    double denom;
    
    // Compute the center for all clusters
    for(int t=0; t < p->numClusters; t++) {
        // Each center has 'd' dimensions, loop over all of them
        for(int d=0; d < p->numDimensions; d++) {
            numerator = 0.0;
            denom = 0.0;
            // Average over all events weighted by membership for this particular cluster
            for(int i=0; i < p->numEvents; i++) {
                double u_ti = p->membership[t*p->numEvents+i];
                double x_i = p->data[i*p->numDimensions+d];
                numerator += u_ti*u_ti*x_i;
                denom += u_ti*u_ti;
            }
            p->centers[t*p->numDimensions+d] = numerator / denom;            
        }
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

		// For each Event:
		for (int j = 0; j < p->numDimensions; j++)
		{
			double numerator = 0;
			for (int i = 0; i < p->numDimensions; i++)
			{
				numerator = 0;
				for (int event_id = 0; event_id < p->numEvents; event_id++)
				{
					numerator = numerator + (p->membership[t*p->numEvents+event_id]*p->membership[t*p->numClusters+event_id]) *
							  (p->data[i + event_id * p->numDimensions] - avgs[i]) *
							  (p->data[j + event_id * p->numDimensions] - avgs[j]);
				}

				// TODO: Move this outside the loop:
				double denominator = 0;
				for (int event_id = 0; event_id < p->numEvents; event_id++)
				{
					denominator = denominator + p->membership[t*p->numEvents+event_id];
				}

				p->scatters[t*p->numClusters*p->numDimensions*p->numDimensions+i*p->numDimensions+j] = numerator / denominator;
			}
		}
	}
}

/*******************************************************************************
 *
 *    Equation 31:
 *    Compute the membership-ness of each cluster.
 *  	sets: 	params->membership;
 *
 *******************************************************************************/
// TODO
void computeGeneralFormula_eq31(Params* p, double beta, double tau, double THETA)
{
    // B(ir), as per equation 27
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
    int inner_iter 	= 1;
    int outer_iter 	= 1;

    int need_to_continue = 0;

    // Step 1:
    	// Initialize the membership functions.
    	// Often it works to initialize things to 1/(number of clusters).
    	//memset( p->membership, 
    	//	1.0 / p->numClusters, 
    	//	p->numClusters * p->numEvents );
    double initial_membership = 1.0 / p->numClusters;
    for(int i=0; i < p->numEvents*p->numClusters; i++) {
        p->membership[i] = initial_membership;
    }

    // TOP OF THE LOOP
    // AS LONG AS IT TAKES...

	double max_epsilon_change = p->threshold; //Set a stopping criteria

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
	    // TODO: Do this...
	    setScatterMatrices( p ); 

	    // Step 3:
	    // TODO;
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

	    int any_Ti_set 			= 0;
	    int need_to_keep_reiterating	= 1;
	    while ( need_to_keep_reiterating )
	    {
    		any_Ti_set 			= 0;
    		need_to_keep_reiterating	= 0;

    		// Evaluate the membership functions according to EQ 31.
    		// If any membership is negative, range clip it to zero.
    		// 		AND set p->Ti( that point ) to 1.
    		//		AND re-calculate the memberships according to EQ 31.

    		// TODO: This is one set of the settings that are used for the equation:
    		//      EQ 31 that is here...

    	    // switch (p->options)                  /* select the type of calculation */
    		// pDimensions is the number of dimensions
            double beta,tau;
        
    		switch (p->option) 
    		{
    			case 1:
    				beta = 1.0/p->numDimensions;
    				tau  = 0.0;
    				break;
    			case 2:
    				beta =1.0;
    				tau =0.0;
    				break;
    			case 3:
    				beta =1.0;
    				tau =0.0;
    				break;
    			case 4:
    				beta = 999.0;
    				tau = 0.0;
    				break;
    			// catch all for bad option
    			default: cout << "Invalid option selected" << endl;
                exit(1);
    		}

    		// SWITCH BASED ON WHICH ALGORITHM TO USE:
		
    		// kClusters is the number of clusters
    		// sHat is the Scatter Matrix
    		// aSubT
    		// beta
    		// bSubIT
    		// tau
    		// THETA is a constant of 1 -- it is really only used in adaptive distance and is ignored in the other methods
    		// THIS IS EQUATION (31)....
    		// TODO: AAA
		
    		computeGeneralFormula_eq31(p, beta, tau, THETA);

    		// Total up all values of "any_Ti_set".
    		// If the sum is non-zero, then one of them is set,
    		// and you need to keep looping.
    		for (int ii=0; ii < p->numEvents; ii++ )
    		{
    		    any_Ti_set = any_Ti_set + p->Ti[ii];
    		}
    		inner_iter++;
		
            cout << "inner_iter: " << inner_iter << endl;

	    } while (any_Ti_set != 0 && inner_iter <= MAXITER);

        need_to_keep_reiterating	= 0;
        if (inner_iter == MAXITER)
	    {
		    cout << "Program was unable to converge using the threshold " << p->threshold << endl;
    		break;
	    }

	// Step 4:
	// COMPARE THE MEMBERSHIP FUNCTIONS TO PREVIOUS METHODS.
		    max_epsilon_change = 0;

	for (int idx=0;idx<=p->numClusters * p->numEvents; idx++)
	{
	    double difference = fabs(p->membership[idx] - p->membership2[idx]);
	    
	    if (difference > max_epsilon_change)
		    max_epsilon_change = difference;
	}

	outer_iter	 = outer_iter + 1;
        if ( (max_epsilon_change >= p->threshold) && (outer_iter <= MAXITER) )
	{
	    need_to_continue = 1;
	}
    }

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



/******************************************************************************* 
 *
 *     MAIN
 *
 *******************************************************************************/


int main(int argc, char** argv)
{	
	
	if (argc != 5) {
		usage();
		return EXIT_FAILURE;
	}

	initRand();

	Params* params = (Params*) malloc(sizeof(Params));
	
	//Initialize Membership values of all objects with respect to each cluster
	params->numClusters 	= atoi(argv[1]);	// K in the paper.
	//params->fuzziness 	= atoi(argv[2]);	// Fuzziness = theta, always 1.0.
	params->threshold 	= atof(argv[2]);	// When to stop... when epsilon < threshold.
	params->option 	= atoi(argv[3]);	// Which algorithm to use, 1 of 4.

    cout << "Number of clusters: " << params->numClusters << endl;
    cout << "Threshold: " << params->threshold << endl;
    cout << "Option: " << params->option << endl << endl;
    
    cout << "Reading in data...";
	int error = readData(argv[4], params);
	if(error) {
        exit(error);
	}
    cout << "done" << endl << endl;

	// Initializes and allocates arrays in the struct
    allocateParamArrays(params);
    
    cout << "Starting Scatter matrices" << endl;
	fuzzySmatrix( params ) ;
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
}

/* TBD: Equation 31 */
int computeGeneralFomula(Params* p) {
	double numerator;
	double denominator = 0;
	double exp = 1;
	double base;
	double temp;

	p->newNorm = 0;

//	if ((p->fuzziness - 1) > 1) {
//		exp = 1 / (p->fuzziness  - 1);
//	}

	for (int i = 0; i < p->numEvents; i++) {
		for (int j = 0; j < p->numClusters; j++) {
			base = computeDistance(p, i, j);
			numerator = pow(base, exp);

			for (int x = 0; x < p->numClusters; x++) {
				base = computeDistance(p, i, x);
				denominator += pow(base, exp);
			}

			temp = numerator / denominator;

			p->membership[j + i * p->numClusters] = temp;

			if (p->newNorm < temp || (p->newNorm == 0 && p->oldNorm == 0)) {
				p->newNorm = temp;
			}

			denominator = 0;
		}
	}

	double diff = fabs(p->newNorm - p->oldNorm);

	if (diff < p->threshold && p->oldNorm != 0) {
		return 1;
	}
	else {
		p->oldNorm = p->newNorm;
		return 0;
	}
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
	cout << "Usage: ./scattermatrix <num clusters> <threshold> <option> <file name>" << endl;
    cout << "\t" << "option:" << endl;
    cout << "\t\t" << "1: Adaptive distances" << endl;
    cout << "\t\t" << "2: Minimum total volume" << endl;
    cout << "\t\t" << "3: SAND" << endl;
    cout << "\t\t" << "4: Maximum likelihood" << endl;
}
