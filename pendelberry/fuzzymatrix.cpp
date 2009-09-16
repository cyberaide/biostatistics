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
float computeDistance(Params* p, int i, int c);
void  computeMaxLikelihood(Params* p); //not really a void -- sets Ti (i) to 0 or returns the Event
void  computeGeneralFormula(Params* p, float beta, float tau, const int THETA); //not really a void -- sets Ti (i) to 0 or returns the  Event


// REFERENCE COPY OF STRUCTURE:
//	struct Params {
//		int 		numEvents;			// Number of cells
//		int 		numDimensions;		// p in the paper.
//		int 		numClusters;		// Set by user.
//		int 		fuzziness;			// User params also.
//		int 		maxLikelihood;		// True or false.
//		float 		threshold;			// Epsilon == max acceptable change 
//		float 		newNorm;			// ???
//		float 		oldNorm;			// ???
//
//		float* 		data;				// NxP matrix of point coordinates.
//
//		float* 		centers;			// Nx<numClusters> coords of centers.
//
//		float 		membership[3][64];		// Px<numclusters> values that tell
//								// how connected each point is to each
//								// cluster center.
//
//		float		membership2[3][64];		// Copy of membership.
//		
//		int*		Ti;				// N data points that are set if a point has any
//								// negative membership.
//	};

// Given the clusters, find the center
void setCenters(Params* p) {

}

// Computes scatter matrices (covariance matrices) for all clusters according to (17)
void setScatterMatrices(Params* p)
{
    float  denominator = 0;
    float *numerator=new float[p->numDimensions];
    int    membIndex;

    float* total = new float[p->numDimensions];
    float* avgs = new float[p->numDimensions];

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
			float numerator = 0;
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
				float denominator = 0;
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
void computeGeneralFormula_eq31(Params* p, float beta, float tau, float THETA)
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
int MAXITER 	= 150;

int need_to_continue = 0;

// Step 1:
	// Initialize the membership functions.
	// Often it works to initialize things to 1/(number of clusters).
	memset( p->membership, 
		(float) 1.0 / p->numClusters, 
		p->numClusters * p->numEvents );

// TOP OF THE LOOP
// AS LONG AS IT TAKES...

	float max_epsilon_change = 0.1; //Set a stopping criteria

	// MAIN STEPS:
	// 1:
	need_to_continue = 1;
	outer_iter	 = 1;
	while ( need_to_continue == 1 )
	{
	    need_to_continue = 0;

	    // Make a copy of all the membership matrices:
	    memcpy( p->membership2, p->membership, 
		    sizeof(float) * p->numClusters * p->numEvents );

	    // Step 2a:
	    // Estimate the cluster centers:
	    // Changes the structure...
	    setCenters( p );

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
        float beta,tau;
        
		switch (p->option) 
		{
			case '1':
				beta = 1.0/p->numDimensions;
				tau  = 0.0;
				break;
			case '2':
				beta =1.0;
				tau =0.0;
				break;
			case '3':
				beta =1.0;
				tau =0.0;
				break;
			case '4':
				beta = 999.0;
				tau = 0.0;
				break;
			// catch all for bad option
			default: cout << "Invalid option selected" << endl;
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
	    float difference = fabs(p->membership[idx] - p->membership2[idx]);
	    
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
	
	if (argc != 6) {
		usage();
		return EXIT_FAILURE;
	}

	initRand();

	Params* params = (Params*) malloc(sizeof(Params));
	
	//Initialize Membership values of all objects with respect to each cluster
	params->numClusters 	= atoi(argv[1]);	// K in the paper.
	params->fuzziness 	= atoi(argv[2]);	// Fuzziness = theta, always 1.0.
	params->threshold 	= atof(argv[3]);	// When to stop... when epsilon < threshold.
	params->option 	= atoi(argv[4]);	// Which algorithm to use, 1 of 4.

	readData(argv[5], params);

	// Initializes and allocates arrays in the struct
    allocateParamArrays(params);

	// Allocate one center per cluster.
	params->centers 	= (float*) 	malloc(sizeof(float) * params->numClusters * params->numDimensions);

	// U(it) in the paper. 
	params->membership 	= (float*) 	malloc(sizeof(float) * params->numClusters * params->numEvents);

	// Copy of last U(it).
	params->membership2 	= (float*) 	malloc(sizeof(float) * params->numClusters * params->numEvents);

	// Flag to see if membership goes negative.
	// T(i) in paper.
	params->Ti		= (int*)	malloc( sizeof(int) * params->numEvents );
	
	fuzzySmatrix( params ) ;
	
}



void computeCenters(Params* p) {
	float denominator = 0;
	float *numerator=new float[p->numDimensions];
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
	float numerator;
	float denominator = 0;
	float exp = 1;
	float base;
	float temp;

	p->newNorm = 0;

	if ((p->fuzziness - 1) > 1) {
		exp = 1 / (p->fuzziness  - 1);
	}

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

	float diff = fabs(p->newNorm - p->oldNorm);

	if (diff < p->threshold && p->oldNorm != 0) {
		return 1;
	}
	else {
		p->oldNorm = p->newNorm;
		return 0;
	}
}

float computeDistance(Params* p, int i, int c) {
	float sum = 0;
	float temp;

	for (int j = 0; j < p->numDimensions; j++) {
		temp = p->data[j + i * p->numDimensions] - p->centers[j + c * p->numDimensions];
		sum += pow(temp, 2);
	}

	return sqrt(sum);
}

void computeMaxLikelihood(Params* p) {
}

void usage() {
	cout << "Usage: ./fcmeans <num clusters> <fuzziness> <threshold> <option> <file name>" << endl;
}
