/*
 * Implement Fuzzy clustering using scatter matrices paper P.J. Rousseeuw, L. Kaufman, E. Trauwaert 1996...
 * 
 */
#include "clusteringutils.h"

void  usage();
void  fcSmatrix(Params* p);
void  computeCenters(Params* p);
float computeDistance(Params* p, int i, int c);
void  computeMaxLikelihood(Params* p); //not really a void -- sets Ti (i) to 0 or returns the Event
void  computeGeneralFormula(Params* p, float beta, float tau, const THETA); //not really a void -- sets Ti (i) to 0 or returns the  Event

int main(int argc, char** argv)
{	
	
	if (argc != 6) {
		usage();
		return EXIT_FAILURE;
	}

	initRand();

	Params* params = (Params*) malloc(sizeof(Params));
	
	//Initialize Membership values of all objects with respect to each cluster
	params->numClusters = atoi(argv[1]);
	params->fuzziness = atoi(argv[2]);
	params->threshold = atof(argv[3]);
	params->options = atoi(argv[4]);

	readData(argv[5], params);

	params->centers 	= (float*) 	malloc(sizeof(float) * params->numClusters * params->numDimensions);
	params->membership 	= (float*) 	malloc(sizeof(float) * params->numClusters * params->numEvents);
	params->membership2 = (float*) 	malloc(sizeof(float) * params->numClusters * params->numEvents);
	params->Ti			= (int*)	malloc( sizeof(int) * params->numEvents );
	

void fuzzySmatrix(Params* p) {
	int det = 0;
	int iter = 1;
	int MAXITER = 150;

	// Step 1:
		// Initialize the membership functions.
		// Often it works to initialize things to 1/(number of clusters).
		memset( params->membership, (float) 1.0 / params->numClusters, params->numClusters * params->numEvents );
		
	// TOP OF THE LOOP
	// AS LONG AS IT TAKES...
		
		float max_epsilon_change = 0.1; //Set a stopping criteria

				switch (params->options)                  /* select the type of calculation */
				// pDimensions is the number of dimensions
				// kClusters is the number of clusters
				// sHat is the Scatter Matrix
				// aSubT
				// beta
				// bSubIT
				// tau
				// THETA is a constant of 1 -- it is really only used in adaptive distance and is ignored in the other methods
				    {
					//adaptive distances
				    case 1:
				    	do {
				    				// Make a copy of all the membership functions.
				    				memcpy( params->membership2, params->membership, sizeof(float) * params->numClusters * params->numEvents );


				    				// Step 2a:
				    				// Estimate the cluster centers:
				    				setCenters( params );

				    				// Step 2b:
				    				// Set Scatter matrix according to equation 17 (covariance matrix):
				    				setScatterMatrices( params );

				    				// Step 3:
				    				// Initialize the set Ti = none.
				    				// This is the set of points to ignore because they had a membership
				    				// value that was negative.
				    				// Compute the Ti's, and iterate as long as any Ti != 0.
				    				memset( params->Ti, (int) 0, params->numEvents );

				    				int any_Ti_set = 0;
				    				do {
				    					any_Ti_set = 0;

				    					// Evaluate the membership functions according to EQ 31.
				    					// If any membership is negative, range clip it to zero.
				    					// 		AND set params->Ti( that point ) to 1.
				    					//		AND re-calculate the memberships according to EQ 31.
				    					//      EQ 31 that is here...
				    					beta = 1/pDimensions;
				    					tau = 0;
				    					computeGeneralFormula(p, beta, tau, THETA);
				    					// Total up all values of "any_Ti_set".
				    					// If the sum is non-zero, then one of them is set,
				    					// and you need to keep looping.
				    					for (int ii=0; ii < params->numEvents; ii++ )
				    						any_Ti_set = any_Ti_set + params->Ti(ii);
				    					iter++;

				                    } while (any_Ti_set != 0 && iter <= MAXITER);
				    				if (iter == MAXITER) {
				    						cout << "Program was unable to converge using the threshold " << p->threshold << endl;
				    						break;
				    					}

				                    // Step 4:
				                    // Compare the membership functions to previous methods.
				    				max_epsilon_change = 0;
				                    for (int idx=0;idx<=params->numClusters * params->numEvents; idx++)
				                    	{
				                    	float difference = params->membership(idx) - params->membership2(idx);
				                    	if (difference < 0)
				                    		difference = -difference;		// Form absolute value of difference.
				                    	if (difference > max_epsilon_change)
				                    		max_epsilon_change = difference;
				                    	}
				        } while (max_epsilon_change >= params->threshold );

				    	cout << "Iterations: " << iter << endl << endl;

				    	cout << "Cluster centers:" << endl << endl;

				    	for (int i = 0; i < p->numClusters; i++) {
				    		for (int j = 0; j < p->numDimensions; j++) {
				    			cout << p->centers[j + i * p->numDimensions] << " ";
				    			}
				    		cout << endl;
				    		}
				        break;


				    //minimum total volume
				    case 2:
				    	do {
				    				// Make a copy of all the membership functions.
				    				memcpy( params->membership2, params->membership, sizeof(float) * params->numClusters * params->numEvents );


				    				// Step 2a:
				    				// Estimate the cluster centers:
				    				setCenters( params );

				    				// Step 2b:
				    				// Set Scatter matrix according to equation 17 (covariance matrix):
				    				setScatterMatrices( params );

				    				// Step 3:
				    				// Initialize the set Ti = none.
				    				// This is the set of points to ignore because they had a membership
				    				// value that was negative.
				    				// Compute the Ti's, and iterate as long as any Ti != 0.
				    				memset( params->Ti, (int) 0, params->numEvents );

				    				int any_Ti_set = 0;
				    				do {
				    					any_Ti_set = 0;

				    					// Evaluate the membership functions according to EQ 31.
				    					// If any membership is negative, range clip it to zero.
				    					// 		AND set params->Ti( that point ) to 1.
				    					//		AND re-calculate the memberships according to EQ 31.
				    					//      EQ 31 that is here...
				    					beta = 1/pDimensions;
				    					tau = 0;
				    					computeGeneralFormula(p, beta, tau, THETA);
				    					// Total up all values of "any_Ti_set".
				    					// If the sum is non-zero, then one of them is set,
				    					// and you need to keep looping.
				    				    for (int ii=0; ii < params->numEvents; ii++ )
				    						any_Ti_set = any_Ti_set + params->Ti(ii);
				    				    iter++;

				                    } while (any_Ti_set != 0 && iter <= MAXITER);
				    				if (iter == MAXITER) {
				    						cout << "Program was unable to converge using the threshold " << p->threshold << endl;
				    						break;
				    					}

				                    // Step 4:
				                    // Compare the membership functions to previous methods.
				    				max_epsilon_change = 0;
				                    for (int idx=0;idx<=params->numClusters * params->numEvents; idx++)
				                    	{
				                    	float difference = params->membership(idx) - params->membership2(idx);
				                    	if (difference < 0)
				                    		difference = -difference;		// Form absolute value of difference.
				                    	if (difference > max_epsilon_change)
				                    		max_epsilon_change = difference;
				                    	}
				        } while (max_epsilon_change >= params->threshold );

				    	cout << "Iterations: " << iter << endl << endl;

				    	cout << "Cluster centers:" << endl << endl;

				    	for (int i = 0; i < p->numClusters; i++) {
				    		for (int j = 0; j < p->numDimensions; j++) {
				    			cout << p->centers[j + i * p->numDimensions] << " ";
				    			}
				    		cout << endl;
				    		}
				        break;


				    //sum of all normalized determinants (SAND)
				    case 3:
				    	do {
				    				// Make a copy of all the membership functions.
				    				memcpy( params->membership2, params->membership, sizeof(float) * params->numClusters * params->numEvents );


				    				// Step 2a:
				    				// Estimate the cluster centers:
				    				setCenters( params );

				    				// Step 2b:
				    				// Set Scatter matrix according to equation 17 (covariance matrix):
				    				setScatterMatrices( params );

				    				// Step 3:
				    				// Initialize the set Ti = none.
				    				// This is the set of points to ignore because they had a membership
				    				// value that was negative.
				    				// Compute the Ti's, and iterate as long as any Ti != 0.
				    				memset( params->Ti, (int) 0, params->numEvents );

				    				int any_Ti_set = 0;
				    				do {
				    					any_Ti_set = 0;

				    					// Evaluate the membership functions according to EQ 31.
				    					// If any membership is negative, range clip it to zero.
				    					// 		AND set params->Ti( that point ) to 1.
				    					//		AND re-calculate the memberships according to EQ 31.
				    					//      EQ 31 that is here...
				    					beta = 1/pDimensions;
				    					tau = 0;
				    					computeGeneralFormula(p, beta, tau, THETA);
				    					// Total up all values of "any_Ti_set".
				    					// If the sum is non-zero, then one of them is set,
				    					// and you need to keep looping.
				    					for (int ii=0; ii < params->numEvents; ii++ )
				    						any_Ti_set = any_Ti_set + params->Ti(ii);
				    					iter++;

				                    } while (any_Ti_set != 0 && iter <= MAXITER);
				    				if (iter == MAXITER) {
				    						cout << "Program was unable to converge using the threshold " << p->threshold << endl;
				    						break;
				    					}

				                    // Step 4:
				                    // Compare the membership functions to previous methods.
				    				max_epsilon_change = 0;
				                    for (int idx=0;idx<=params->numClusters * params->numEvents; idx++)
				                    	{
				                    	float difference = params->membership(idx) - params->membership2(idx);
				                    	if (difference < 0)
				                    		difference = -difference;		// Form absolute value of difference.
				                    	if (difference > max_epsilon_change)
				                    		max_epsilon_change = difference;
				                    	}
				        } while (max_epsilon_change >= params->threshold );

				    	cout << "Iterations: " << iter << endl << endl;

				    	cout << "Cluster centers:" << endl << endl;

				    	for (int i = 0; i < p->numClusters; i++) {
				    		for (int j = 0; j < p->numDimensions; j++) {
				    			cout << p->centers[j + i * p->numDimensions] << " ";
				    			}
				    		cout << endl;
				    		}
				        break;


				    //maxmum likelihood method
				    case 4:
				    	do {
				    				// Make a copy of all the membership functions.
				    				memcpy( params->membership2, params->membership, sizeof(float) * params->numClusters * params->numEvents );


				    				// Step 2a:
				    				// Estimate the cluster centers:
				    				setCenters( params );

				    				// Step 2b:
				    				// Set Scatter matrix according to equation 17 (covariance matrix):
				    				setScatterMatrices( params );

				    				// Step 3:
				    				// Initialize the set Ti = none.
				    				// This is the set of points to ignore because they had a membership
				    				// value that was negative.
				    				// Compute the Ti's, and iterate as long as any Ti != 0.
				    				memset( params->Ti, (int) 0, params->numEvents );

				    				int any_Ti_set = 0;
				    				do {
				    					any_Ti_set = 0;

				    					// Evaluate the membership functions according to EQ 31.
				    					// If any membership is negative, range clip it to zero.
				    					// 		AND set params->Ti( that point ) to 1.
				    					//		AND re-calculate the memberships according to EQ 31.
				    					//      EQ 31 that is here...
				                        computeMaxLikelihood(p);
				                        // Total up all values of "any_Ti_set".
				                        // If the sum is non-zero, then one of them is set,
				                        // and you need to keep looping.
				                        for (int ii=0; ii < params->numEvents; ii++ )
				                        	any_Ti_set = any_Ti_set + params->Ti(ii);
				                        iter++;
				                    } while (any_Ti_set != 0 && iter <= MAXITER);
				    				if (iter == MAXITER) {
				    						cout << "Program was unable to converge using the threshold " << p->threshold << endl;
				    						break;
				    					}

				                    // Step 4:
				                    // Compare the membership functions to previous methods.
				    				max_epsilon_change = 0;
				                    for (int idx=0;idx<=params->numClusters * params->numEvents; idx++)
				                    	{
				                    	float difference = params->membership(idx) - params->membership2(idx);
				                    	if (difference < 0)
				                    		difference = -difference;		// Form absolute value of difference.
				                    	if (difference > max_epsilon_change)
				                    		max_epsilon_change = difference;
				                    	}
				        } while (max_epsilon_change >= params->threshold );

				    	cout << "Iterations: " << iter << endl << endl;

				    	cout << "Cluster centers:" << endl << endl;

				    	for (int i = 0; i < p->numClusters; i++) {
				    		for (int j = 0; j < p->numDimensions; j++) {
				    			cout << p->centers[j + i * p->numDimensions] << " ";
				    			}
				    		cout << endl;
				    		}
				        break;

				    // catch all for bad option
				    default: cout << "Invalid option selected" << endl;
				    }


		writeData(params, "output.dat");

		free(params->centers);
		free(params->data);
		free(params->membership);
		free(params);

		return EXIT_SUCCESS;
	}
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
