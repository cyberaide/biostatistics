/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: reduce.cu 
	Time: 2012-07-01 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */

#ifndef __REDUCE_CU__
#define __REDUCE_CU__

#include "Panda.h"

//-------------------------------------------------------------------------
//Reduce Function in this application
//-------------------------------------------------------------------------


__device__ void reduce2(void *key, val_t* vals, int keySize, int valCount, gpu_context d_g_state)
{
		//printf("valCount:%d\n",valCount);
		KM_KEY_T* pKey = (KM_KEY_T*)key;
        //KM_VAL_T* pVal = (KM_VAL_T*)vals;
        int dim = pKey->dim;
        int K = pKey->K;
				
        float* myClusters = (float*) malloc(sizeof(float)*dim*K);
        float* myDenominators = (float*) malloc(sizeof(float)*K);
        memset(myClusters,0,sizeof(float)*dim*K);
        memset(myDenominators,0,sizeof(float)*K);

        float *tempClusters = NULL;
        float *tempDenominators = NULL;
        for (int i = 0; i < valCount; i++)
        {
                int index = pKey->i;
				KM_VAL_T* pVal = (KM_VAL_T*)(vals[i].val);
                tempClusters = pVal->d_tempClusters + index*K*dim;
                tempDenominators = pVal->d_tempDenominators+ index*K;
                for (int k = 0; k< K; k++){
                        for (int j = 0; j< dim; j++)
                                myClusters[k*dim+j] += tempClusters[k*dim+j];
                        myDenominators[k] += tempDenominators[k];
                }//for
        }//end for

        for (int k = 0; k< K; k++){
			for (int i = 0; i < dim; i++){
                        myClusters[i] /= (float)myDenominators[i];
						//printf("%f ",myClusters[i]);
			}//for
			//printf("\n");
        }//for

		//printf("TID reduce2:%d\n",TID);
		Emit2(key,vals,sizeof(KM_KEY_T), sizeof(KM_VAL_T), &d_g_state);
		
		free(myClusters);
		free(myDenominators);
				
}//reduce2


#endif //__REDUCE_CU__
