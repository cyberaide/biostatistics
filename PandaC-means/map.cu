/*	
Copyright 2012 The Trustees of Indiana University.  All rights reserved.
CGL MapReduce Framework on GPUs and CPUs
Code Name: Panda 0.1
File: map.cu 
Time: 2012-07-01 
Developer: Hui Li (lihui@indiana.edu)

This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
*/

#ifndef __MAP_CU__
#define __MAP_CU__

#include "Panda.h"
#include "Global.h"

/*
__device__ int hash_func(char* str, int len)
{
int hash, i;
for (i = 0, hash=len; i < len; i++)
hash = (hash<<4)^(hash>>28)^str[i];
return hash;
}
*/

__device__ void map2(void *key, void *val, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){

	KM_KEY_T* pKey = (KM_KEY_T*)key;
	KM_VAL_T* pVal = (KM_VAL_T*)val;
	
	int dim = pKey->dim;
	int K = pKey->K;
	int start = pKey->start;
	int end = pKey->end;
	int index = pKey->i;

	float *point = pVal->d_Points;
	float* cluster = pVal->d_Clusters;

	float * tempClusters = pVal->d_tempClusters+index*dim*K;
	float * tempDenominators = pVal->d_tempDenominators+index*K;

	float denominator = 0.0f;
	float membershipValue = 0.0f;
	float *distances = (float *)malloc(sizeof(float)*K);
	float *numerator = (float *)malloc(sizeof(float)*K);

	for(int i=0;i<K;i++){
		distances[i]=0.0f;
		numerator[i]=0.0f;
	}//for

	for (int i=start;i<end;i++){
		float* curPoint = point + i*dim;
		for (int k = 0; k < K; ++k)
		{
			float* curCluster = cluster + k*dim;
			distances[k] = 0.0;
			for (int j = 0; j < dim; ++j)
			{
				float pt = curPoint[j];
				float cl = curCluster[j];
				float delta = pt - cl;
				distances[k] += (delta * delta);
			}//for
			numerator[k]=powf(distances[k],2.0f/(2.0-1.0))+1e-30;
			denominator = denominator + 1.0f/(numerator[k]+1e-30);
		}//for

		for (int k = 0; k < K; ++k)
		{
			membershipValue = 1.0f/powf(numerator[k]*denominator,(float)2.0);
			for(int d =0;d<dim;d++)
				tempClusters[k*dim+d]+= curPoint[d]*membershipValue;
			tempDenominators[k]+= membershipValue;
		}//for 

	}//for

	free(distances);
	free(numerator);

	pKey->i = 0;
	pKey->end = 0;
	pKey->start = 0;
	pKey->point_id = 0;
	//pKey->

	EmitIntermediate2(key, val, sizeof(KM_KEY_T), sizeof(KM_VAL_T), d_g_state, map_task_idx);
	
}//map2


#endif //__MAP_CU__