/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: compare.cu 
	Time: 2012-07-01 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */

#ifndef __COMPARE_CU__
#define __COMPARE_CU__
#include "Panda.h"
#include "Global.h"

//-----------------------------------------------------------
//No Sort in this application
//-----------------------------------------------------------

__device__ int compare(const void *key_a, int len_a, const void *key_b, int len_b)
{
	KM_KEY_T *ka = (KM_KEY_T*)key_a;
	KM_KEY_T *kb = (KM_KEY_T*)key_b;

	if (ka->i > kb->i)
		return 1;

	if (ka->i > kb->i)
		return -1;

	if (ka->i == kb->i)
		return 0;

}



__device__ int default_compare(const void *key_a, int len_a, const void *key_b, int len_b)
{
	
	if (len_a>len_b)
		return 1;
	else if (len_a<len_b)
		return -1;

	for (int i=0;i<len_a;i++){
		/*
		if ((char*)(key_a)[i]>(char*)(key_b)[i])
			return 1;
		if ((char*)(key_a)[i]<(char*)(key_b)[i])
			return -1;
			*/
	}
	return 0;

}



#endif //__COMPARE_CU__
