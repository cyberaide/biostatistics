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
__device__ int compare(const void *d_a, int len_a, const void *d_b, int len_b)
{
	char* word1 = (char*)d_a;
	char* word2 = (char*)d_b;

	for (; *word1 != '\0' && *word2 != '\0' && *word1 == *word2; word1++, word2++);
	if (*word1 > *word2) return 1;
	if (*word1 < *word2) return -1;

	return 0;
}


int compare_host(const void *d_a, int len_a, const void *d_b, int len_b)
{
	char* word1 = (char*)d_a;
	char* word2 = (char*)d_b;

	for (; *word1 != '\0' && *word2 != '\0' && *word1 == *word2; word1++, word2++);
	if (*word1 > *word2) return 1;
	if (*word1 < *word2) return -1;

	return 0;
}


#endif //__COMPARE_CU__
