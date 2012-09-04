/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: Global.h 
	Time: 2012-07-01 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */

#ifndef __GLOBAL_H__
#define __GLOBAL_H__

typedef struct
{
	char* file; 
} WC_KEY_T;

typedef __align__(16) struct
{
	int line_offset;
	int line_size;
} WC_VAL_T;

#endif
